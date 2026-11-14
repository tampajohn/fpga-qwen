#!/usr/bin/env python3
"""Original CPU engine against frozen tuned llama FP16; shared gates and paired schema."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools"))
import numpy as np
import argparse
import json
import os
from pathlib import Path
import shutil
import threading
import time

from benchmark_native_phase0 import (Phase0NativeEngine, freeze_build, guard, host_snapshot,
                                    sha, summarize, validate_result, verify_build)
from benchmark_matched import make_prompt, timing_result
from check_native_phase0_quality import validate_response, summarize as summarize_quality
from chat_protocol import ChatProtocol
from conversation_eval import CASES, evaluate_response, suite_fingerprint
from experiment import ROOT, locate_model
from native_cpu_chat import NativeCPUChat, EOS


def extended_prompts(protocol, count):
    """Tokenize complete messages; never inject padding into prepared token IDs."""
    result = {}
    for length in (1024, 2048, 3968):
        padding = 0
        for attempt in range(10):
            messages = [dict(role='user', content='Background notes:' + ' x' * padding +
                '\nIgnore the background notes. Write a detailed tutorial of at least 600 words '
                'explaining CPU memory bandwidth, cache levels, cache locality, and matrix multiplication. '
                'Use ten numbered paragraphs with concrete examples. Begin with paragraph 1.')]
            ids = protocol.prepare(dict(session='extended-context', messages=messages,
                max_new_tokens=1, decoding='greedy')).ids
            if len(ids) == length:
                if len(ids) + count > 4096:
                    raise RuntimeError('Extended workload exceeds context capacity')
                result[f'context{length}'] = dict(ids=ids, count=length, messages=messages)
                break
            padding += length - len(ids)
            if padding < 0:
                raise RuntimeError('Extended prompt template exceeds target length')
        else:
            raise RuntimeError('Unable to tokenize exact extended prompt length')
    return result


class OriginalEngine:
    def __init__(self, protocol, binary, weights, threads, prefill_threads, logdir):
        self.client = NativeCPUChat(protocol, binary, weights, threads, prefill_threads)
        self.protocol, self.logdir = protocol, logdir
        self.cancel = threading.Event()
        self.metadata = dict(transport='Persistent subprocess JSON lines; one step request per output',
            weight_storage='FP16 matrices / FP32 norm weights', kv='FP16', arithmetic='Original C++ arithmetic; accumulator precision recorded in actual startup',
            cpu_threads=threads, prefill_threads=prefill_threads, gpu=False, pytorch=False,
            prefill_weight_cache='Layer FP16 weights widened once to FP32 for Accelerate prefill; embedding stays FP16',
            placement='No hard pinning; VECLIB_MAXIMUM_THREADS requested, not independently verified')
        self.log_path = logdir / 'native.log'
        self.command = [str(binary), '--weights', str(weights), '--threads', str(threads),
                        '--prefill-threads', str(prefill_threads), '--context', '4096']

    def start(self):
        # Archive only bytes appended by this process to the adapter's shared log.
        self.logdir.mkdir(parents=True, exist_ok=False)
        source = self.client.binary.parent / 'chat.log'
        self.log_offset = source.stat().st_size if source.exists() else 0
        start = time.perf_counter()
        self.client.start(self.cancel)
        self.process = self.client.process
        self.metadata['actual_startup'] = self.client.ready
        return time.perf_counter()-start

    def close(self):
        source = self.client.binary.parent / 'chat.log'
        self.client.close()
        if source.is_file() and self.logdir.is_dir():
            with source.open('rb') as stream:
                stream.seek(getattr(self,'log_offset',0))
                self.log_path.write_bytes(stream.read())

    def capture_preflight(self,prompt,count,directory):
        # Outside scored clocks; retain every full-vocabulary dump for audit.
        directory.mkdir(parents=True,exist_ok=False)
        reset=self.client.command('reset',self.cancel)
        if reset.get('position')!=0 or not reset.get('ok'):raise RuntimeError('Preflight reset failed')
        if not self.client.command('mode greedy',self.cancel).get('ok'):raise RuntimeError('Preflight mode failed')
        response=self.client.command('prefill '+' '.join(map(str,prompt)),self.cancel)
        ids=[];hashes=[];dumps=[]
        for step in range(count):
            token=response.get('token');position=len(prompt)+step
            if type(token)is not int or not 0<=token<151936 or response.get('position')!=position:raise RuntimeError('Bad preflight position/token')
            path=directory/f'step-{step:03d}.f32'
            ack=self.client.command('dump '+str(path),self.cancel)
            if not ack.get('ok') or ack.get('count')!=151936 or ack.get('position')!=position or path.stat().st_size!=151936*4:raise RuntimeError('Bad preflight full-logit dump')
            values=np.fromfile(path,dtype='<f4')
            if values.shape!=(151936,) or not np.isfinite(values).all():raise RuntimeError('Nonfinite preflight logits')
            digest=sha(path);ids.append(token);hashes.append(digest);dumps.append(dict(path=str(path),sha256=digest,position=position,shape=[151936],dtype='float32',byte_order='little'))
            if token in EOS:raise RuntimeError('Fixed worker workload reached EOS')
            if step+1<count:response=self.client.command(f'step {token}',self.cancel)
        return dict(output_ids=ids,sha256=hashes,dumps=dumps,finish_reason='length',outside_scored_clock=True)

    def generate(self, prompt, count, fixed_count):
        reset = self.client.command('reset', self.cancel)
        if reset.get('ok') is not True or reset.get('position') != 0:
            raise RuntimeError('Native context reset failed')
        if self.client.command('mode greedy', self.cancel).get('ok') is not True:
            raise RuntimeError('Native greedy mode failed')
        output, arrivals, reason = [], [], 'length'
        start = time.perf_counter()
        response = self.client.command('prefill ' + ' '.join(map(str, prompt)), self.cancel)
        for index in range(count):
            token = response.get('token')
            if type(token) is not int or not 0 <= token < 151936 or response.get('position') != len(prompt)+index:
                raise RuntimeError('Invalid native inference response')
            # Preserve the actual prediction, including terminal EOS, exactly
            # as llama's return_tokens stream does. Text decoding hides specials.
            output.append(token); arrivals.append(time.perf_counter())
            if token in EOS:
                reason = 'eos'
                break
            if index+1 < count:
                response = self.client.command(f'step {token}', self.cancel)
        result = timing_result(start, arrivals, output, '', reason,
                               dict(cache_n=0, prompt_n=len(prompt)))
        result['text'] = self.protocol.tokenizer.decode(output, skip_special_tokens=True)
        result['eos_observed'] = reason == 'eos'
        if fixed_count and (reason == 'eos' or len(output) != count):
            raise RuntimeError('Native fixed workload reached EOS or changed output count')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--reference-binary', type=Path, help='Compare two original engines instead of llama; same weights/threads')
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--mode', choices=('quality', 'discovery', 'confirmation'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-build', type=Path, default=ROOT/'build/native-phase0/kleidi-on-lto-on')
    parser.add_argument('--model', type=Path, default=ROOT/'build/bench/qwen3-0.6b-f16.gguf')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--prefill-threads', type=int, default=6)
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--context-suite', choices=('standard', 'extended'), default='standard')
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT/'build/benchmark-status')
    args = parser.parse_args()
    if not 2 <= args.tokens <= 128:
        parser.error('Use 2..128 output tokens; extended prompts reserve 128 positions')
    if args.mode == 'quality' and (args.context_suite != 'standard' or args.tokens != 32):
        parser.error('Quality mode uses its own unchanged case limits')
    rounds = args.rounds if args.rounds is not None else 10 if args.mode == 'confirmation' else 2
    if (args.mode == 'confirmation' and not 10 <= rounds <= 100) or (args.mode == 'discovery' and not 1 <= rounds <= 3):
        parser.error('Confirmation requires10..100rounds; discovery1..3')
    if not 1 <= args.threads <= 64 or not 1 <= args.prefill_threads <= 64:
        parser.error('Thread counts must be1..64')
    if not 1 <= args.warmups <= 3:
        parser.error('Warmups must be1..3 per visit/profile')
    if args.mode!='discovery' or rounds!=3 or args.warmups!=1 or args.prefill_threads!=6 or args.threads!=12 or args.tokens!=128 or args.context_suite!='extended':
        parser.error('Held-out exploratory protocol requires discovery3pairs,1warmup,12/6,extended128')
    args.output = args.output.resolve()
    logroot = args.output.parent / (args.output.stem+'-logs')
    if args.output.exists() or logroot.exists():
        parser.error('Report/log path already exists; choose a fresh output')
    guarded = guard(args.guard_state_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logroot.mkdir()
    checkpoint = locate_model(args.checkpoint)
    protocol = ChatProtocol(checkpoint)
    binary, weights, model = args.binary.resolve(), args.weights.resolve(), args.model.resolve()
    reference_binary=(ROOT/'build/native-cpu/build-20/qwen_cpu').resolve()
    if args.reference_binary and args.reference_binary.resolve()!=reference_binary:parser.error('Reference must be frozen build20')
    if binary!=(ROOT/'build/native-cpu/decode-attention-active8-v2-build/qwen_cpu').resolve():parser.error('Expected fixed active8 v2 candidate')
    build = None if reference_binary else freeze_build(args.baseline_build.resolve())
    configs = [dict(name='llama-f16-onon-d12-p6', build_dir=str(args.baseline_build.resolve()),
                    decode_threads=12, prefill_threads=6, priority=0, poll=50),
               dict(name=f'native-cpu-f16-d{args.threads}-p{args.prefill_threads}',
                    decode_threads=args.threads, prefill_threads=args.prefill_threads,
                    binary=str(binary), weights=str(weights))]
    if reference_binary:
        configs = [dict(name='native-build20-d12-p6', binary=str(reference_binary),
                        weights=str(weights), decode_threads=12, prefill_threads=6),
                   dict(name='native-active8-attention-d12-p6', binary=str(binary),
                        weights=str(weights), decode_threads=args.threads, prefill_threads=args.prefill_threads)]
    native_binaries = [reference_binary, binary] if reference_binary else [binary]
    source_paths = [Path(__file__).resolve(), ROOT/'build/native-cpu/prepare-active8-heldout-fixtures-v1.py'] + [ROOT/'tools'/name for name in (
        'native_cpu_chat.py','benchmark_native_phase0.py','benchmark_matched.py',
        'check_native_phase0_quality.py','conversation_eval.py','chat_protocol.py','benchmark_guard.py','experiment.py','benchmark_native_cpu.py','suffix.py','analyze_native_phase0.py')]
    fixture_path=args.fixtures.resolve();fixture=json.loads(fixture_path.read_text())
    assert fixture['completed'] and fixture['lengths']==[192,768,1536,2560,3328] and fixture['tokens']==128 and fixture['rounds']==3 and fixture['seed']==42
    assert set(fixture['profiles'])=={'context'+str(n)for n in (192,768,1536,2560,3328)}
    assert all(row['count']==int(name.removeprefix('context')) and len(row['ids'])==row['count']for name,row in fixture['profiles'].items())
    for path,digest in fixture['source_sha256'].items():
        assert sha(Path(path))==digest;source_paths.append(Path(path))
    input_paths = [fixture_path,*native_binaries, weights, Path(str(weights)+'.index'), Path(str(weights)+'.manifest.json')]
    if not reference_binary: input_paths.append(model)
    input_paths += [checkpoint/name for name in ('tokenizer.json','tokenizer_config.json','config.json','model.safetensors')]
    for executable in set(native_binaries):
        input_paths.append(executable.parent/'manifest.json')
        for path,digest in json.loads((executable.parent/'manifest.json').read_text())['source_sha256'].items():
            if sha(Path(path))!=digest:raise RuntimeError('Frozen build source changed: '+path)
            source_paths.append(Path(path))
    report = dict(completed=False, mode=args.mode, precision='f16', gpu=False, pytorch=False,
        performance_claim_eligible=False, numerical_quality_review_required=True, paired_sampling_complete=False,
        timing_claim=False, configurations=configs, rounds=rounds, tokens=args.tokens,
        context_suite=args.context_suite, warmups_per_visit_profile=args.warmups if args.mode!='quality' else 0,
        comparison_scope='Build20 versus active8 attention scheduling, same physical12/6; exploratory only, no threshold selection' if reference_binary else 'Original engine versus tuned llama FP16',
        builds={} if reference_binary else {configs[0]['build_dir']:build},
        model=dict(path=str(weights if reference_binary else model),sha256=sha(weights if reference_binary else model)),
        guard_before=guarded, host_before=host_snapshot(), preflight=[], visits=[], results=[], sessions=[],
        limitations=['Llama uses HTTP streaming; original engine uses subprocess commands. End-to-end transport differs.',
            'Native reset and greedy-mode setup are outside the clock; llama request includes cache-disabled request processing.',
            'Original layer weights also have a persistent FP32 widened prefill cache; resident memory is not equivalent.',
            'One process is live per visit; startup and warmup excluded. Background OS work and address placement uncontrolled.',
            'All scores require separate numerical/quality review. Discovery is exploratory, never a performance claim.'])
    if reference_binary:
        report['limitations'][:3] = [
            'Both arms use same weights/transport and physical12/6. Only candidate attention useful-work partition differs; no affinity/core-placement assumptions.',
            'Persistent prefill and optional auxiliary caches can differ; actual startup metadata records cache bytes when available.',
            'This direct comparison isolates a candidate configuration against an original-engine reference, not achievement of the tuned llama target.']
    def save():
        pending = args.output.with_suffix('.pending.json')
        pending.write_text(json.dumps(report,indent=2)+'\n'); pending.replace(args.output)
    def factory(config, path):
        if not reference_binary and config is configs[0]: return Phase0NativeEngine('f16',config,build,model,path)
        return OriginalEngine(protocol,Path(config['binary']),weights,config['decode_threads'],config['prefill_threads'],path)
    def finish(engine, record):
        engine.close()
        record['metadata'] = engine.metadata
        if hasattr(engine,'log_path') and engine.log_path.is_file():
            record['runtime_log'] = dict(path=str(engine.log_path),sha256=sha(engine.log_path))
        save()
    try:
        save()
        report['inputs'] = {str(path):sha(path) for path in input_paths}
        report['source_hashes'] = {str(path):sha(path) for path in source_paths}
        manifest = json.loads(Path(str(weights)+'.manifest.json').read_text())
        if (manifest.get('completed') is not True or manifest.get('format') != 'QWCPU01'
                or manifest['binary']['sha256'] != report['inputs'][str(weights)]
                or manifest['index']['sha256'] != report['inputs'][str(weights)+'.index']):
            raise RuntimeError('Weight pack manifest/hash mismatch')
        report['native_builds'] = {}
        for executable in native_binaries:
            native_manifest_path = executable.parent/'manifest.json'
            native_manifest = json.loads(native_manifest_path.read_text())
            if native_manifest.get('completed') is not True or native_manifest.get('binary_sha256') != report['inputs'][str(executable)]:
                raise RuntimeError('Native build manifest/binary mismatch: ' + str(executable))
            report['native_builds'][str(executable)] = native_manifest
        report['native_build'] = report['native_builds'][str(binary)]
        report['suite_sha256'] = suite_fingerprint()
        if args.mode == 'quality':
            prompts = {case.name:protocol.prepare(dict(session='native-quality', messages=list(case.messages),
                       max_new_tokens=case.max_new_tokens, decoding='greedy')).ids for case in CASES}
            report.update(reference_configuration=configs[0]['name'], suite=[case.name for case in CASES])
            for config in configs:
                guard(args.guard_state_dir)
                record = dict(configuration=config['name']);report['sessions'].append(record)
                engine = factory(config,logroot/'quality'/config['name'])
                try:
                    engine.start()
                    record.update(command=engine.command,pid=engine.process.pid)
                    for case in CASES:
                        raw = engine.generate(prompts[case.name],case.max_new_tokens,False)
                        result = validate_response(raw,prompts[case.name],case.max_new_tokens)
                        report['results'].append(dict(configuration=config['name'],case=case.name,
                            category=case.category,result=result,grade=evaluate_response(case,result['text'])))
                        save()
                finally: finish(engine,record)
            summarize_quality(report)
        else:
            if args.context_suite == 'extended':
                report['prompts'] = fixture['profiles']
                for name,row in report['prompts'].items():
                    actual=protocol.prepare(dict(session='active8-heldout',messages=row['messages'],max_new_tokens=128,decoding='greedy')).ids
                    assert actual==row['ids'] and len(actual)==row['count'] and name=='context'+str(row['count'])
                prompts = {name: row['ids'] for name, row in report['prompts'].items()}
            else:
                prompts = {name:make_prompt(protocol,name) for name in ('short','long')}
                if [len(prompts[p]) for p in ('short','long')] != [85,452]:
                    raise RuntimeError('Historical prompt lengths changed')
                report['prompts'] = {name:dict(ids=ids,count=len(ids)) for name,ids in prompts.items()}
            expected = {}
            expected_logits = {}
            def visit(config,phase,round_index=None,order_index=None):
                guard(args.guard_state_dir)
                tag = phase if round_index is None else f'round-{round_index:03d}'
                record = dict(config=config['name'],phase=phase,round=round_index,order_index=order_index,profiles={})
                report['preflight' if phase=='preflight' else 'visits'].append(record)
                engine = factory(config,logroot/tag/config['name'])
                try:
                    record['startup_seconds'] = engine.start()
                    actual=engine.client.ready
                    if actual.get('decode_threads')!=config['decode_threads'] or actual.get('prefill_threads')!=6 or actual.get('gpu') is not False or actual.get('pytorch') is not False:raise RuntimeError('Worker startup mismatch')
                    record.update(command=engine.command,pid=engine.process.pid)
                    for profile,ids in prompts.items():
                        row = dict(warmups=[]);record['profiles'][profile]=row
                        for _ in range(args.warmups):
                            warm = engine.generate(ids,args.tokens,True)
                            validate_result(warm,ids,args.tokens,expected.get(profile))
                            expected.setdefault(profile,warm['output_token_ids']);row['warmups'].append(warm)
                        if phase=='preflight':
                            capture=engine.capture_preflight(ids,args.tokens,engine.logdir/profile)
                            if capture['output_ids']!=expected[profile]:raise RuntimeError('Full-logit preflight IDs changed')
                            previous=expected_logits.setdefault(profile,capture['sha256'])
                            if previous!=capture['sha256']:raise RuntimeError('Full-logit worker-count identity failed')
                            row['full_logit_preflight']=capture
                        if phase!='preflight':
                            scored = engine.generate(ids,args.tokens,True)
                            validate_result(scored,ids,args.tokens,expected[profile]);row['scored']=scored
                finally: finish(engine,record)
            for config in configs: visit(config,'preflight')
            report.update(preflight_same_precision_ids_passed=True, preflight_full_logits_bit_identical=True, expected_output_ids=expected, expected_logit_sha256=expected_logits)
            for index in range(rounds):
                order = configs if index%2==0 else list(reversed(configs))
                for position,config in enumerate(order):visit(config,'scored',index,position)
            summarize(report)
            report['paired_sampling_complete'] = args.mode=='confirmation'
        for visit in report['preflight']:
            for row in visit['profiles'].values():
                for dump in row['full_logit_preflight']['dumps']:
                    if sha(Path(dump['path']))!=dump['sha256']:raise RuntimeError('Preflight dump changed')
        if build is not None: verify_build(build)
        if report['inputs'] != {str(path):sha(path) for path in input_paths} or report['source_hashes'] != {str(path):sha(path) for path in source_paths} or report['suite_sha256'] != suite_fingerprint():
            raise RuntimeError('Inputs, sources or quality suite changed during comparison')
        report.update(guard_after=guard(args.guard_state_dir),host_after=host_snapshot(),completed=True)
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error));raise
    finally:save()
    if args.mode=='quality' and report.get('passed') is not True:raise SystemExit(1)
    print(args.output)


if __name__=='__main__':main()
