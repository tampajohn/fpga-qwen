#!/usr/bin/env python3
"""Original CPU engine against frozen tuned llama FP16; shared gates and paired schema."""
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


class OriginalEngine:
    def __init__(self, protocol, binary, weights, threads, prefill_threads, logdir):
        self.client = NativeCPUChat(protocol, binary, weights, threads, prefill_threads)
        self.protocol, self.logdir = protocol, logdir
        self.cancel = threading.Event()
        self.metadata = dict(transport='Persistent subprocess JSON lines; one step request per output',
            weight_storage='FP16 matrices / FP32 norms', kv='FP16', arithmetic='Original C++ FP32 math',
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
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT/'build/benchmark-status')
    args = parser.parse_args()
    rounds = args.rounds if args.rounds is not None else 10 if args.mode == 'confirmation' else 2
    if (args.mode == 'confirmation' and not 10 <= rounds <= 100) or (args.mode == 'discovery' and not 1 <= rounds <= 3):
        parser.error('Confirmation requires10..100rounds; discovery1..3')
    if not 1 <= args.threads <= 64 or not 1 <= args.prefill_threads <= 64:
        parser.error('Thread counts must be1..64')
    if not 1 <= args.warmups <= 3:
        parser.error('Warmups must be1..3 per visit/profile')
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
    reference_binary = args.reference_binary.resolve() if args.reference_binary else None
    if reference_binary == binary:
        parser.error('Reference and candidate must be distinct immutable binaries')
    build = None if reference_binary else freeze_build(args.baseline_build.resolve())
    configs = [dict(name='llama-f16-onon-d12-p6', build_dir=str(args.baseline_build.resolve()),
                    decode_threads=12, prefill_threads=6, priority=0, poll=50),
               dict(name=f'native-cpu-f16-d{args.threads}-p{args.prefill_threads}',
                    decode_threads=args.threads, prefill_threads=args.prefill_threads,
                    binary=str(binary), weights=str(weights))]
    if reference_binary:
        configs = [dict(name=f'native-reference-{reference_binary.parent.name}', binary=str(reference_binary),
                        weights=str(weights), decode_threads=args.threads, prefill_threads=args.prefill_threads),
                   dict(name=f'native-candidate-{binary.parent.name}', binary=str(binary),
                        weights=str(weights), decode_threads=args.threads, prefill_threads=args.prefill_threads)]
    native_binaries = [reference_binary, binary] if reference_binary else [binary]
    source_paths = [Path(__file__).resolve()] + [ROOT/'tools'/name for name in (
        'native_cpu_chat.py','benchmark_native_phase0.py','benchmark_matched.py',
        'check_native_phase0_quality.py','conversation_eval.py','chat_protocol.py','benchmark_guard.py','experiment.py')]
    input_paths = [*native_binaries, weights, Path(str(weights)+'.index'), Path(str(weights)+'.manifest.json')]
    if not reference_binary: input_paths.append(model)
    input_paths += [checkpoint/name for name in ('tokenizer.json','tokenizer_config.json','config.json','model.safetensors')]
    for executable in native_binaries:
        input_paths.append(executable.parent/'manifest.json')
    report = dict(completed=False, mode=args.mode, precision='f16', gpu=False, pytorch=False,
        performance_claim_eligible=False, numerical_quality_review_required=True, paired_sampling_complete=False,
        timing_claim=False, configurations=configs, rounds=rounds, tokens=32, warmups_per_visit_profile=args.warmups if args.mode!='quality' else 0,
        comparison_scope='Original engine optimization comparison; not a llama target claim' if reference_binary else 'Original engine versus tuned llama FP16',
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
            'Both engines use identical subprocess transport, weights, and requested thread counts; reset/mode setup is outside the clock.',
            'Persistent prefill and optional auxiliary caches can differ; actual startup metadata records cache bytes when available.',
            'This direct comparison isolates a candidate configuration against an original-engine reference, not achievement of the tuned llama target.']
    def save():
        pending = args.output.with_suffix('.pending.json')
        pending.write_text(json.dumps(report,indent=2)+'\n'); pending.replace(args.output)
    def factory(config, path):
        if not reference_binary and config is configs[0]: return Phase0NativeEngine('f16',config,build,model,path)
        return OriginalEngine(protocol,Path(config['binary']),weights,args.threads,args.prefill_threads,path)
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
            prompts = {name:make_prompt(protocol,name) for name in ('short','long')}
            if [len(prompts[p]) for p in ('short','long')] != [85,452]:
                raise RuntimeError('Historical prompt lengths changed')
            report['prompts'] = {name:dict(ids=ids,count=len(ids)) for name,ids in prompts.items()}
            expected = {}
            def visit(config,phase,round_index=None,order_index=None):
                guard(args.guard_state_dir)
                tag = phase if round_index is None else f'round-{round_index:03d}'
                record = dict(config=config['name'],phase=phase,round=round_index,order_index=order_index,profiles={})
                report['preflight' if phase=='preflight' else 'visits'].append(record)
                engine = factory(config,logroot/tag/config['name'])
                try:
                    record['startup_seconds'] = engine.start()
                    record.update(command=engine.command,pid=engine.process.pid)
                    for profile,ids in prompts.items():
                        row = dict(warmups=[]);record['profiles'][profile]=row
                        for _ in range(args.warmups):
                            warm = engine.generate(ids,32,True)
                            validate_result(warm,ids,32,expected.get(profile))
                            expected.setdefault(profile,warm['output_token_ids']);row['warmups'].append(warm)
                        if phase!='preflight':
                            scored = engine.generate(ids,32,True)
                            validate_result(scored,ids,32,expected[profile]);row['scored']=scored
                finally: finish(engine,record)
            for config in configs: visit(config,'preflight')
            report.update(preflight_same_precision_ids_passed=True, expected_output_ids=expected)
            for index in range(rounds):
                order = configs if index%2==0 else list(reversed(configs))
                for position,config in enumerate(order):visit(config,'scored',index,position)
            summarize(report)
            report['paired_sampling_complete'] = args.mode=='confirmation'
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
