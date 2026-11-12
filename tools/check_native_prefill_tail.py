#!/usr/bin/env python3
"""Isolate final-tail prefill versus teacher-forced B1 steps; no timing claim."""
import argparse
import array
import json
import math
import os
from pathlib import Path
import sys
import threading

from benchmark_native_phase0 import guard, sha
from experiment import ROOT
from native_cpu_chat import NativeCPUChat


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def save(path, report):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT/'build/native-cpu/build-15/qwen_cpu')
    parser.add_argument('--weights', type=Path, default=ROOT/'build/native-cpu/weights.bin')
    parser.add_argument('--prompt', type=Path, default=ROOT/'build/native-cpu/extended-tutorial-fixtures/context3968.json')
    parser.add_argument('--continuation', type=Path, default=ROOT/'build/native-cpu/extended-tutorial-fixtures/context3968-continuation1.json')
    parser.add_argument('--reference-capture', type=Path, default=ROOT/'build/native-cpu/build-12-context3968-logits/report.json')
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT/'build/benchmark-status')
    args = parser.parse_args()
    guarded = guard(args.guard_state_dir)
    directory = args.directory.resolve();directory.mkdir(parents=True, exist_ok=False)
    output = directory/'report.json'
    binary, weights = args.binary.resolve(), args.weights.resolve()
    report = dict(completed=False, diagnostic_only=True, timing_claim=False,
        performance_claim_eligible=False, gpu=False, pytorch=False, guard_before=guarded,
        prefix_tokens=3584, tail_tokens=384, compared_contexts=[3968,3969], branches=[],
        scope='Diagnostic build15 (hidden-dump capable, not live build12) primes canonical3584 once, then final384 via one prefill versus384 explicit teacher-forced steps after rewind. Same fixed next input in both branches; A must match frozen12 full logits.',
        limitations=['Not an independent oracle gate or a proof of numerical cause.',
                     'BranchB changes final-tail projection batching; earlier3584-token KV remains the identical shared prefill state.',
                     'Intermediate predictions do not select inputs. Every tail/continuation input is fixed by frozen fixtures.',
                     'BaselineA full logits must exactly match both existing baseline capture repeats. Hidden vectors are newly captured diagnostics.'])
    client = None
    removed = {}
    try:
        for key in list(os.environ):
            if key.startswith(('DYLD_', 'GGML_', 'LLAMA_ARG_')) or key == 'VECLIB_MAXIMUM_THREADS':
                removed[key] = os.environ.pop(key)
        report['removed_environment_names'] = sorted(removed)
        ids = json.loads(args.prompt.read_text());teacher = json.loads(args.continuation.read_text())
        require(isinstance(ids,list) and len(ids)==3968 and isinstance(teacher,list) and len(teacher)==1
                and all(type(x)is int and 0<=x<151936 for x in ids+teacher), 'Expected3968 prompt IDs and one fixed continuation ID')
        paths = [binary,binary.parent/'manifest.json',weights,Path(str(weights)+'.index'),Path(str(weights)+'.manifest.json'),
                 args.prompt.resolve(),args.continuation.resolve(),args.reference_capture.resolve()]
        report['inputs'] = {str(path):sha(path) for path in paths}
        sources = [Path(__file__).resolve()]+[ROOT/'tools'/name for name in ('native_cpu_chat.py','suffix.py','benchmark_native_phase0.py','benchmark_guard.py','experiment.py')]
        report['source_hashes'] = {str(path):sha(path) for path in sources}
        build = json.loads((binary.parent/'manifest.json').read_text())
        pack = json.loads(Path(str(weights)+'.manifest.json').read_text())
        require(build.get('completed')is True and build.get('gpu')is False and build.get('pytorch')is False
                and build.get('binary_sha256')==sha(binary), 'Build provenance mismatch')
        require(pack.get('completed')is True and pack.get('format')=='QWCPU01'
                and pack['binary']['sha256']==sha(weights)
                and pack['index']['sha256']==sha(Path(str(weights)+'.index')), 'Pack provenance mismatch')
        old = json.loads(args.reference_capture.read_text())
        require(old.get('completed')is True and old.get('gpu')is False and old.get('pytorch')is False
                and old.get('prompt_ids')==ids and old.get('requested_steps')==2 and len(old.get('runs',[]))==2,
                'Reference trajectory mismatch')
        for path in (weights,Path(str(weights)+'.index'),Path(str(weights)+'.manifest.json')):
            require(old['inputs'].get(str(path))==sha(path), 'Reference input scope mismatch')
        # Older capture schema records its executable in inputs, not configuration.
        reference_binaries = [Path(path).resolve() for path in old['inputs'] if Path(path).name == 'qwen_cpu']
        require(len(reference_binaries)==1, 'Ambiguous reference executable provenance')
        reference_binary = reference_binaries[0]
        require(old['inputs'][str(reference_binary)]==sha(reference_binary), 'Frozen reference binary changed')
        reference_manifest_path = reference_binary.parent/'manifest.json'
        reference_build = json.loads(reference_manifest_path.read_text())
        require(reference_build.get('completed')is True and reference_build.get('gpu')is False
                and reference_build.get('pytorch')is False
                and reference_build.get('binary_sha256')==sha(reference_binary), 'Reference build provenance mismatch')
        if str(reference_manifest_path) in old['inputs']:
            require(old['inputs'][str(reference_manifest_path)]==sha(reference_manifest_path), 'Captured reference manifest changed')
        for path in (reference_binary,reference_manifest_path):
            report['inputs'][str(path)]=sha(path)
        report['reference_binary']=dict(path=str(reference_binary),sha256=sha(reference_binary),
            manifest_path=str(reference_manifest_path),manifest_sha256=sha(reference_manifest_path),
            manifest_recorded_in_original_capture=str(reference_manifest_path)in old['inputs'])
        report['diagnostic_binary']=dict(path=str(binary),sha256=sha(binary),
            role='Hidden-dump capable diagnostic; not a claim that reference and diagnostic executables are identical')
        reference = {}
        for run in old['runs']:
            require(run.get('consumed_continuation_ids')==teacher
                    and run.get('own_repeat_logits_bit_identical')is True, 'Reference teacher/repeat mismatch')
            for step in (0,1):
                dumps=[r for r in run['logit_dumps']if r['step']==step]
                require(len(dumps)==1,'Missing reference dump')
                row=dumps[0];path=Path(row['path']).resolve()
                require(row['context_tokens']==3968+step and row['sha256']==sha(path)
                        and path.stat().st_size==151936*4,'Reference dump mismatch')
                report['inputs'][str(path)]=sha(path)
                reference.setdefault(step,row['sha256'])
                require(reference[step]==row['sha256'],'Reference repeat bytes differ')
        report.update(native_build=build,prompt_ids=ids,teacher_continuation_ids=teacher)
        save(output,report)
        client=NativeCPUChat(None,binary=binary,weights=weights,threads=12,prefill_threads=6,reuse_prefix=False)
        client.start(threading.Event());require(client.prefix_reuse_supported(),'Canonical rewind capability missing')
        report['ready']=client.ready
        ack=client.command('reset',threading.Event());require(ack.get('ok')is True and ack.get('position')==0,'Reset failed')
        require(client.command('mode greedy',threading.Event()).get('ok')is True,'Mode failed')
        primed=client.command('prefill '+' '.join(map(str,ids[:3584])),threading.Event())
        require(primed.get('position')==3584,'Prefix priming failed')
        report['canonical_prefix_primed_once']=True
        def capture(branch,step,response):
            require(response.get('position')==3968+step and type(response.get('token'))is int
                    and 0<=response['token']<151936,'Capture position/token mismatch')
            row=dict(step=step,context_tokens=3968+step,predicted_id=response['token'],arrays={})
            for kind,op,count in (('logits','dump',151936),('hidden','dump-hidden',1024)):
                path=directory/f'{branch}-step{step}-{kind}.f32';require(not path.exists(),'Dump path exists')
                ack=client.command(f'{op} {path}',threading.Event());require(ack.get('ok')is True and ack.get('count')==count and ack.get('position')==3968+step,'Dump acknowledgement mismatch')
                data=path.read_bytes();values=array.array('f');values.frombytes(data)
                require(sys.byteorder=='little' and len(data)==count*4 and len(values)==count and all(math.isfinite(x)for x in values),'Nonfinite/malformed dump')
                row['arrays'][kind]=dict(path=str(path),sha256=sha(path),count=count,dtype='float32',byte_order='little')
                report['inputs'][str(path)]=sha(path)
            return row
        for label in ('A-prefill','B-steps'):
            guard(args.guard_state_dir)
            ack=client.command('rewind 3584',threading.Event());require(ack.get('ok')is True and ack.get('position')==3584,'Rewind failed')
            branch=dict(name=label,teacher_forced_tail_ids=list(ids[3584:]),rows=[])
            report['branches'].append(branch);save(output,report)
            if label=='A-prefill':
                response=client.command('prefill '+' '.join(map(str,ids[3584:])),threading.Event())
            else:
                for position,token in enumerate(ids[3584:],3585):
                    response=client.command(f'step {token}',threading.Event())
                    require(response.get('position')==position,'Teacher-forced step position mismatch')
            branch['rows'].append(capture(label,0,response))
            response=client.command(f'step {teacher[0]}',threading.Event())
            branch['rows'].append(capture(label,1,response))
            if label=='A-prefill':
                require(all(row['arrays']['logits']['sha256']==reference[row['step']]for row in branch['rows']),
                        'BaselineA differs from original baseline12 captures')
                report['baseline_a_original_capture_bit_identical']=True
            save(output,report)
        report['comparisons']=[]
        for step in (0,1):
            a,b=[branch['rows'][step]for branch in report['branches']]
            result=dict(context_tokens=3968+step,greedy_equal=a['predicted_id']==b['predicted_id'],arrays={})
            for kind in ('logits','hidden'):
                values=[]
                for row in (a,b):
                    arr=array.array('f');arr.frombytes(Path(row['arrays'][kind]['path']).read_bytes());values.append(arr)
                delta=[float(y)-float(x)for x,y in zip(*values)]
                worst=max(range(len(delta)),key=lambda i:abs(delta[i]))
                result['arrays'][kind]=dict(bit_identical=a['arrays'][kind]['sha256']==b['arrays'][kind]['sha256'],
                    max_absolute_difference=abs(delta[worst]),rms_difference=math.sqrt(math.fsum(d*d for d in delta)/len(delta)),
                    worst_index=worst,signed_B_minus_A=delta[worst])
            report['comparisons'].append(result)
        require(all(sha(Path(path))==digest for path,digest in report['inputs'].items())
                and all(sha(Path(path))==digest for path,digest in report['source_hashes'].items()),'Frozen input/source changed')
        report.update(completed=True,guard_after=guard(args.guard_state_dir))
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error));raise
    finally:
        if client is not None:
            try:client.close()
            except BaseException as error:report.update(completed=False,cleanup_error=str(error))
        os.environ.update(removed)
        save(output,report)
    require(report['completed'],'Cleanup failed')
    print(output)


if __name__=='__main__':
    main()
