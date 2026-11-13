#!/usr/bin/env python3
"""Guarded original CPU diagnostic captures; two exact repeats, no timing claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from benchmark_guard import BenchmarkGate
from native_cpu_chat import NativeCPUChat, EOS

VOCABULARY = 151936


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def guard(directory):
    status = BenchmarkGate(directory).status()
    metadata = json.loads((directory / 'run.json').read_text())
    if not status['running'] or metadata.get('pid') != os.getppid():
        raise RuntimeError('Run directly under serving-checkout benchmark_guard.py')
    return dict(directory=str(directory), metadata=metadata)


def read_ids(path):
    value = json.loads(path.read_text())
    if not isinstance(value, list) or any(type(x) is not int or not 0 <= x < VOCABULARY for x in value):
        raise ValueError('Token input must be a JSON list of vocabulary IDs')
    return value


def save(path, report):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=ROOT / 'build/native-cpu/weights.bin')
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--continuation', type=Path)
    parser.add_argument('--tokens', type=int, required=True)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--prefill-threads', type=int, default=6)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT / 'build/benchmark-status')
    args = parser.parse_args()
    if not 1 <= args.tokens <= 512 or not 1 <= args.threads <= 64 or not 1 <= args.prefill_threads <= 64:
        parser.error('tokens must be1..512 and thread counts1..64')
    guarded = guard(args.guard_state_dir.resolve())
    prompt = read_ids(args.prompt)
    continuation = read_ids(args.continuation) if args.continuation else None
    if not prompt or len(prompt) + args.tokens - 1 > 4096:
        parser.error('Nonempty prompt and requested positions must fit4096')
    if continuation is not None and len(continuation) != args.tokens - 1:
        parser.error('Teacher-forced continuation must contain tokens-1 IDs')
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report_path = directory / 'report.json'
    binary, weights = args.binary.resolve(), args.weights.resolve()
    sources = [Path(__file__).resolve(), ROOT / 'tools/capture_native_cpu_logits.py', ROOT / 'tools/native_cpu_chat.py', ROOT / 'engine/INTERFACE.md']
    inputs = [binary, weights, Path(str(weights)+'.index'), Path(str(weights)+'.manifest.json'), args.prompt.resolve()]
    if args.continuation: inputs.append(args.continuation.resolve())
    report = dict(completed=False, diagnostic_only=True, timing_claim=False,
        prompt_evaluation='sequential_step_diagnostic',
        prompt_evaluation_note='Each fixed prompt ID is consumed by step; intermediate predictions are ignored. This is not canonical batched prefill or a prefix-reuse conformance claim.',
        performance_claim_eligible=False, gpu=False, pytorch=False, guard_before=guarded,
        prompt_ids=prompt, continuation_ids=continuation or [], requested_steps=args.tokens,
        mode='teacher_forced' if continuation is not None else 'greedy', runs=[],
        configuration=dict(prompt_evaluation='sequential_step_diagnostic', architecture='qwen3', layers=28, hidden=1024, query_heads=16,
            kv_heads=8, vocabulary=VOCABULARY, weight_format='QWCPU01', weight_precision='f16',
            norm_precision='f32', activation_precision='f32', kv_type='f16', gpu=False,
            pytorch=False, decode_threads=args.threads, prefill_threads=args.prefill_threads),
        limits='Two identical repeats test this capture only; quality and NumPy tolerance gates remain separate.')
    engine = NativeCPUChat(None, binary=binary, weights=weights,
                           threads=args.threads, prefill_threads=args.prefill_threads)
    cancel = threading.Event()
    try:
        save(report_path, report)
        report['inputs'] = {str(path): sha(path) for path in inputs}
        report['source_hashes'] = {str(path): sha(path) for path in sources}
        manifest = json.loads(Path(str(weights)+'.manifest.json').read_text())
        if manifest.get('format') != 'QWCPU01' or manifest.get('completed') is not True:
            raise RuntimeError('Incomplete or unsupported pack manifest')
        if (manifest['binary']['sha256'] != report['inputs'][str(weights)]
                or manifest['index']['sha256'] != report['inputs'][str(weights)+'.index']):
            raise RuntimeError('Weight pack differs from immutable manifest')
        report['weight_pack_manifest'] = manifest
        engine.start(cancel)
        report['startup'] = engine.ready
        report['pid'] = engine.process.pid
        for repeat in range(2):
            row = dict(index=repeat, warmup=False, fixed_work_valid=False, own_repeat_ids_equal=False,
                       predicted_token_ids=[], consumed_continuation_ids=[], eog_prediction_steps=[], logit_dumps=[])
            report['runs'].append(row)
            reset = engine.command('reset', cancel)
            if reset != {'ok': True, 'position': 0}:
                raise RuntimeError('Unexpected reset acknowledgement')
            if engine.command('mode greedy', cancel).get('ok') is not True:
                raise RuntimeError('Greedy mode acknowledgement missing')
            for position, prompt_token in enumerate(prompt, 1):
                response = engine.command('step ' + str(prompt_token), cancel)
                if response.get('position') != position:
                    raise RuntimeError('Sequential prompt step position mismatch')
            for step in range(args.tokens):
                token = response.get('token')
                if type(token) is not int or not 0 <= token < VOCABULARY or response.get('position') != len(prompt)+step:
                    raise RuntimeError('Invalid prediction or context position')
                row['predicted_token_ids'].append(token)
                if token in EOS:
                    row['eog_prediction_steps'].append(step)
                    if continuation is None:
                        raise RuntimeError('Greedy capture reached EOS; use an explicit teacher-forced trajectory')
                dump = directory / f'run{repeat}-step{step}.f32'
                ack = engine.command('dump ' + str(dump), cancel)
                if ack.get('ok') is not True or ack.get('count') != VOCABULARY or ack.get('position') != len(prompt)+step:
                    raise RuntimeError('Invalid dump acknowledgement')
                if dump.stat().st_size != VOCABULARY*4:
                    raise RuntimeError('Invalid full-vocabulary dump size')
                row['logit_dumps'].append(dict(path=str(dump), step=step, dtype='float32',
                    byte_order='little', shape=[VOCABULARY], bytes=VOCABULARY*4,
                    context_tokens=len(prompt)+step, sha256=sha(dump)))
                save(report_path, report)
                if step+1 < args.tokens:
                    consumed = continuation[step] if continuation is not None else token
                    row['consumed_continuation_ids'].append(consumed)
                    response = engine.command(f'step {consumed}', cancel)
            row['fixed_work_valid'] = True
        first, second = report['runs']
        if (first['predicted_token_ids'] != second['predicted_token_ids']
                or first['consumed_continuation_ids'] != second['consumed_continuation_ids']
                or [d['sha256'] for d in first['logit_dumps']] != [d['sha256'] for d in second['logit_dumps']]):
            raise RuntimeError('Repeated native trajectory or full-logit bytes changed')
        for row in report['runs']:
            row['own_repeat_ids_equal'] = True
            row['own_repeat_logits_bit_identical'] = True
        if report['inputs'] != {str(path):sha(path) for path in inputs} or report['source_hashes'] != {str(path):sha(path) for path in sources}:
            raise RuntimeError('Capture inputs/source changed during execution')
        report['guard_after'] = guard(args.guard_state_dir.resolve())
        report['completed'] = True
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        try:
            engine.close()
        except BaseException as error:
            report['completed'] = False
            report['cleanup_error'] = str(error)
        save(report_path, report)
    if not report['completed']:
        raise RuntimeError('Capture cleanup failed')
    print(report_path)


if __name__ == '__main__':
    main()
