#!/usr/bin/env python3
"""Capture coupled normalized hidden states and full logits, twice; no timing claims."""
import argparse
import hashlib
import json
import math
import struct
import os
from pathlib import Path
import threading

from benchmark_guard import BenchmarkGate
from native_cpu_chat import NativeCPUChat, EOS

ROOT = Path(__file__).resolve().parents[1]
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
    sources = [Path(__file__).resolve(), ROOT / 'tools/native_cpu_chat.py', ROOT / 'tools/benchmark_guard.py']
    build_manifest_path = binary.parent / 'manifest.json'
    inputs = [build_manifest_path, binary, weights, Path(str(weights)+'.index'), Path(str(weights)+'.manifest.json'), args.prompt.resolve()]
    if args.continuation: inputs.append(args.continuation.resolve())
    report = dict(completed=False, diagnostic_only=True, timing_claim=False, coupled_dumps=True,
        performance_claim_eligible=False, gpu=False, pytorch=False, guard_before=guarded,
        prompt_ids=prompt, continuation_ids=continuation or [], requested_steps=args.tokens,
        mode='teacher_forced' if continuation is not None else 'greedy', runs=[],
        configuration=dict(architecture='qwen3', layers=28, hidden=1024, query_heads=16,
            kv_heads=8, vocabulary=VOCABULARY, weight_format='QWCPU01', weight_precision='f16',
            norm_precision='f32', activation_precision='f32', kv_type='f16', gpu=False,
            pytorch=False, decode_threads=args.threads, prefill_threads=args.prefill_threads, reuse_prefix=False),
        limits='Two identical repeats test this capture only; quality and NumPy tolerance gates remain separate.')
    engine = NativeCPUChat(None, binary=binary, weights=weights,
                           threads=args.threads, prefill_threads=args.prefill_threads, reuse_prefix=False)
    cancel = threading.Event()
    original_environment = dict(os.environ)
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
        build_manifest = json.loads(build_manifest_path.read_text())
        if (build_manifest.get('completed') is not True
                or build_manifest.get('gpu') is not False
                or build_manifest.get('pytorch') is not False
                or build_manifest.get('binary_sha256') != report['inputs'][str(binary)]):
            raise RuntimeError('CPU binary differs from immutable build manifest')
        report['build_manifest'] = build_manifest
        removed = []
        for key in list(os.environ):
            if key.startswith(('DYLD_', 'GGML_KLEIDIAI_', 'LLAMA_ARG_')) or key in ('GGML_BACKEND_PATH', 'GGML_TOTAL_THREADS', 'VECLIB_MAXIMUM_THREADS'):
                removed.append(key); del os.environ[key]
        report['removed_environment_names'] = sorted(removed)
        engine.start(cancel)
        report['startup'] = engine.ready
        report['pid'] = engine.process.pid
        for repeat in range(2):
            row = dict(index=repeat, warmup=False, fixed_work_valid=False, own_repeat_ids_equal=False,
                       predicted_token_ids=[], consumed_continuation_ids=[], eog_prediction_steps=[], logit_dumps=[], hidden_dumps=[])
            report['runs'].append(row)
            reset = engine.command('reset', cancel)
            if reset != {'ok': True, 'position': 0}:
                raise RuntimeError('Unexpected reset acknowledgement')
            if engine.command('mode greedy', cancel).get('ok') is not True:
                raise RuntimeError('Greedy mode acknowledgement missing')
            response = engine.command('prefill ' + ' '.join(map(str, prompt)), cancel)
            for step in range(args.tokens):
                token = response.get('token')
                if type(token) is not int or not 0 <= token < VOCABULARY or response.get('position') != len(prompt)+step:
                    raise RuntimeError('Invalid prediction or context position')
                row['predicted_token_ids'].append(token)
                if token in EOS:
                    row['eog_prediction_steps'].append(step)
                    if continuation is None:
                        raise RuntimeError('Greedy capture reached EOS; use an explicit teacher-forced trajectory')
                hidden = directory / f'run{repeat}-step{step}.hidden.f32'
                ack = engine.command('dump-hidden ' + str(hidden), cancel)
                if (ack.get('ok') is not True or ack.get('count') != 1024
                        or ack.get('position') != len(prompt)+step or ack.get('format') != 'f32le'):
                    raise RuntimeError('Invalid normalized-hidden dump acknowledgement')
                payload = hidden.read_bytes()
                if len(payload) != 1024*4 or not all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', payload)):
                    raise RuntimeError('Invalid/nonfinite normalized hidden state')
                row['hidden_dumps'].append(dict(path=str(hidden), step=step, dtype='float32',
                    byte_order='little', shape=[1024], bytes=1024*4,
                    context_tokens=len(prompt)+step, sha256=sha(hidden),
                    semantics='Final RMS-normalized hidden vector immediately used by tied vocabulary projection'))
                dump = directory / f'run{repeat}-step{step}.logits.f32'
                ack = engine.command('dump ' + str(dump), cancel)
                if ack.get('ok') is not True or ack.get('count') != VOCABULARY or ack.get('position') != len(prompt)+step:
                    raise RuntimeError('Invalid dump acknowledgement')
                payload = dump.read_bytes()
                if len(payload) != VOCABULARY*4 or not all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', payload)):
                    raise RuntimeError('Invalid/nonfinite full-vocabulary logits')
                if engine.command('stats', cancel).get('position') != len(prompt)+step:
                    raise RuntimeError('Position changed between coupled diagnostic dumps')
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
                or [d['sha256'] for d in first['logit_dumps']] != [d['sha256'] for d in second['logit_dumps']]
                or [d['sha256'] for d in first['hidden_dumps']] != [d['sha256'] for d in second['hidden_dumps']]):
            raise RuntimeError('Repeated trajectory, hidden-state or full-logit bytes changed')
        for kind in ('hidden_dumps', 'logit_dumps'):
            for left, right in zip(first[kind], second[kind]):
                if Path(left['path']).read_bytes() != Path(right['path']).read_bytes():
                    raise RuntimeError('Repeated diagnostic blobs are not bit identical')
        for row in report['runs']:
            row['own_repeat_ids_equal'] = True
            row['own_repeat_logits_bit_identical'] = True
            row['own_repeat_hidden_bit_identical'] = True
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
        os.environ.clear(); os.environ.update(original_environment)
        save(report_path, report)
    if not report['completed']:
        raise RuntimeError('Capture cleanup failed')
    print(report_path)


if __name__ == '__main__':
    main()
