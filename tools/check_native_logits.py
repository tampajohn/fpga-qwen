"""Independent NumPy full-logit calibration or an explicitly declared gate.

Accept profile_native_cpu.py report.json wrappers or raw C++ client.json files.
A single selected run from each report is replayed with its actual consumed
continuation IDs; all reports must share that trajectory. No autoregressive
oracle drift, GPU, PyTorch, or simulator execution. Run under benchmark_guard.
Without --atol this is report-only calibration: passed is null, never inferred
from the observed errors. With --atol, greedy identity and all declared numerical
limits are required. New tolerances require a separately justified validation
run; this tool never derives or widens them from calibration results.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_guard import BenchmarkGate
from tools.experiment import locate_model
from tests.test_full_model import NumpyQwen
import numpy as np

VOCABULARY = 151936
FILE_TYPES = {'f32': 0, 'f16': 1, 'bf16': 32}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def guard(directory):
    status = BenchmarkGate(directory).status()
    metadata = json.loads((directory / 'run.json').read_text())
    if not status['running'] or metadata.get('pid') != os.getppid():
        raise RuntimeError('Run directly under serving-checkout benchmark_guard.py')
    return dict(directory=str(directory), metadata=metadata)


def ids(value, label):
    if not isinstance(value, list) or any(type(x) is not int or not 0 <= x < VOCABULARY for x in value):
        raise ValueError(f'Invalid token-ID list: {label}')
    return value


def load_capture(path, run_index, inputs):
    path = path.resolve()
    supplied = json.loads(path.read_text())
    inputs[str(path)] = sha(path)
    wrapped = 'client_report' in supplied
    client = supplied['client_report'] if wrapped else supplied
    if supplied.get('completed') is not True or client.get('completed') is not True:
        raise RuntimeError('Numerical capture must be complete')
    if client.get('gpu') is not False or client.get('pytorch') is not False:
        raise RuntimeError('Expected CPU-only diagnostic capture')
    config = client['configuration']
    expected = dict(architecture='qwen3', layers=28, hidden=1024, query_heads=16,
                    kv_heads=8, vocabulary=VOCABULARY)
    if config.get('weight_format') == 'QWCPU01':
        expected.update(weight_precision='f16', norm_precision='f32', activation_precision='f32',
                        kv_type='f16', gpu=False, pytorch=False)
    else:
        expected.update(n_gpu_layers=0, offload_kqv=False, op_offload=False)
    if any(type(config.get(k)) is not type(v) or config[k] != v for k, v in expected.items()):
        raise RuntimeError('Unexpected native model or offload configuration')
    prompt = ids(client['prompt_ids'], 'prompt')
    if not prompt:
        raise ValueError('Empty diagnostic prompt')
    runs = client['runs']
    matches = [row for row in runs if row['index'] == run_index] if run_index is not None else [row for row in runs if row['warmup'] is False][:1]
    if len(matches) != 1:
        raise ValueError('Select exactly one existing diagnostic run (default: first non-warmup run)')
    row = matches[0]
    predictions = ids(row['predicted_token_ids'], 'predictions')
    consumed = ids(row['consumed_continuation_ids'], 'consumed continuation')
    if (row.get('fixed_work_valid') is not True or row.get('own_repeat_ids_equal') is not True
            or len(predictions) != client['requested_steps'] or len(consumed) != len(predictions) - 1):
        raise RuntimeError('Incomplete or inconsistent fixed numerical trajectory')
    if client['mode'] == 'teacher_forced':
        if consumed != ids(client['continuation_ids'], 'teacher-forced continuation'):
            raise RuntimeError('Consumed continuation differs from the prescribed teacher forcing')
    elif client['mode'] == 'greedy':
        if consumed != predictions[:-1] or row.get('eog_prediction_steps'):
            raise RuntimeError('Greedy capture has mismatched consumption or early EOS')
    else:
        raise RuntimeError('Unknown native capture mode')
    dumps = row['logit_dumps']
    if len(dumps) != len(predictions):
        raise RuntimeError('A full logit dump is required for every native prediction')
    for step, item in enumerate(dumps):
        dump = Path(item['path'])
        if (item['step'] != step or item['dtype'] != 'float32' or item['byte_order'] != 'little'
                or item['shape'] != [VOCABULARY] or item['bytes'] != VOCABULARY * 4
                or item['context_tokens'] != len(prompt) + step or dump.stat().st_size != VOCABULARY * 4):
            raise RuntimeError('Invalid dump position, shape or type')
        digest = sha(dump)
        if item.get('sha256') is not None and item['sha256'] != digest:
            raise RuntimeError('Logit dump changed since capture')
        if wrapped and supplied['logit_dumps'].get(str(dump)) != digest:
            raise RuntimeError('Native dump changed since profiling wrapper verified it')
        inputs[str(dump)] = digest
    if wrapped or config.get('weight_format') == 'QWCPU01':
        for source, digest in supplied['inputs'].items():
            if source not in inputs:
                inputs[source] = sha(source)
            if inputs[source] != digest:
                raise RuntimeError('Native capture model/prompt input changed since recording')
    return dict(path=str(path), wrapped_provenance=wrapped, configuration=config,
                run_index=row['index'], warmup=row['warmup'], native_mode=client['mode'],
                prompt=prompt, consumed=consumed, predictions=predictions, dumps=dumps,
                eog_prediction_steps=row['eog_prediction_steps'],
                client_manifest=supplied.get('client_manifest'))


def top_values(values, count):
    order = np.argsort(-values, kind='stable')[:max(2, count)]
    return order, [dict(id=int(index), logit=float(values[index])) for index in order[:count]]


def compare(actual, expected, native_prediction, top_k, atol, rtol, rms_atol):
    if actual.shape != (VOCABULARY,) or expected.shape != (VOCABULARY,):
        raise AssertionError('Wrong full-vocabulary output shape')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError('Nonfinite native/oracle logit')
    native_order, native_top = top_values(actual, top_k)
    oracle_order, oracle_top = top_values(expected, top_k)
    if actual[native_prediction] != actual[native_order[0]]:
        raise AssertionError('Dump argmax is inconsistent with the native greedy prediction')
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    absolute = np.abs(delta)
    maximum_index = int(np.argmax(absolute))
    rms = float(np.sqrt(np.mean(delta * delta)))
    greedy_equal = native_prediction == int(oracle_order[0])
    numerical = None if atol is None else bool(np.all(absolute <= atol + rtol * np.abs(expected.astype(np.float64))))
    rms_passed = None if rms_atol is None else rms <= rms_atol
    return dict(finite=True, vocabulary=VOCABULARY,
                max_absolute_error=float(absolute[maximum_index]), rms_error=rms,
                max_error_token_id=maximum_index,
                native_at_max_error=float(actual[maximum_index]), oracle_at_max_error=float(expected[maximum_index]),
                native_predicted_id=native_prediction, native_argmax_id=int(native_order[0]),
                oracle_argmax_id=int(oracle_order[0]), greedy_equal=greedy_equal,
                native_top=native_top, oracle_top=oracle_top,
                native_top1_margin=float(actual[native_order[0]]) - float(actual[native_order[1]]),
                oracle_top1_margin=float(expected[oracle_order[0]]) - float(expected[oracle_order[1]]),
                oracle_preference_over_native_id=float(expected[oracle_order[0]]) - float(expected[native_prediction]),
                native_preference_over_oracle_id=float(actual[native_order[0]]) - float(actual[oracle_order[0]]),
                allclose_passed=numerical, rms_limit_passed=rms_passed,
                passed=None if atol is None else greedy_equal and numerical and rms_passed is not False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-report', type=Path, nargs='+', required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--weight-precision', choices=('bf16', 'f16', 'f32'), default='f16')
    parser.add_argument('--kv-precision', choices=('f16', 'f32'), default='f16')
    parser.add_argument('--run-index', type=int)
    parser.add_argument('--steps', default='all', help='all or an explicit comma-separated list of prediction steps')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--atol', type=float)
    parser.add_argument('--rtol', type=float)
    parser.add_argument('--rms-atol', type=float)
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT / 'build/benchmark-status')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.atol is None and (args.rtol is not None or args.rms_atol is not None):
        parser.error('--rtol/--rms-atol require an explicit --atol')
    if any(value is not None and (not math.isfinite(value) or value < 0) for value in (args.atol, args.rtol, args.rms_atol)):
        parser.error('Tolerances must be finite and nonnegative')
    if not 2 <= args.top_k <= 20 or args.run_index is not None and args.run_index < 0:
        parser.error('Use top-k2..20 and a nonnegative run index')
    rtol = 0.0 if args.rtol is None else args.rtol
    guarded = guard(args.guard_state_dir.resolve())
    if args.output.exists():
        parser.error('Preserve existing reports; choose a fresh output path')
    inputs = {}
    captures = [load_capture(path, args.run_index, inputs) for path in args.profile_report]
    first = captures[0]
    if any(capture['prompt'] != first['prompt'] or capture['consumed'] != first['consumed'] for capture in captures):
        raise RuntimeError('Native reports must share identical prompt and consumed continuation IDs')
    if args.steps == 'all':
        steps = list(range(len(first['predictions'])))
    else:
        try:
            steps = sorted(set(int(value) for value in args.steps.split(',')))
        except ValueError:
            parser.error('Steps must be all or comma-separated integers')
        if not steps or steps[0] < 0 or steps[-1] >= len(first['predictions']):
            parser.error('Selected steps are outside the captured trajectory')
    for capture in captures:
        configuration = capture['configuration']
        matched_weight = (configuration['weight_precision'] == args.weight_precision
                          if configuration.get('weight_format') == 'QWCPU01' else
                          int(configuration['model_file_type']) == FILE_TYPES[args.weight_precision])
        capture['precision_matched'] = (matched_weight and configuration['kv_type'] == args.kv_precision)
        if args.atol is not None and not capture['precision_matched']:
            raise RuntimeError('A validation gate requires matched weight and KV precision classes')
    checkpoint = locate_model(args.checkpoint)
    inputs[str(checkpoint / 'model.safetensors')] = sha(checkpoint / 'model.safetensors')
    source_paths = [Path(__file__), ROOT / 'tests/test_full_model.py', ROOT / 'tools/experiment.py']
    report = dict(completed=False, calibration_only=args.atol is None, passed=None,
                  timing_claim=False, gpu=False, pytorch=False, guard_before=guarded,
                  declared_tolerances=None if args.atol is None else dict(atol=args.atol, rtol=rtol, rms_atol=args.rms_atol,
                                                                       greedy_identity_required=True),
                  inputs=inputs, source_hashes={str(path): sha(path) for path in source_paths},
                  oracle=dict(checkpoint=str(checkpoint), weight_precision=args.weight_precision,
                              kv_precision=args.kv_precision, arithmetic='NumPy FP32 network math; existing FP64 RoPE angle generation then FP32 sine/cosine.',
                              kv_representation='FP16-rounded values widened to FP32 before attention and cached as FP32' if args.kv_precision == 'f16' else 'FP32',
                              matrix_weight_rounding_only=args.weight_precision == 'f16',
                              norm_weights='Original BF16 checkpoint values widened to FP32 without FP16 rounding',
                              numpy_version=np.__version__, library_threads={key: os.environ.get(key) for key in
                                  ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS')}),
                  captures=[{key: value for key, value in capture.items() if key != 'dumps'} for capture in captures],
                  selected_steps=steps, comparisons=[],
                  limitations=['Oracle replays native consumed IDs, including supplied teacher forcing after an EOS prediction; this is not an autonomous conversation-quality test.',
                               'Native mixed-precision kernels and reduction orders differ from independent NumPy FP32 math. Calibration reports errors and never declares a tolerance pass.',
                               'A gate applies only supplied limits at selected positions; tolerances must be justified before a separate validation run, not fitted retrospectively to these measurements.',
                               'One-dimensional norm tensors stay FP32; f16 rounds only two-dimensional checkpoint matrices. No quantized-weight oracle is provided.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    try:
        save()  # Preserve declared thresholds before constructing/running the oracle.
        oracle = NumpyQwen(checkpoint, weight_precision=args.weight_precision, kv_precision=args.kv_precision)
        prompt = first['prompt']
        for offset, token in enumerate(prompt[:-1]):
            oracle.token(token, logits=False)
            if (offset + 1) % 64 == 0:
                print(f'Oracle prefix {offset + 1}/{len(prompt) - 1}', flush=True)
        trajectory = [prompt[-1], *first['consumed']]
        for step, token in enumerate(trajectory[:steps[-1] + 1]):
            expected = oracle.token(token, logits=step in steps)
            if step not in steps:
                continue
            for index, capture in enumerate(captures):
                dump = Path(capture['dumps'][step]['path'])
                actual = np.fromfile(dump, dtype='<f4')
                comparison = compare(actual, expected, capture['predictions'][step], args.top_k,
                                     args.atol, rtol, args.rms_atol)
                comparison.update(capture=index, step=step, context_tokens=len(prompt) + step,
                                  oracle_logits_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                                  native_logits_sha256=inputs[str(dump)])
                report['comparisons'].append(comparison)
                print(json.dumps({key: comparison[key] for key in ('capture', 'step', 'context_tokens',
                                  'max_absolute_error', 'rms_error', 'greedy_equal', 'passed')}), flush=True)
            save()
        report['summary'] = dict(compared_positions=len(report['comparisons']),
            max_absolute_error=max(row['max_absolute_error'] for row in report['comparisons']),
            max_position_rms_error=max(row['rms_error'] for row in report['comparisons']),
            greedy_mismatches=sum(not row['greedy_equal'] for row in report['comparisons']))
        if any(sha(path) != digest for path, digest in inputs.items()):
            raise RuntimeError('Captured report, logits or model input changed during numerical replay')
        if any(sha(path) != digest for path, digest in report['source_hashes'].items()):
            raise RuntimeError('Independent oracle/helper source changed during replay')
        report['guard_after'] = guard(args.guard_state_dir.resolve())
        report['passed'] = None if args.atol is None else all(row['passed'] for row in report['comparisons'])
        report['completed'] = True
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        save()
    if report['passed'] is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
