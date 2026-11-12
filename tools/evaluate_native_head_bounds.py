#!/usr/bin/env python3
"""Offline int8 routing-bound feasibility using saved hidden state and dense logits.

No inference or production pruning runs here. Dense logits supply the exact
threshold only to estimate possible candidate counts. Observed interval coverage
is a test of this implementation, not a formal proof of a future runtime kernel.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from benchmark_guard import BenchmarkGate

ROOT = Path(__file__).resolve().parents[1]
ROWS, COLS, BLOCK = 151936, 1024, 64


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_capture(path, hidden_path, logits_path, weights, frozen):
    capture = json.loads(path.read_text())
    require(capture.get('completed') is True and capture.get('passed', True) is True
            and capture.get('coupled_dumps') is True and 'error' not in capture
            and 'cleanup_error' not in capture and capture.get('gpu') is False
            and capture.get('pytorch') is False, 'Capture did not complete/pass coupled CPU dumps')
    configuration = capture.get('configuration', {})
    require(all(configuration.get(key) == value for key, value in dict(
        hidden=COLS, vocabulary=ROWS, weight_format='QWCPU01', weight_precision='f16',
        activation_precision='f32', norm_precision='f32').items()), 'Capture precision/shape mismatch')
    for artifact in (weights, Path(str(weights)+'.index'), Path(str(weights)+'.manifest.json')):
        require(capture.get('inputs', {}).get(str(artifact)) == frozen[str(artifact)],
                'Capture weight-pack mismatch: '+str(artifact))
    runs = capture.get('runs', [])
    require(len(runs) >= 2 and len({run.get('index') for run in runs}) == len(runs),
            'Capture requires distinct repeated runs')
    matches = []
    for run in runs:
        require(all(run.get(key) is True for key in ('fixed_work_valid', 'own_repeat_ids_equal',
            'own_repeat_logits_bit_identical', 'own_repeat_hidden_bit_identical')),
            'Capture repeat checks did not pass')
        require(len(run.get('hidden_dumps', [])) == capture.get('requested_steps')
                and len(run.get('logit_dumps', [])) == capture.get('requested_steps'), 'Incomplete dump trajectory')
        hidden = [row for row in run['hidden_dumps'] if Path(row['path']).resolve() == hidden_path]
        logits = [row for row in run['logit_dumps'] if Path(row['path']).resolve() == logits_path]
        if hidden or logits:
            require(len(hidden) == len(logits) == 1, 'Selected arrays come from different capture runs')
            matches.append((run, hidden[0], logits[0]))
    require(len(matches) == 1, 'Selected arrays must identify one coupled capture run')
    selected_run, selected_hidden, selected_logits = matches[0]
    step = selected_hidden.get('step')
    require(type(step) is int and 0 <= step < capture['requested_steps']
            and selected_logits.get('step') == step, 'Selected arrays are from different steps')
    position = len(capture.get('prompt_ids', [])) + step
    added = []
    for run in runs:
        for kind, count, selected in (('hidden_dumps', COLS, selected_hidden),
                                      ('logit_dumps', ROWS, selected_logits)):
            rows = [row for row in run[kind] if row.get('step') == step]
            require(len(rows) == 1, 'Repeated capture position is missing or duplicated')
            row = rows[0]
            artifact = Path(row['path']).resolve()
            require(row.get('context_tokens') == position and row.get('dtype') == 'float32'
                    and row.get('byte_order') == 'little' and row.get('shape') == [count]
                    and row.get('bytes') == count*4 and artifact.stat().st_size == count*4,
                    'Invalid coupled dump position, dtype, shape or size')
            digest = sha(artifact)
            selected_path = hidden_path if kind == 'hidden_dumps' else logits_path
            require(digest == row.get('sha256') == selected.get('sha256') == frozen[str(selected_path)],
                    'Selected or repeated dump differs from capture hash')
            added.append(artifact)
    linked_build = None
    if 'build_manifest' in capture:
        build = capture['build_manifest']
        require(build.get('completed') is True and build.get('gpu') is False
                and build.get('pytorch') is False, 'Capture build manifest is incomplete')
        binary = Path(build['binary']).resolve()
        build_path = binary.parent/'manifest.json'
        require(sha(binary) == build.get('binary_sha256') == capture['inputs'].get(str(binary)),
                'Capture binary differs from its build manifest')
        require(sha(build_path) == capture['inputs'].get(str(build_path))
                and json.loads(build_path.read_text()) == build, 'Embedded build manifest differs from frozen artifact')
        added.extend((binary, build_path))
        linked_build = dict(binary=str(binary), binary_sha256=build['binary_sha256'],
                            manifest=str(build_path), manifest_sha256=sha(build_path))
    return dict(verified=True, path=str(path), sha256=sha(path), run=selected_run['index'],
                step=step, context_tokens=position, verified_repeats=len(runs),
                build=linked_build,
                scope='Coupled capture metadata, repeated array bytes, pack hashes and listed binary/build linkage checked. '
                      'Floating-point control-register assumptions are not measured by this capture.'), added


def up(value):
    return np.nextafter(value, np.float64(np.inf))


def down(value):
    return np.nextafter(value, np.float64(-np.inf))


def positive_sum_up(values):
    """Outward round every addition; no reliance on NumPy's reduction order."""
    total = np.zeros(values.shape[:-1], dtype=np.float64)
    for index in range(values.shape[-1]):
        total = up(total + values[..., index])
    return total


def float32_up(value):
    """Smallest computed FP32 enclosure at or above a finite FP64 value."""
    with np.errstate(over='ignore', invalid='ignore'):
        rounded = np.asarray(value, dtype=np.float32)
    return np.where(rounded.astype(np.float64) < value,
                    np.nextafter(rounded, np.float32(np.inf)), rounded)


def quantize(values):
    """Power-of-two reconstruction, with actual residuals bounded afterward."""
    maximum = np.max(np.abs(values), axis=-1)
    scaled = np.where(maximum > 0, maximum / 127.0, 1.0)
    fraction, exponent = np.frexp(scaled)
    scale = np.ldexp(np.ones_like(scaled), exponent - (fraction == .5))
    quantized = np.clip(np.rint(values / scale[..., None]), -127, 127).astype(np.int8)
    reconstructed = quantized.astype(np.float64) * scale[..., None]
    # A single nextafter encloses each correctly rounded FP64 subtraction.
    residual = np.max(up(np.abs(values - reconstructed)), axis=-1)
    return quantized, scale, reconstructed, residual


def matrix_view(weights):
    require(weights.stat().st_size >= 64, 'Truncated pack')
    with weights.open('rb') as stream:
        require(stream.read(64) == b'QWCPU01' + bytes(57), 'Invalid pack header')
    lines = Path(str(weights) + '.index').read_text().splitlines()
    require(lines and lines[0].split() == ['QWCPU01', str(len(lines)-1)], 'Invalid index header/count')
    matches = [line.split() for line in lines[1:] if line.split()[0] == 'model.embed_tokens.weight']
    require(len(matches) == 1 and len(matches[0]) == 6, 'Expected one tied embedding/output matrix')
    _, kind, rows, cols, offset, length = matches[0]
    rows, cols, offset, length = map(int, (rows, cols, offset, length))
    require(kind == 'F16' and (rows, cols) == (ROWS, COLS) and length == ROWS*COLS*2
            and offset >= 64 and offset % 64 == 0 and offset+length <= weights.stat().st_size,
            'Invalid output matrix metadata')
    return np.memmap(weights, mode='r', dtype='<f2', offset=offset, shape=(ROWS, COLS)), offset


def statistics(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return dict(finite_count=0)
    return dict(finite_count=int(finite.size), minimum=float(np.min(finite)),
                median=float(np.median(finite)), p95=float(np.percentile(finite, 95)),
                maximum=float(np.max(finite)))


def ordered_top(scores, count):
    # Current sampler ties select smaller token IDs.
    return np.lexsort((np.arange(ROWS), -scores))[:count]


def candidate_report(dense, upper, count):
    selected = ordered_top(dense, count)
    threshold = dense[selected[-1]]
    candidates = upper >= threshold  # Equality remains eligible, including ties.
    retained = bool(np.all(candidates[selected]))
    return dict(k=count, exact_dense_threshold=float(threshold), true_top_ids=selected.tolist(),
                candidate_rows=int(np.count_nonzero(candidates)),
                candidate_fraction=float(np.mean(candidates)),
                true_top_retained=retained,
                note='Uses an unavailable production oracle threshold; candidate count is an optimistic feasibility diagnostic.')


def presence_report(dense, upper):
    temperature, penalty = np.float32(.7), np.float32(1.5)
    # Every possible history score lies between the all-seen and none-seen
    # scores. Monotonic FP32 subtraction/division lets the kth all-seen score
    # serve as a common lower threshold for all presence masks.
    with np.errstate(over='ignore', invalid='ignore'):
        lower_scores = (dense - penalty) / temperature
        none_scores = dense / temperature
        upper_scores = float32_up(upper) / temperature
    if not np.all(np.isfinite(lower_scores)) or not np.all(np.isfinite(none_scores)):
        return dict(available=False, candidate_rows=ROWS, candidate_fraction=1.0,
                    reason='FP32 sampler-score transformation may overflow; retain every row',
                    observed_mask_checks=[])
    threshold = lower_scores[ordered_top(lower_scores, 20)[-1]]
    candidates = upper_scores >= threshold
    checks = []
    for name, seen in (('none', np.zeros(ROWS, dtype=bool)),
                       ('all', np.ones(ROWS, dtype=bool)),
                       ('alternating', np.arange(ROWS) % 2 == 0)):
        scores = (dense - np.where(seen, penalty, np.float32(0))) / temperature
        selected = ordered_top(scores, 20)
        checks.append(dict(mask=name, true_top_ids=selected.tolist(),
                           true_top_retained=bool(np.all(candidates[selected]))))
    return dict(available=True, presence_penalty=float(penalty), temperature=float(temperature),
                candidate_rows=int(np.count_nonzero(candidates)), candidate_fraction=float(np.mean(candidates)),
                all_seen_kth_score_lower_threshold=float(threshold), observed_mask_checks=checks,
                scope='Conservative none-seen upper scores versus all-seen kth threshold. '
                      'Three masks checked empirically; not exhaustive enumeration or formal verification.')


def evaluate(matrix, hidden, dense, rows_per_chunk):
    x = hidden.astype(np.float64).reshape(COLS//BLOCK, BLOCK)
    qx, sx, ax, ex = quantize(x)
    sum_x = positive_sum_up(np.abs(x))
    max_x = np.max(np.abs(x), axis=-1)
    unit_roundoff = np.float64(2**-24)
    gamma = up((131*unit_roundoff)/(1-131*unit_roundoff))
    amplification = up(1+gamma)
    tiny = np.float64(np.finfo(np.float32).tiny)
    maximum = np.float64(np.finfo(np.float32).max)
    lower, upper, center, quant_error, reference_error = [np.empty(ROWS, dtype=np.float64) for _ in range(5)]
    fallback = np.zeros(ROWS, dtype=bool)
    invalid_weight_rows = np.zeros(ROWS, dtype=bool)
    for begin in range(0, ROWS, rows_per_chunk):
        end = min(begin+rows_per_chunk, ROWS)
        w = np.asarray(matrix[begin:end], dtype=np.float64).reshape(-1, COLS//BLOCK, BLOCK)
        valid = np.all(np.isfinite(w), axis=(1, 2))
        invalid_weight_rows[begin:end] = ~valid
        w = np.where(np.isfinite(w), w, 0)
        qw, sw, aw, ew = quantize(w)
        # Model the planned three-FP32 metadata fields: power-of-two scale,
        # upward max residual, and exact reconstructed block L1 norm.
        ew = float32_up(ew).astype(np.float64)
        sum_a = np.sum(np.abs(qw.astype(np.int32)), axis=-1, dtype=np.int64).astype(np.float64)*sw
        require(np.all(sw.astype(np.float32).astype(np.float64) == sw)
                and np.all(sum_a.astype(np.float32).astype(np.float64) == sum_a),
                'Weight metadata is not exactly representable as planned FP32 fields')
        integer_dot = np.sum(qw.astype(np.int32)*qx.astype(np.int32), axis=-1, dtype=np.int64)
        require(np.all(np.abs(integer_dot) <= BLOCK*127*127), 'Integer accumulation exceeded exact int32 range')
        # Each term is an <=21-bit integer times two power-of-two scales. For
        # F16 weights/F32 activations it is exactly representable in FP64.
        terms = integer_dot.astype(np.float64)*(sw*sx)
        approximate_lo = np.zeros(end-begin, dtype=np.float64)
        approximate_hi = approximate_lo.copy()
        for block in range(COLS//BLOCK):
            approximate_lo = down(approximate_lo + terms[:, block])
            approximate_hi = up(approximate_hi + terms[:, block])
        # wx-ab = (w-a)x + a(x-b). Every positive operation rounds outward.
        quant = positive_sum_up(up(up(ew*sum_x) + up(ex*sum_a)))
        sum_w = positive_sum_up(up(sum_a + up(BLOCK*ew)))
        absolute_products = positive_sum_up(up(up(sum_a + up(BLOCK*ew))*max_x))
        # 1024 FMAs plus seven reduction additions. Three tiny allowances per
        # operation conservatively cover subnormal intermediate/result flushing;
        # hidden-input flushing additionally contributes tiny*sum|w|. Inflate
        # every additive allowance through the longest accumulation path.
        underflow = up(up(up(tiny*sum_w) + up(np.float64(3*1031)*tiny))*amplification)
        reference = up(up(gamma*absolute_products) + underflow)
        unsafe = ~valid | (up(absolute_products+reference) > maximum)
        total = up(quant+reference)
        lower[begin:end] = np.where(unsafe, -np.inf, down(approximate_lo-total))
        upper[begin:end] = np.where(unsafe, np.inf, up(approximate_hi+total))
        center[begin:end] = np.sum(terms, axis=-1)
        quant_error[begin:end], reference_error[begin:end] = quant, reference
        fallback[begin:end] = unsafe
    actual = dense.astype(np.float64)
    covered = (actual >= lower) & (actual <= upper)
    failures = np.flatnonzero(~covered)
    top = [candidate_report(actual, upper, count) for count in (1, 20)]
    presence = presence_report(dense, upper)
    return dict(passed=not failures.size and all(row['true_top_retained'] for row in top)
                       and all(row['true_top_retained'] for row in presence['observed_mask_checks']),
        gamma131=float(gamma), fp32_unit_roundoff=float(unit_roundoff),
        observed_coverage=dict(total_rows=ROWS, bounded_rows=int(np.count_nonzero(~fallback)),
            fallback_rows=int(np.count_nonzero(fallback)), invalid_weight_rows=int(np.count_nonzero(invalid_weight_rows)),
            uncovered_rows=int(failures.size), first_uncovered_ids=failures[:32].tolist(),
            fallback_reason='Nonfinite weight row or absolute-product/rounding enclosure cannot rule out FP32 intermediate overflow',
            all_dense_logits_finite=True),
        quantization_error_bound=statistics(quant_error[~fallback]),
        reference_rounding_bound=statistics(reference_error[~fallback]),
        interval_width=statistics((upper-lower)[~fallback]),
        observed_absolute_dense_minus_approximate=statistics(np.abs(actual-center)[~fallback]),
        top_candidates=top, presence_envelope=presence,
        auxiliary_activation=dict(quantized_dtype='int8', reconstruction_scale_dtype='FP64',
            network_activation_dtype_unchanged='FP32', maximum_residual=float(np.max(ex)),
            note='Only this offline routing approximation quantizes hidden state. Canonical inference activations are unchanged.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--hidden', type=Path, required=True, help='Exactly1024 little-endian FP32 values')
    parser.add_argument('--logits', type=Path, required=True, help='Exactly151936 little-endian dense FP32 values')
    parser.add_argument('--capture-manifest', type=Path, help='Optional completed coupled capture report; verify arrays, repeats, pack and listed build linkage')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows-per-chunk', type=int, default=256)
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT/'build/benchmark-status')
    args = parser.parse_args()
    if not 1 <= args.rows_per_chunk <= 4096:
        parser.error('Use1..4096 rows per temporary chunk')
    if args.output.exists():
        parser.error('Preserve earlier evidence; choose a new output')
    metadata = json.loads((args.guard_state_dir/'run.json').read_text())
    require(BenchmarkGate(args.guard_state_dir).status()['running'] and metadata.get('pid') == os.getppid(),
            'Run directly under benchmark_guard.py')
    weights, hidden_path, logits_path = args.weights.resolve(), args.hidden.resolve(), args.logits.resolve()
    report = dict(completed=False, passed=False, offline_only=True, inference_run=False,
        performance_claim_eligible=False, formal_proof=False, gpu=False, pytorch=False,
        model_shape=dict(vocabulary=ROWS, hidden=COLS, quantization_block=BLOCK), numpy_version=np.__version__,
        assumptions=[
            'Saved dense logits use exactly this hidden state and canonical FP16 output matrix.',
            'Reference head uses eight FP32 accumulation lanes:128FMA steps per lane followed by three addition levels.',
            'FP16-to-FP32 conversion preserves half subnormals (FZ16 disabled). The FP32-tiny allowance does not cover flushed FP16 weights.',
            'FP32 arithmetic is round-to-nearest, with correct FMA; no reassociation, reduced-precision products or hidden extra transforms.',
            'FP64 NumPy scalar/elementwise operations are correctly rounded; every interval accumulation is explicitly outward-rounded.',
            'Additive underflow allowance covers flushing FP32 subnormal inputs/results; possible intermediate overflow forces unbounded fallback.',
            'Coverage on supplied snapshots cannot prove these assumptions or certify a future runtime implementation.'],
        limitations=[
            'Dense exact top-k thresholds are used only for optimistic feasibility counts; production must discover a certified threshold through exact rescoring.',
            'Every canonical FP16 weight remains necessary for exact rescoring/fallback. The auxiliary cache increases resident storage.',
            'No kernel timings, throughput estimates, candidate-set runtime, or altered inference precision are reported.'])
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pending = args.output.with_suffix('.pending.json')
        pending.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        pending.replace(args.output)
    try:
        paths = [weights, Path(str(weights)+'.index'), Path(str(weights)+'.manifest.json'),
                 hidden_path, logits_path, Path(__file__).resolve(), ROOT/'tools/benchmark_guard.py']
        if args.capture_manifest:
            paths.append(args.capture_manifest.resolve())
        report['inputs'] = {str(path): sha(path) for path in paths}
        manifest = json.loads(Path(str(weights)+'.manifest.json').read_text())
        require(manifest.get('completed') is True and manifest.get('format') == 'QWCPU01'
                and manifest['binary']['sha256'] == report['inputs'][str(weights)]
                and manifest['index']['sha256'] == report['inputs'][str(weights)+'.index'], 'Weight manifest/hash mismatch')
        if args.capture_manifest:
            provenance, added = validate_capture(args.capture_manifest.resolve(), hidden_path, logits_path,
                                                 weights, report['inputs'])
            report['capture_provenance'] = provenance
            for path in added:
                report['inputs'][str(path)] = sha(path)
        else:
            report['capture_provenance'] = dict(verified=False,
                limitation='No coupled capture report supplied; same-position and build provenance remain assumptions.')
        require(hidden_path.stat().st_size == COLS*4 and logits_path.stat().st_size == ROWS*4, 'Wrong saved-array size')
        hidden, dense = np.fromfile(hidden_path, dtype='<f4'), np.fromfile(logits_path, dtype='<f4')
        if not np.all(np.isfinite(hidden)) or not np.all(np.isfinite(dense)):
            report['global_fallback'] = dict(rows=ROWS, reason='Nonfinite saved hidden state or dense output; no finite certificate')
            raise RuntimeError('Nonfinite input snapshot')
        matrix, offset = matrix_view(weights)
        report['output_matrix_offset'] = offset
        report['memory'] = dict(canonical_fp16_head_bytes=ROWS*COLS*2,
            proposed_int8_head_bytes=ROWS*COLS, proposed_three_fp32_metadata_bytes=ROWS*(COLS//BLOCK)*12,
            proposed_total_auxiliary_bytes=ROWS*COLS+ROWS*(COLS//BLOCK)*12,
            offline_rows_per_chunk=args.rows_per_chunk,
            note='Proposed persistent cache sizes, not measured RSS. Offline quantization uses temporary row chunks.')
        save()
        report['evaluation'] = evaluate(matrix, hidden, dense, args.rows_per_chunk)
        require(all(sha(Path(path)) == digest for path, digest in report['inputs'].items()), 'Immutable inputs changed')
        report.update(completed=True, passed=report['evaluation']['passed'])
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error)); raise
    finally:
        save()
    print(args.output)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
