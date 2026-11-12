#!/usr/bin/env python3
"""Guarded standalone head probe: validate every selected snapshot before timing.

This is a sampler/head component experiment, not an LLM throughput benchmark.
The candidate obtains thresholds from its own exact seed rescoring, never from
saved dense outputs. Saved outputs are used solely by correctness checks.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import time

from analyze_native_phase0 import interval
from benchmark_native_phase0 import guard, host_snapshot, sha
from evaluate_native_head_bounds import validate_capture

ROOT = Path(__file__).resolve().parents[1]
VOCABULARY = 151936


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(path, value):
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    pending.replace(path)


def environment():
    result = dict(os.environ)
    removed = []
    for key in list(result):
        if key.startswith(('DYLD_', 'GGML_', 'LLAMA_ARG_')) or key == 'VECLIB_MAXIMUM_THREADS':
            removed.append(key); result.pop(key)
    return result, sorted(removed)


def build(directory, source, env):
    directory.mkdir(parents=True, exist_ok=False)
    copied = directory/'native_head_probe.cpp'
    shutil.copyfile(source, copied)
    binary = directory/'native_head_probe'
    compiler = shutil.which('clang++')
    require(compiler is not None, 'Apple clang++ is required')
    command = [compiler, '-std=c++20', '-O3', '-mcpu=native', '-flto=thin',
               '-ffp-contract=off', '-fno-math-errno', '-pthread', str(copied), '-o', str(binary)]
    manifest = dict(completed=False, gpu=False, pytorch=False, head_only=True,
                    binary=str(binary), source=str(source), source_sha256=sha(source),
                    copied_source=str(copied), command=command,
                    scalar_contraction='off for outward bounds; original dot uses explicit NEON FMA intrinsics')
    save(directory/'manifest.json', manifest)
    try:
        manifest['compiler'] = subprocess.check_output([compiler, '--version'], text=True, env=env)
        with (directory/'build.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=180)
        manifest['exit_code'] = result.returncode
        require(result.returncode == 0, 'Head probe compilation failed; inspect build.log')
        require(sha(copied) == sha(source) == manifest['source_sha256'], 'Source changed during build')
        linked = subprocess.check_output(['otool', '-L', str(binary)], text=True)
        libraries = '\n'.join(linked.splitlines()[1:]).lower()
        require(not any(name in libraries for name in ('metal', 'cuda', 'torch', 'llama', 'ggml')),
                'Unexpected accelerator/reference-runtime linkage')
        manifest.update(completed=True, binary_sha256=sha(binary), linked_libraries=linked)
    except BaseException as error:
        manifest['error'] = dict(type=type(error).__name__, message=str(error)); raise
    finally:
        save(directory/'manifest.json', manifest)
    return binary


def validate_result(raw, rounds, validation_only, threads, expected_seen):
    require(raw.get('completed') is True and raw.get('preflight_passed') is True
            and raw.get('head_only') is True and raw.get('gpu') is False and raw.get('pytorch') is False,
            'Head correctness/preflight did not complete')
    require(raw.get('rounds') == rounds and len(raw.get('worker_fpcr', [])) == threads
            and raw.get('seen_tokens') == expected_seen and raw.get('required_fpcr') == 0,
            'Actual head configuration differs from requested configuration')
    correctness = raw.get('correctness', [])
    require({row['mode'] for row in correctness} == {'greedy', 'top20'} and len(correctness) == 2,
            'Missing greedy/top20 correctness checks')
    for row in correctness:
        require(row.get('saved_dense_bit_identical') is True
                and row.get('runtime_top_ids_and_logits_bit_identical') is True
                and row['dense']['top'] == row['routed']['top'], 'Head exact-output gate failed')
        if not row['routed']['dense_fallback']:
            require(row.get('upper_coverage_checked') is True, 'Missing observed upper-bound check')
    count = 0 if validation_only else 4*rounds
    require(len(raw.get('measurements', [])) == count, 'Incomplete paired measurements')


def analyze(raw, expected, rounds):
    rows = raw['measurements']
    keyed = {(row['mode'], row['round'], row['configuration']): row for row in rows}
    require(len(keyed) == 4*rounds, 'Duplicate/missing measurement keys')
    rng = random.Random(20261112)
    resamples = [[rng.randrange(rounds) for _ in range(rounds)] for _ in range(10000)]
    analysis = []
    for mode in ('greedy', 'top20'):
        pairs = []
        for index in range(rounds):
            dense, routed = keyed[mode, index, 'dense'], keyed[mode, index, 'routed']
            require(dense['order'] == index % 2 and routed['order'] == 1-index % 2,
                    'Head timings did not alternate configuration order')
            for row in (dense, routed):
                require(type(row['seconds']) in (float, int) and math.isfinite(row['seconds']) and row['seconds'] > 0,
                        'Invalid head timing')
                require(row['result']['top'] == expected[mode], 'Scored head result differs from corpus validation')
            pairs.append(dict(round=index, dense_first=dense['order'] == 0,
                              dense_seconds=dense['seconds'], routed_seconds=routed['seconds'],
                              speedup=dense['seconds']/routed['seconds'],
                              rescored_rows=routed['result']['rescored_rows'],
                              dense_fallback=routed['result']['dense_fallback']))
        analysis.append(dict(mode=mode, pairs=pairs,
            dense_median_seconds=statistics.median(row['dense_seconds'] for row in pairs),
            routed_median_seconds=statistics.median(row['routed_seconds'] for row in pairs),
            paired_speedup=interval([row['speedup'] for row in pairs], resamples)))
    return dict(comparisons=analysis,
                bootstrap=dict(method='Paired-round percentile bootstrap of median dense/routed latency ratio',
                               confidence_level=.95, resamples=10000, seed=20261112),
                limitation='Unadjusted intervals; fixed saved hidden vectors and OS variability. No full-model speedup inference.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--build-directory', type=Path, help='New immutable standalone build directory')
    group.add_argument('--binary', type=Path, help='Existing verified standalone probe build')
    parser.add_argument('--weights', type=Path, default=ROOT/'build/native-cpu/weights.bin')
    parser.add_argument('--capture-manifest', type=Path, required=True)
    parser.add_argument('--run-index', type=int, default=0)
    parser.add_argument('--steps', type=int, nargs='+', help='Validate these capture steps; default all')
    parser.add_argument('--timed-steps', type=int, nargs='+', default=[0, 3, 7], help='Preselected subset to time after all validation')
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT/'build/benchmark-status')
    args = parser.parse_args()
    if not 10 <= args.rounds <= 100 or not 1 <= args.warmups <= 10 or not 1 <= args.threads <= 64:
        parser.error('Use10..100 preselected paired rounds,1..10 warmups,1..64 threads')
    guarded = guard(args.guard_state_dir)
    directory, weights = args.directory.resolve(), args.weights.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    output = directory/'report.json'
    source = ROOT/'tools/native_head_probe.cpp'
    env, removed = environment()
    report = dict(completed=False, passed=False, head_only=True, gpu=False, pytorch=False,
        performance_claim_eligible=False, formal_proof=False, validations=[], timings=[],
        rounds=args.rounds, warmups=args.warmups, threads=args.threads,
        guard_before=guarded, host_before=host_snapshot(), removed_environment_names=removed,
        timing_scope='Canonical FP16 vocabulary projection plus exact greedy/top20 selection versus auxiliary whole-row SDOT routing plus certified-threshold exact rescoring.',
        limitations=[
            'Standalone saved-hidden component probe; no full-network, browser TTFT, or LLM throughput claim.',
            'Top20 compares the actual presence-adjusted/temperature-scaled candidate set and exact logits; it does not run top-p sampling or RNG.',
            'All selected snapshots validate before timing any snapshot. Timed steps are selected before results are observed.',
            'Both configurations retain the same canonical weights and auxiliary allocation; cache construction/startup are outside timed calls.',
            'Original persistent pool and dot kernels adapted from repository engine/core.cpp; no external inference kernels.',
            'Scalar FP contraction is disabled for bounds; explicit NEON FMA preserves the canonical dot topology.',
            'Nonzero coordinator/worker FPCR or unavailable SDOT forces dense fallback, without changing FPCR.',
            'Empirical bound coverage and exact outputs supplement independent source review; they are not formal verification.'])
    try:
        capture_path = args.capture_manifest.resolve()
        capture = json.loads(capture_path.read_text())
        runs = [row for row in capture.get('runs', []) if row.get('index') == args.run_index]
        require(len(runs) == 1, 'Select an existing unique capture run')
        run = runs[0]
        steps = sorted(args.steps if args.steps is not None else range(capture['requested_steps']))
        timed_steps = list(args.timed_steps)
        require(steps and len(set(steps)) == len(steps) and len(set(timed_steps)) == len(timed_steps)
                and timed_steps and set(timed_steps) <= set(steps), 'Timed steps must be a distinct subset of validated steps')
        require(all(0 <= step < capture['requested_steps'] for step in steps), 'Capture step out of range')
        report.update(validation_steps=steps, preselected_timed_steps=timed_steps)
        paths = [weights, Path(str(weights)+'.index'), Path(str(weights)+'.manifest.json'), capture_path]
        snapshots = {}
        for step in steps:
            hidden = [row for row in run['hidden_dumps'] if row['step'] == step]
            logits = [row for row in run['logit_dumps'] if row['step'] == step]
            require(len(hidden) == len(logits) == 1, 'Missing coupled snapshot')
            snapshots[step] = dict(hidden=Path(hidden[0]['path']).resolve(), logits=Path(logits[0]['path']).resolve())
            paths.extend(snapshots[step].values())
        report['inputs'] = {str(path): sha(path) for path in paths}
        source_paths = [source, Path(__file__).resolve()] + [ROOT/'tools'/name for name in (
            'evaluate_native_head_bounds.py', 'benchmark_guard.py', 'analyze_native_phase0.py',
            'benchmark_native_phase0.py', 'benchmark_matched.py', 'chat_protocol.py', 'experiment.py')]
        report['source_hashes'] = {str(path): sha(path) for path in source_paths}
        pack = json.loads(Path(str(weights)+'.manifest.json').read_text())
        require(pack.get('completed') is True and pack.get('format') == 'QWCPU01'
                and pack['binary']['sha256'] == report['inputs'][str(weights)]
                and pack['index']['sha256'] == report['inputs'][str(weights)+'.index'], 'Canonical weight manifest mismatch')
        report['snapshots'] = []
        for step, snapshot in snapshots.items():
            provenance, added = validate_capture(capture_path, snapshot['hidden'], snapshot['logits'], weights, report['inputs'])
            for path in added:
                report['inputs'][str(path)] = sha(path)
            history = list(capture['prompt_ids']) + list(run['consumed_continuation_ids'][:step])
            require(len(history) == provenance['context_tokens']
                    and all(type(token) is int and 0 <= token < VOCABULARY for token in history), 'Invalid consumed history')
            mask = bytearray(VOCABULARY)
            for token in history:
                mask[token] = 1
            mask_path = directory/f'step-{step:03d}.seen.bin'; mask_path.write_bytes(mask)
            snapshot.update(mask=mask_path, seen_count=sum(mask))
            report['inputs'][str(mask_path)] = sha(mask_path)
            report['snapshots'].append(dict(step=step, provenance=provenance, consumed_ids=history,
                seen_mask_path=str(mask_path), seen_count=sum(mask)))
        save(output, report)
        binary = build(args.build_directory.resolve(), source, env) if args.build_directory else args.binary.resolve()
        manifest_path = binary.parent/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        require(manifest.get('completed') is True and manifest.get('head_only') is True
                and manifest.get('gpu') is False and manifest.get('pytorch') is False
                and manifest.get('binary_sha256') == sha(binary)
                and manifest.get('source_sha256') == sha(source), 'Standalone build provenance mismatch')
        copied = Path(manifest['copied_source'])
        require(sha(copied) == manifest['source_sha256'], 'Frozen standalone source changed')
        report['build'] = manifest
        for path in (binary, manifest_path, copied):
            report['inputs'][str(path)] = sha(path)
        def execute(step, validation_only):
            guard(args.guard_state_dir)
            snapshot = snapshots[step]
            run_dir = directory/('validation' if validation_only else 'timing')/f'step-{step:03d}'
            run_dir.mkdir(parents=True)
            raw_path = run_dir/'raw.json'
            command = [str(binary), '--weights', str(weights), '--hidden', str(snapshot['hidden']),
                       '--logits', str(snapshot['logits']), '--seen-mask', str(snapshot['mask']),
                       '--output', str(raw_path), '--threads', str(args.threads), '--rounds', str(args.rounds),
                       '--warmups', str(args.warmups), '--validation-only', str(int(validation_only))]
            record = dict(step=step, command=command)
            report['validations' if validation_only else 'timings'].append(record)
            save(output, report)
            with (run_dir/'stdout.log').open('x') as out, (run_dir/'stderr.log').open('x') as err:
                start = time.perf_counter()
                result = subprocess.run(command, stdout=out, stderr=err, env=env, timeout=600)
                record.update(returncode=result.returncode, process_wall_seconds=time.perf_counter()-start)
            require(result.returncode == 0, 'Head probe failed; inspect '+str(run_dir/'stderr.log'))
            raw = json.loads(raw_path.read_text())
            validate_result(raw, args.rounds, validation_only, args.threads, snapshot['seen_count'])
            record.update(raw=raw, raw_path=str(raw_path), raw_sha256=sha(raw_path))
            save(output, report)
            return raw
        expected = {}
        for step in steps:
            raw = execute(step, True)
            expected[step] = {row['mode']: row['dense']['top'] for row in raw['correctness']}
        report['all_selected_snapshots_passed_before_timing'] = True
        save(output, report)
        for step in timed_steps:
            raw = execute(step, False)
            report['timings'][-1]['analysis'] = analyze(raw, expected[step], args.rounds)
            save(output, report)
        require(all(sha(Path(path)) == digest for path, digest in report['inputs'].items())
                and all(sha(Path(path)) == digest for path, digest in report['source_hashes'].items()), 'Frozen input or helper changed')
        report.update(completed=True, passed=True, host_after=host_snapshot(), guard_after=guard(args.guard_state_dir))
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error)); raise
    finally:
        save(output, report)
    print(output)


if __name__ == '__main__':
    main()
