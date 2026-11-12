#!/usr/bin/env python3
"""Compare a native capture to persisted oracle vectors at fixed numerical gates.

No model replay or inference. A completed oracle report may have passed=False;
its failed candidate comparison does not invalidate verified reference arrays.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_native_phase0 import guard
from check_native_logits import compare, load_capture, sha
from check_native_prefill_tail_oracle import require, save

ROOT = Path(__file__).resolve().parents[1]
VOCABULARY = 151936


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-report', type=Path, required=True)
    parser.add_argument('--oracle-report', type=Path, required=True)
    parser.add_argument('--run-index', type=int)
    parser.add_argument('--experiment-note', help='Explicit arithmetic/experiment description; not a conformance claim')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT/'build/benchmark-status')
    args = parser.parse_args()
    guarded = guard(args.guard_state_dir)
    output = args.output.resolve()
    require(not output.exists() and not output.with_suffix('.pending.json').exists(), 'Choose a fresh output')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(completed=False, passed=False, offline_only=True, inference_run=False,
        gpu=False, pytorch=False, performance_claim_eligible=False, guard_before=guarded,
        declared_tolerances=dict(atol=.01, rtol=0., rms_atol=.002, greedy_identity_required=True),
        experiment_note=args.experiment_note, inputs={}, comparisons=[], limitations=[
            'FP32 activation metadata describes stored outputs and does not certify internal accumulator precision; inspect the experiment note and build source hashes.',
            'Only the selected saved oracle positions and one specified native capture run are compared.',
            'No new numerical trajectory, inference, broad quality or performance evidence is generated.'])
    hashes = report['inputs']

    def freeze(path, digest=None):
        path = Path(path).resolve()
        key = str(path)
        if key not in hashes:
            hashes[key] = sha(path)
        require(digest is None or hashes[key] == digest, 'Artifact/hash mismatch: '+key)
        return path

    def recorded(mapping):
        require(isinstance(mapping, dict) and mapping, 'Missing hash provenance')
        for path, digest in mapping.items():
            require(isinstance(digest, str) and len(digest) == 64, 'Malformed digest')
            freeze(path, digest)

    def vector(path, digest):
        path = freeze(path, digest)
        require(path.stat().st_size == VOCABULARY*4, 'Wrong vector size')
        values = np.fromfile(path, dtype='<f4')
        require(values.shape == (VOCABULARY,) and np.isfinite(values).all(), 'Invalid vector')
        return values

    try:
        oracle_path = freeze(args.oracle_report)
        native_path = freeze(args.profile_report)
        oracle = json.loads(oracle_path.read_text())
        native = json.loads(native_path.read_text())
        for data in (oracle, native):
            require(data.get('completed') is True and not data.get('error') and not data.get('cleanup_error'),
                    'Incomplete/error evidence')
            recorded(data.get('inputs'))
            recorded(data.get('source_hashes'))
        require(oracle.get('gpu') is False and oracle.get('pytorch') is False, 'Non-CPU oracle evidence')
        capture = load_capture(native_path, args.run_index, hashes)
        require(capture['configuration'].get('weight_format') == 'QWCPU01', 'Expected original native capture')
        configuration = oracle['oracle']
        require(configuration.get('weight_precision') == 'f16' and configuration.get('kv_precision') == 'f16'
                and configuration.get('matrix_weight_rounding_only') is True, 'Oracle precision mismatch')
        storage = oracle.get('oracle_logits_storage', {})
        require(storage.get('dtype') == 'float32' and storage.get('byte_order') == 'little'
                and storage.get('shape') == [VOCABULARY] and storage.get('numerical_conversion') is False
                and storage.get('cache_consumption') is False
                and Path(storage.get('report', '')).resolve() == oracle_path, 'Invalid persistence contract')
        prompt, consumed = capture['prompt'], capture['consumed']
        require(storage.get('prompt_ids') == prompt and storage.get('consumed_continuation_ids') == consumed,
                'Native and oracle consumed trajectories differ')
        require(oracle.get('captures') and all(c.get('prompt') == prompt and c.get('consumed') == consumed
                for c in oracle['captures']), 'Oracle provenance trajectory mismatch')
        checkpoint = Path(configuration['checkpoint'])/'model.safetensors'
        checkpoint_digest = oracle['inputs'].get(str(checkpoint))
        require(checkpoint_digest is not None, 'Missing checkpoint binding')
        packs = []
        for path in native['inputs']:
            if path.endswith('.manifest.json'):
                pack = json.loads(Path(path).read_text())
                if pack.get('format') == 'QWCPU01':
                    packs.append((Path(path), pack))
        require(len(packs) == 1, 'Expected one native pack manifest')
        pack_path, pack = packs[0]
        require(pack.get('completed') is True and pack.get('tied_lm_head_verified') is True
                and pack['checkpoint']['sha256'] == checkpoint_digest, 'Pack/oracle checkpoint mismatch')
        for key in ('binary', 'index'):
            artifact = (pack_path.parent/pack[key]['path']).resolve()
            require(native['inputs'].get(str(artifact)) == pack[key]['sha256'], 'Pack artifact missing/binding mismatch')
            freeze(artifact, pack[key]['sha256'])
        # Ordinary captures freeze the executable, but older producers did not
        # include its adjacent build manifest. Validate that manifest now against
        # the executable digest already pinned at capture time, then freeze it.
        builds = []
        for artifact, digest in native['inputs'].items():
            candidate = Path(artifact).parent/'manifest.json'
            if not candidate.is_file():
                continue
            build = json.loads(candidate.read_text())
            if (build.get('binary') and Path(build['binary']).resolve() == Path(artifact).resolve()
                    and build.get('binary_sha256') == digest):
                builds.append((candidate, build))
        require(len(builds) == 1, 'Expected one executable-bound native build manifest')
        build_path, build = builds[0]
        require(build.get('completed') is True and build.get('gpu') is False and build.get('pytorch') is False,
                'Incomplete/non-CPU native build')
        freeze(build_path)
        freeze(build['binary'], build['binary_sha256'])
        report['native_build_manifest'] = build
        report['native_build_manifest_path'] = str(build_path.resolve())
        report['source_hashes'] = {str(path): sha(path) for path in (
            Path(__file__).resolve(), ROOT/'tools/check_native_logits.py', ROOT/'tools/check_native_prefill_tail_oracle.py',
            ROOT/'tests/test_full_model.py', ROOT/'tools/benchmark_native_phase0.py', ROOT/'tools/benchmark_guard.py')}
        report['provenance'] = dict(oracle_report=str(oracle_path), oracle_report_passed=oracle.get('passed'),
            profile_report=str(native_path), run_index=capture['run_index'], binary=build['binary'],
            checkpoint_sha256=checkpoint_digest, pack_manifest=str(pack_path),
            prompt_ids=prompt, consumed_continuation_ids=consumed)
        dumps = storage.get('dumps', [])
        require(dumps and len({d['step'] for d in dumps}) == len(dumps)
                and sorted(d['step'] for d in dumps) == oracle['selected_steps'], 'Missing/duplicate oracle steps')
        save(output, report)
        for row in dumps:
            step = row['step']
            require(type(step) is int and 0 <= step < len(capture['predictions']) and row.get('completed') is True
                    and row.get('context_tokens') == len(prompt)+step and row.get('consumed_continuation_count') == step
                    and row.get('shape') == [VOCABULARY] and row.get('dtype') == 'float32'
                    and row.get('byte_order') == 'little' and row.get('bytes') == VOCABULARY*4,
                    'Invalid oracle vector metadata')
            linked = [c for c in oracle['comparisons'] if c['step'] == step]
            require(linked and all(c['context_tokens'] == len(prompt)+step and c['oracle_logits_sha256'] == row['sha256']
                    for c in linked), 'Oracle vector not linked to original comparison')
            expected = vector(row['path'], row['sha256'])
            dump = capture['dumps'][step]
            actual_digest = hashes[str(Path(dump['path']).resolve())]
            actual = vector(dump['path'], actual_digest)
            result = compare(actual, expected, capture['predictions'][step], 5, .01, 0., .002)
            result.update(step=step, context_tokens=len(prompt)+step,
                          native_logits_sha256=actual_digest, oracle_logits_sha256=row['sha256'])
            report['comparisons'].append(result)
            save(output, report)
        require(all(sha(Path(p)) == h for p, h in hashes.items())
                and all(sha(Path(p)) == h for p, h in report['source_hashes'].items()), 'Frozen inputs/helpers changed')
        report.update(completed=True, passed=all(r['passed'] for r in report['comparisons']),
                      guard_after=guard(args.guard_state_dir))
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        save(output, report)
    print(output)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
