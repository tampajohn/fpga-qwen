#!/usr/bin/env python3
"""Offline fixed-gate comparison of saved oracle vectors and tail-isolation arrays.

No network replay, model construction, cache consumption by inference, or changed
numerical limits. An oracle report may have passed=False because an earlier
candidate failed; its completed/error-free saved reference vectors remain valid.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_native_phase0 import guard
from check_native_logits import compare, sha

ROOT = Path(__file__).resolve().parents[1]
VOCABULARY = 151936


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def save(path, report):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle-report', type=Path, required=True)
    parser.add_argument('--tail-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT/'build/benchmark-status')
    args = parser.parse_args()
    guarded = guard(args.guard_state_dir)
    output = args.output.resolve()
    require(not output.exists() and not output.with_suffix('.pending.json').exists(), 'Choose a fresh output; preserve earlier evidence')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(completed=False, passed=False, offline_only=True, inference_run=False,
        gpu=False, pytorch=False, performance_claim_eligible=False, guard_before=guarded,
        declared_tolerances=dict(atol=.01, rtol=0., rms_atol=.002, greedy_identity_required=True),
        inputs={}, comparisons=[],
        limitations=['Compares existing full-vocabulary FP32 arrays only; no inference or new oracle trajectory is run.',
                     'A passing or failing sampled final-tail result does not establish the cause of earlier numerical drift.',
                     'Only selected saved oracle positions are checked; no claim of full-context or full-generation conformance.'])
    hashes = report['inputs']
    def freeze(path, expected=None):
        path = Path(path).resolve()
        key = str(path)
        if key not in hashes:
            hashes[key] = sha(path)
        require(expected is None or hashes[key] == expected, 'Artifact/hash mismatch: '+key)
        return path
    def verify_recorded_hashes(mapping, label):
        require(isinstance(mapping,dict) and mapping, 'Missing '+label+' hash provenance')
        for path, digest in mapping.items():
            require(isinstance(digest,str) and len(digest)==64, 'Malformed digest: '+label)
            freeze(path,digest)
    def vector(path, digest, count):
        path=freeze(path,digest)
        require(path.stat().st_size==count*4, 'Array size mismatch: '+str(path))
        values=np.fromfile(path,dtype='<f4')
        require(values.shape==(count,) and np.isfinite(values).all(), 'Invalid/nonfinite saved vector')
        return values
    try:
        oracle_path=freeze(args.oracle_report);tail_path=freeze(args.tail_report)
        oracle=json.loads(oracle_path.read_text());tail=json.loads(tail_path.read_text())
        for name, data in (('oracle',oracle),('tail',tail)):
            require(data.get('completed')is True and not data.get('error') and not data.get('cleanup_error')
                    and data.get('gpu')is False and data.get('pytorch')is False,
                    name+' report is not completed/error-free CPU evidence')
            verify_recorded_hashes(data.get('inputs'),name+' inputs')
            verify_recorded_hashes(data.get('source_hashes'),name+' source')
        report['source_hashes']={str(path):sha(path)for path in (
            Path(__file__).resolve(),ROOT/'tools/check_native_logits.py',ROOT/'tests/test_full_model.py',
            ROOT/'tools/benchmark_native_phase0.py',ROOT/'tools/benchmark_guard.py')}
        storage=oracle.get('oracle_logits_storage',{})
        require(storage.get('dtype')=='float32' and storage.get('byte_order')=='little'
                and storage.get('shape')==[VOCABULARY] and storage.get('numerical_conversion')is False
                and storage.get('cache_consumption')is False
                and Path(storage.get('report','')).resolve()==oracle_path, 'Missing or incompatible oracle persistence metadata')
        configuration=oracle['oracle']
        require(configuration.get('weight_precision')=='f16' and configuration.get('kv_precision')=='f16'
                and configuration.get('matrix_weight_rounding_only')is True, 'Oracle precision differs from native F16/KV-F16 contract')
        prompt=storage.get('prompt_ids');consumed=storage.get('consumed_continuation_ids')
        require(isinstance(prompt,list) and len(prompt)==3968 and isinstance(consumed,list) and len(consumed)==1
                and all(type(token)is int and 0<=token<VOCABULARY for token in prompt+consumed), 'Unexpected fixed oracle trajectory')
        require(tail.get('prompt_ids')==prompt and tail.get('teacher_continuation_ids')==consumed
                and tail.get('prefix_tokens')==3584 and tail.get('tail_tokens')==384
                and tail.get('baseline_a_original_capture_bit_identical')is True
                and tail.get('canonical_prefix_primed_once')is True, 'Tail report trajectory/canonical reference mismatch')
        for capture in oracle.get('captures',[]):
            require(capture.get('prompt')==prompt and capture.get('consumed')==consumed, 'Oracle capture trajectory mismatch')
        require(oracle.get('captures'), 'Missing oracle capture provenance')
        # Bind the tail pack to the exact checkpoint used by the saved oracle,
        # even if that oracle replay originally compared a GGUF capture only.
        checkpoint=Path(configuration['checkpoint'])/'model.safetensors'
        checkpoint_digest=oracle['inputs'].get(str(checkpoint))
        require(checkpoint_digest is not None, 'Oracle checkpoint hash is missing')
        pack_matches=[]
        for path in tail['inputs']:
            if path.endswith('.manifest.json'):
                pack=json.loads(Path(path).read_text())
                if pack.get('format')=='QWCPU01':
                    pack_matches.append((Path(path),pack))
        require(len(pack_matches)==1, 'Expected one native weight-pack manifest')
        pack_path,pack=pack_matches[0]
        require(pack.get('completed')is True and pack.get('tied_lm_head_verified')is True
                and pack['checkpoint']['sha256']==checkpoint_digest, 'Native pack and oracle checkpoint differ')
        for key in ('binary','index'):
            artifact=(pack_path.parent/pack[key]['path']).resolve()
            require(str(artifact)in tail['inputs'] and tail['inputs'][str(artifact)]==pack[key]['sha256'], 'Native pack artifact binding mismatch')
            freeze(artifact,pack[key]['sha256'])
        report['provenance']=dict(oracle_report=str(oracle_path),oracle_report_passed=oracle.get('passed'),
            tail_report=str(tail_path),checkpoint_sha256=checkpoint_digest,pack_manifest=str(pack_path),
            prompt_ids=prompt,teacher_continuation_ids=consumed,
            diagnostic_binary=tail.get('diagnostic_binary'),reference_binary=tail.get('reference_binary'))
        dumps=storage.get('dumps',[])
        require(dumps and len({row['step']for row in dumps})==len(dumps), 'Missing/duplicate saved oracle steps')
        require(sorted(row['step']for row in dumps)==oracle['selected_steps'], 'Saved oracle steps differ from declared selection')
        expected={}
        for row in dumps:
            step=row['step']
            require(type(step)is int and step in (0,1) and row.get('completed')is True
                    and row.get('context_tokens')==len(prompt)+step
                    and row.get('consumed_continuation_count')==step
                    and row.get('shape')==[VOCABULARY] and row.get('dtype')=='float32'
                    and row.get('byte_order')=='little' and row.get('bytes')==VOCABULARY*4,
                    'Invalid saved oracle position/shape/type')
            linked=[c for c in oracle['comparisons']if c['step']==step]
            require(linked and all(c['context_tokens']==len(prompt)+step and c['oracle_logits_sha256']==row['sha256']for c in linked), 'Saved oracle dump not linked to original comparisons')
            expected[step]=vector(row['path'],row['sha256'],VOCABULARY)
        branches=tail.get('branches',[])
        require(len(branches)==2 and {b['name']for b in branches}=={'A-prefill','B-steps'}, 'Missing tail branches')
        save(output,report)
        for branch in branches:
            require(branch.get('teacher_forced_tail_ids')==prompt[3584:] and len(branch.get('rows',[]))==2,
                    'Tail branch inputs/rows differ')
            require({row['step']for row in branch['rows']}=={0,1}, 'Duplicate/missing branch positions')
            for row in branch['rows']:
                step=row['step']
                require(row.get('context_tokens')==len(prompt)+step and type(row.get('predicted_id'))is int
                        and 0<=row['predicted_id']<VOCABULARY, 'Invalid native position/prediction')
                for kind,count in (('logits',VOCABULARY),('hidden',1024)):
                    artifact=row['arrays'][kind]
                    require(artifact.get('count')==count and artifact.get('dtype')=='float32'
                            and artifact.get('byte_order')=='little', 'Invalid native dump metadata')
                    require(tail['inputs'].get(str(Path(artifact['path']).resolve()))==artifact['sha256'], 'Native dump missing from recorded inputs')
                    actual=vector(artifact['path'],artifact['sha256'],count)
                    if kind=='logits' and step in expected:
                        result=compare(actual,expected[step],row['predicted_id'],5,.01,0.,.002)
                        result.update(branch=branch['name'],step=step,context_tokens=len(prompt)+step,
                                      native_logits_sha256=artifact['sha256'],
                                      oracle_logits_sha256=next(d['sha256']for d in dumps if d['step']==step))
                        report['comparisons'].append(result)
                save(output,report)
        report['branch_summary']={name:dict(passed=all(r['passed']for r in report['comparisons']if r['branch']==name),
            max_absolute_error=max(r['max_absolute_error']for r in report['comparisons']if r['branch']==name),
            max_position_rms_error=max(r['rms_error']for r in report['comparisons']if r['branch']==name))for name in ('A-prefill','B-steps')}
        require(all(sha(Path(path))==digest for path,digest in hashes.items())
                and all(sha(Path(path))==digest for path,digest in report['source_hashes'].items()), 'Frozen evidence/helper changed')
        report.update(completed=True,passed=all(r['passed']for r in report['comparisons']),guard_after=guard(args.guard_state_dir))
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error));raise
    finally:
        save(output,report)
    print(output)
    if not report['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
