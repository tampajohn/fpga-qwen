"""Paired-query V accumulation versus deployed fused phase backend; identical arithmetic required."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np
from benchmark_matched import make_prompt
from rtl_model import RTLModel
from check_phase_pools_real import telemetry
from compare_attention_dot import request
from chat_protocol import ChatProtocol
from experiment import locate_model
from evaluate_projection_quality import validate_logits,model_provenance


def read_dump_logits(model,path):
    # Both evaluated artifacts close/check the file before acknowledging.
    # Historical lane barriers recreate pools and would invalidate warmup.
    path=Path(path)
    if model.command(f'dump {path}')!='dumped':raise RuntimeError('Invalid dump acknowledgement')
    if path.stat().st_size!=151936*8:raise ValueError('Wrong full-vocabulary dump byte count')
    return validate_logits(np.fromfile(path,dtype=np.float64))


def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def diagnostics(models,outputs,directory):
    arrays=[read_dump_logits(m,Path(directory)/f'{i}.bin')for i,m in enumerate(models)]
    greedy=[int(np.argmax(a))for a in arrays]
    if greedy!=[r['output_token_ids'][-1]for r in outputs]:raise AssertionError('Dump alignment differs from final prediction')
    delta=arrays[1]-arrays[0]
    return dict(max_absolute_logit_error=float(np.max(np.abs(delta))),rms_logit_error=float(np.sqrt(np.mean(delta*delta))),
                full_vocabulary_size=len(delta),top1_ids=greedy,bit_identical=bool(np.array_equal(arrays[0].view(np.uint64),arrays[1].view(np.uint64))),
                interpretation='Same-profile full-vocabulary bit equality and matching greedy IDs are required; no tolerance gate.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--pairs',type=int,default=3);parser.add_argument('--tokens',type=int,default=32)
    parser.add_argument('--profiles',default='short,long');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();profiles=args.profiles.split(',')
    if not 1<=args.pairs<=10 or not 2<=args.tokens<=128 or any(p not in ('short','long')for p in profiles):parser.error('Invalid workload')
    checkpoint=locate_model(None);protocol=ChatProtocol(checkpoint)
    paths=[args.reference.resolve(),args.candidate.resolve()]
    manifests=[json.loads((p.parent.parent/'manifest.json').read_text())for p in paths]
    reference_profiles={
        '6c9ab44bf49b9f35ebf1a86897c284640041fb2a1a811f38184bff306f9e1b89':'fp32_b1_fp32_fma_prefill'}
    ref_hash=sha(paths[0]);candidate_hash=sha(paths[1])
    if ref_hash not in reference_profiles or candidate_hash!='5153c95534e5aae1689c426f81925d270fcb2d47c2568dcc258dbf0fdd244bf4':raise RuntimeError('Not reviewed frozen artifacts')
    if any(m['binary_sha256']!=sha(p) for m,p in zip(manifests,paths)):raise RuntimeError('Manifest hash mismatch')
    for m in manifests:
        for key in ('attention_qk2','attention_head_pairs','attention_value_pairs','static_kv','batch_attention'):
            if m['options'].get(key) is not True:raise RuntimeError('Unexpected attention/KV route')
    if not manifests[1]['options'].get('fp32_prefill_fma'):raise RuntimeError('FMA prefill not enabled')
    reference_fma=reference_profiles[ref_hash]=='fp32_b1_fp32_fma_prefill'
    if bool(manifests[0]['options'].get('fp32_prefill_fma'))!=reference_fma:raise RuntimeError('Wrong reference arithmetic')
    if not manifests[0]['options'].get('phase_pools_experiment'):raise RuntimeError('Wrong reference pool build')
    if not manifests[1]['options'].get('phase_pools_experiment') or not manifests[1]['options'].get('fp32_fma_canonical_prefix'):raise RuntimeError('Missing final candidate capabilities')
    assert manifests[1]['options'].get('attention_query_value_pairs') is True
    assert not manifests[0]['options'].get('attention_query_value_pairs',False)
    selectors=['PREFILL_12','PREFILL_12']
    profiles_expected=[reference_profiles[ref_hash],'fp32_b1_fp32_fma_prefill']
    policies=['fp32_fma_b8_aligned_v1','fp32_fma_b8_aligned_v1']
    models=[RTLModel(model=checkpoint,binary=path,workers=8,lanes=8,block_size=4,prefill_size=8)for path in paths]
    report=dict(variant_options=[m["options"]for m in manifests],completed=False,gpu=False,pytorch=False,workers=8,tokens=args.tokens,
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        binaries=[dict(path=str(p),sha256=sha(p))for p in paths],model_provenance=model_provenance(checkpoint),
        source_hashes={p.name:sha(p)for p in (Path(__file__),Path(__file__).with_name('compare_attention_dot.py'),Path(__file__).with_name('benchmark_matched.py'),Path(__file__).with_name('rtl_model.py'),Path(__file__).with_name('check_phase_pools_real.py'))},
        reference_profile=reference_profiles[ref_hash],cross_profile_bit_exact_required=reference_fma,resident_worker_threads=[20,20],prefill_workers=[12,12],decode_workers=[8,8],selectors=selectors,policy='Fresh logical KV reset before every request; canonical B8 prefill, separate FP32 scalar decode/head; no cache reuse or suffix. Candidate uses fused FP32 prefill with12 workers and8 decode; reference is deployed fused FP32 prefill with12/8; candidate shares paired-query value loads while preserving each output reduction order. Two separately linked resident models. No placement guarantees.',
        limitations='Two resident models, alternating warmed requests; no OS cache eviction or hard isolation. Direct closed dumps preserve pools. Both profiles must have identical full logit bits; each profile must reproduce its own full logit bits and both profiles must produce identical fixed-count greedy IDs. Separate real quality/replay/suffix gates required.',profiles=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        report['startup_seconds']=[];report['capabilities']=[]
        for i,model in enumerate(models):
            model.phase_pool_selector=selectors[i];start=time.perf_counter();model.start()
            actual=json.loads(model.command('precision'))
            expected=dict(precision_profile=profiles_expected[i],scalar_projection='fp32',prefill_projection='fp32' if profiles_expected[i]=='fp32_b1_fp32_fma_prefill' else 'fp64',verification_projection='fp32',attention_kv='fp64',max_verification_block=4,prefix_reuse_supported=False,canonical_prefix_policy=policies[i])
            if profiles_expected[i]=='fp32_b1_fp32_fma_prefill':expected.update(scalar_accumulation='separate',prefill_accumulation='fused',verification_accumulation='separate',vocabulary_projection='fp32',vocabulary_accumulation='separate')
            if not isinstance(actual,dict) or any(type(actual.get(k)) is not type(v) or actual[k]!=v for k,v in expected.items()):raise RuntimeError('Runtime profile mismatch')
            report['capabilities'].append(actual);report['startup_seconds'].append(time.perf_counter()-start)
            if selectors[i]:telemetry(model,True)
        def guarded_request(index,prompt):
            model=models[index];before=telemetry(model,True)if selectors[index]else None
            result=request(model,prompt,args.tokens)
            if before:
                after=telemetry(model,True);delta=[[after['dispatch_counts'][p][k]-before['dispatch_counts'][p][k]for k in range(3)]for p in range(2)]
                if delta[0][0]<=0 or delta[1][1]<=0 or any(delta[p][k]for p,k in ((0,1),(0,2),(1,0),(1,2))):raise RuntimeError('Phase purpose mismatch')
                result.update(phase_before=before,phase_after=after,phase_dispatch_delta=delta)
            return result
        with tempfile.TemporaryDirectory()as directory:
            for profile in profiles:
                prompt=make_prompt(protocol,profile)
                warm=[guarded_request(i,prompt)for i in range(2)]
                if any(r['final_position']!=len(prompt)+args.tokens-1 for r in warm):raise AssertionError('Warm consumed-token position differs')
                expected=warm[0]['output_token_ids']
                if len(expected)!=args.tokens or warm[1]['output_token_ids']!=expected:raise AssertionError('Warm fixed-count greedy IDs differ')
                warm_bits=[read_dump_logits(m,Path(directory)/f'warm-{i}.bin').tobytes() for i,m in enumerate(models)]
                if reference_fma and warm_bits[0]!=warm_bits[1]:raise AssertionError('Same FMA arithmetic changed warm logit bits')
                record=dict(profile=profile,prompt_ids=prompt,prompt_tokens=len(prompt),warmup=warm,
                            warm_full_logits=diagnostics(models,warm,directory),pairs=[])
                report['profiles'].append(record)
                for pair in range(args.pairs):
                    order=[0,1]if pair%2==0 else[1,0]
                    results={i:guarded_request(i,prompt)for i in order}
                    if any(r['output_token_ids']!=expected or len(r['output_token_ids'])!=args.tokens for r in results.values()):raise AssertionError('Scored fixed-count greedy IDs differ')
                    if any(r['final_position']!=len(prompt)+args.tokens-1 for r in results.values()):raise AssertionError('Scored consumed-token position differs')
                    for i,m in enumerate(models):
                        if read_dump_logits(m,Path(directory)/f'replay-{i}.bin').tobytes()!=warm_bits[i]:raise AssertionError('Within-profile full logits changed')
                    ratios={k:results[0][k]/results[1][k]for k in ('prefill_seconds','ttft_seconds','decode_seconds','total_seconds')}
                    record['pairs'].append(dict(index=pair,order=order,baseline=results[0],candidate=results[1],ratios=ratios,within_profile_full_logits_exact=True))
                    args.output.write_text(json.dumps(report,indent=2)+'\n')
                    print(json.dumps(dict(profile=profile,pair=pair,ratios=ratios)),flush=True)
                record['median_ratios']={k:statistics.median(p['ratios'][k]for p in record['pairs'])for k in ratios}
                record['ratio_ranges']={k:[min(p['ratios'][k]for p in record['pairs']),max(p['ratios'][k]for p in record['pairs'])]for k in ratios}
                record['final_full_logits']=diagnostics(models,[results[0],results[1]],directory)
        if any(sha(path)!=record['sha256'] for path,record in zip(paths,report['binaries'])):raise RuntimeError('Binary changed during evaluation')
        report['completed']=True
    finally:
        for model in models:model.close()
        args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
