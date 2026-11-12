"""Paired fused FP32 prefill against matched separate prefill or deployed FP64 prefill."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np
from benchmark_matched import SVEngine,make_prompt
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
                interpretation='Cross-precision differences are descriptive; finite full-vocabulary dumps and matching greedy IDs are required. No quality threshold is inferred.')


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
        'f17ae654f0c64ac47b63225a3406edc8bcd114cc86e67d56610090bf5d7c075d':'fp32_b1_fp64_prefill',
        'ecc943d7390c3e84a156ac04f05d95ee8d98dc88049aac8f4a80d039bd305545':'fp32_b1_fp32_prefill'}
    ref_hash=sha(paths[0]);candidate_hash=sha(paths[1])
    if ref_hash not in reference_profiles or candidate_hash!='013a56b2271120f6ca39cf38f74398b2d918e7d8c5e7dfe0b37679bd8317ec32':raise RuntimeError('Not reviewed frozen artifacts')
    if any(m['binary_sha256']!=sha(p) for m,p in zip(manifests,paths)):raise RuntimeError('Manifest hash mismatch')
    for m in manifests:
        for key in ('attention_qk2','attention_head_pairs','attention_value_pairs','static_kv','batch_attention'):
            if m['options'].get(key) is not True:raise RuntimeError('Unexpected attention/KV route')
    if not manifests[1]['options'].get('fp32_prefill_fma'):raise RuntimeError('FMA prefill not enabled')
    if manifests[0]['options'].get('fp32_prefill_fma',False):raise RuntimeError('Reference must not fuse FP32 prefill')
    engines=[SVEngine(checkpoint,8,protocol.tokenizer,paths[0],8,block_size=1,prefill_size=8,precision_profile=reference_profiles[ref_hash],verification_capacity=4),
             SVEngine(checkpoint,8,protocol.tokenizer,paths[1],8,block_size=1,prefill_size=8,precision_profile='fp32_b1_fp32_fma_prefill',verification_capacity=4)]
    report=dict(variant_options=[m["options"]for m in manifests],completed=False,gpu=False,pytorch=False,workers=8,tokens=args.tokens,
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        binaries=[dict(path=str(p),sha256=sha(p))for p in paths],model_provenance=model_provenance(checkpoint),
        source_hashes={p.name:sha(p)for p in (Path(__file__),Path(__file__).with_name('compare_attention_dot.py'),Path(__file__).with_name('benchmark_matched.py'),Path(__file__).with_name('rtl_model.py'))},
        reference_profile=reference_profiles[ref_hash],policy='Both reset KV, prefill prompt[:-1] in canonical chunks8 with B4 tails, then separate-rounded FP32 final-prompt/decode/vocabulary. Candidate hidden prefill is explicit fused FP32; reference is separate FP32 or deployed FP64. Combined attention and KV remain FP64. No prefix reuse or suffix.',
        limitations='Two resident models, alternating warmed requests; no OS cache eviction or hard isolation. Direct closed dumps preserve pools. Cross-profile logits differ and are descriptive only; each profile must reproduce its own full logit bits and both profiles must produce identical fixed-count greedy IDs. Separate real quality/replay/suffix gates required.',profiles=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        report['startup_seconds']=[engine.start()for engine in engines]
        report['engines']=[engine.metadata for engine in engines]
        models=[engine.model for engine in engines]
        with tempfile.TemporaryDirectory()as directory:
            for profile in profiles:
                prompt=make_prompt(protocol,profile)
                warm=[request(m,prompt,args.tokens)for m in models]
                if any(r['final_position']!=len(prompt)+args.tokens-1 for r in warm):raise AssertionError('Warm consumed-token position differs')
                expected=warm[0]['output_token_ids']
                if len(expected)!=args.tokens or warm[1]['output_token_ids']!=expected:raise AssertionError('Warm fixed-count greedy IDs differ')
                warm_bits=[read_dump_logits(m,Path(directory)/f'warm-{i}.bin').tobytes() for i,m in enumerate(models)]
                record=dict(profile=profile,prompt_ids=prompt,prompt_tokens=len(prompt),warmup=warm,
                            warm_full_logits=diagnostics(models,warm,directory),pairs=[])
                report['profiles'].append(record)
                for pair in range(args.pairs):
                    order=[0,1]if pair%2==0 else[1,0]
                    results={i:request(models[i],prompt,args.tokens)for i in order}
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
        for engine in engines:engine.close()
        args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
