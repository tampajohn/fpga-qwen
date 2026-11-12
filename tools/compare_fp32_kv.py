"""Paired combined attention with FP64 versus FP32 KV storage."""
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
    required=('attention_qk2','attention_head_pairs','attention_value_pairs','batch_attention')
    if not all(m['options'].get(k) is True for m in manifests for k in required):raise RuntimeError('Both artifacts require identical combined attention routes')
    if not manifests[0]['options'].get('static_kv') or manifests[0]['options'].get('fp32_kv',False):raise RuntimeError('Reference must have FP64 static KV')
    if not manifests[1]['options'].get('fp32_kv') or manifests[1]['options'].get('static_kv'):raise RuntimeError('Candidate must have FP32 KV only')
    if manifests[1]['fp32']['kv_payload_bytes']!=939524096 or manifests[1]['fp32']['kv_storage']!='fp32':raise RuntimeError('Wrong candidate KV payload manifest')
    ignored={'fp32_kv','static_kv'}
    options=[{k:v for k,v in m['options'].items()if k not in ignored}for m in manifests]
    if options[0]!=options[1]:raise RuntimeError('Other build options differ')
    frozen=('f17ae654f0c64ac47b63225a3406edc8bcd114cc86e67d56610090bf5d7c075d','c7a41fd46cb72ae0d2b1789c5be4b2926b8b7bb41ca440b0ecbe52fe6b3f5624')
    if tuple(sha(p)for p in paths)!=frozen:raise RuntimeError('Not the reviewed frozen artifact pair')
    if any(m['binary_sha256']!=sha(p)for m,p in zip(manifests,paths)):raise RuntimeError('Manifest binary mismatch')
    engines=[SVEngine(checkpoint,8,protocol.tokenizer,paths[0],8,block_size=1,prefill_size=8,precision_profile='fp32_b1_fp64_prefill',verification_capacity=4),
             SVEngine(checkpoint,8,protocol.tokenizer,paths[1],8,block_size=1,prefill_size=8,precision_profile='fp32_b1_fp64_prefill_fp32_kv',verification_capacity=4)]
    report=dict(variant_options=[m["options"]for m in manifests],completed=False,gpu=False,pytorch=False,workers=8,tokens=args.tokens,
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        binaries=[dict(path=str(p),sha256=sha(p))for p in paths],model_provenance=model_provenance(checkpoint),
        source_hashes={p.name:sha(p)for p in (Path(__file__),Path(__file__).with_name('compare_attention_dot.py'),Path(__file__).with_name('benchmark_matched.py'),Path(__file__).with_name('rtl_model.py'))},
        policy='Both requests reset KV and prefill prompt[:-1] in canonical chunks8 with B4 tails; Both use FP64 prefill projections and FP32 scalar final-prompt/decode. Both retain batched prefill attention; Both use paired-query prefill QK and eligible paired-head scalar QK/V. Reference KV storage is FP64; candidate KV storage is FP32 with SV narrowing and exact FP64 read widening. No prefix reuse or suffix verification.',
        limitations='Two resident models; sequential warmed inference in alternating order. Direct acknowledged/closed dumps preserve worker pools after warmup. No OS cache eviction or hard host isolation. Fixed greedy IDs must match; full-logit differences are descriptive across KV precision. Same-route replay and sampled verification were gated separately. Payload claim is generated array bytes, not measured RSS.',profiles=[])
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
                record=dict(profile=profile,prompt_ids=prompt,prompt_tokens=len(prompt),warmup=warm,
                            warm_full_logits=diagnostics(models,warm,directory),pairs=[])
                report['profiles'].append(record)
                for pair in range(args.pairs):
                    order=[0,1]if pair%2==0 else[1,0]
                    results={i:request(models[i],prompt,args.tokens)for i in order}
                    if any(r['output_token_ids']!=expected or len(r['output_token_ids'])!=args.tokens for r in results.values()):raise AssertionError('Scored fixed-count greedy IDs differ')
                    if any(r['final_position']!=len(prompt)+args.tokens-1 for r in results.values()):raise AssertionError('Scored consumed-token position differs')
                    ratios={k:results[0][k]/results[1][k]for k in ('prefill_seconds','ttft_seconds','decode_seconds','total_seconds')}
                    record['pairs'].append(dict(index=pair,order=order,baseline=results[0],candidate=results[1],ratios=ratios))
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
