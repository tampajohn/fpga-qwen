"""Paired B8 engines differing only in attention dot reduction; no host neural math."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import tempfile
from evaluate_projection_quality import read_dump_logits
from benchmark_blocks import compare_logits

from rtl_model import RTLModel
from chat_protocol import ChatProtocol
from benchmark_matched import make_prompt


def request(model,prompt,count):
    model.reset()
    start=time.perf_counter()
    for i in range(0,len(prompt)-1,model.prefill_size):
        model.prefill(prompt[i:min(i+model.prefill_size,len(prompt)-1)])
    prefilled=time.perf_counter()
    token,position,_=model.token(prompt[-1]);ids=[token]
    first=time.perf_counter()
    for _ in range(count-1):
        token,position,_=model.token(token);ids.append(token)
    end=time.perf_counter()
    if any(t in (151643,151645) for t in ids):raise RuntimeError('EOS invalidates fixed-workload comparison')
    return dict(prefill_seconds=prefilled-start,ttft_seconds=first-start,
                decode_seconds=end-first,decode_tokens_per_second=(count-1)/(end-first),
                total_seconds=end-start,output_token_ids=ids,final_position=position)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=3);p.add_argument('--tokens',type=int,default=32)
    p.add_argument('--profiles',default='short,long');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.pairs<=10 or not 2<=a.tokens<=128:p.error('Invalid sample count')
    paths=[a.reference.resolve(),a.candidate.resolve()]
    models=[RTLModel(binary=b,workers=8,lanes=8,block_size=4,prefill_size=s) for b,s in zip(paths,(8,8))]
    protocol=ChatProtocol(models[0].model)
    report=dict(completed=False,workers=8,verification_capacity=4,prefill_capacities=[8,8],
                arithmetic='BF16 weights; SV FP64; candidate attention QK uses eight partial sums instead of serial reduction',
                limitation='Two resident models, sequential inference, logical KV reset each request; warm OS caches, no eviction. Protocol and copies included.',
                binaries=[dict(path=str(b),sha256=hashlib.sha256(b.read_bytes()).hexdigest()) for b in paths],profiles=[])
    report['checkpoint_sha256']=hashlib.file_digest((models[0].model/'model.safetensors').open('rb'),'sha256').hexdigest()
    report['source_hashes']={name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
                             for name in ('tools/compare_attention_dot.py','tools/rtl_model.py')}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        startup=[]
        for m in models:
            t=time.perf_counter();m.start();startup.append(time.perf_counter()-t)
        report['startup_seconds']=startup
        for profile in a.profiles.split(','):
            if profile not in ('short','medium','long'):raise ValueError('Invalid profile')
            prompt=make_prompt(protocol,profile)
            record=dict(profile=profile,prompt_ids=prompt,prompt_tokens=len(prompt),pairs=[])
            report['profiles'].append(record)
            warm=[request(m,prompt,a.tokens) for m in models];record['warmup']=warm
            with tempfile.TemporaryDirectory() as directory:
                arrays=[read_dump_logits(m,Path(directory)/str(i)) for i,m in enumerate(models)]
                record['warm_full_logit_gate']=compare_logits(*arrays,1e-6)
            expected=warm[0]['output_token_ids']
            if warm[1]['output_token_ids']!=expected:raise AssertionError('Warm greedy outputs differ')
            for i in range(a.pairs):
                order=[0,1] if i%2==0 else [1,0]
                results={j:request(models[j],prompt,a.tokens) for j in order}
                if any(r['output_token_ids']!=expected for r in results.values()):raise AssertionError('Scored outputs differ')
                ratios={key:results[0][key]/results[1][key] for key in ('prefill_seconds','ttft_seconds','decode_seconds','total_seconds')}
                record['pairs'].append(dict(order=order,baseline=results[0],candidate=results[1],ratios=ratios))
                a.output.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(dict(profile=profile,pair=i,ratios=ratios)),flush=True)
            record['median_ratios']={k:statistics.median(x['ratios'][k] for x in record['pairs']) for k in ratios}
        report['completed']=True
    finally:
        for m in models:m.close()
        a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
