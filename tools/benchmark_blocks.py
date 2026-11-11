"""Independent real RTL block protocol gates and paired prefill timing (no GPU)."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np

try:
    from .rtl_model import RTLModel
    from .benchmark_matched import make_prompt
    from .chat_protocol import ChatProtocol
except ImportError:
    from rtl_model import RTLModel
    from benchmark_matched import make_prompt
    from chat_protocol import ChatProtocol


class BlockProtocolModel(RTLModel):
    """Raw protocol adapter independent of production batching/orchestration."""
    def block(self,tokens,logits=False,uniforms=None,presence_penalty=1.5,cancel=None):
        tokens=list(tokens)
        if not 1<=len(tokens)<=4: raise ValueError('Block count must be1..4')
        if any(type(t) is not int or not 0<=t<151936 for t in tokens): raise ValueError('Invalid token')
        sample=uniforms is not None
        uniforms=list(uniforms) if sample else [0.0]*len(tokens)
        if len(uniforms)!=len(tokens) or any(not 0<=u<1 for u in uniforms): raise ValueError('Invalid uniforms')
        if not 0<=presence_penalty<=2: raise ValueError('Invalid penalty')
        command=' '.join(['block',str(len(tokens)),str(int(logits)),str(int(sample)),str(presence_penalty),
                          *map(str,tokens),*(format(u,'.17g') for u in uniforms)])
        fields=self.command(command,cancel).split()
        if fields[:2]!=['block',str(len(tokens))] or len(fields)!=2+3*len(tokens):
            raise RuntimeError(f'Invalid block response: {fields}')
        return [(int(fields[i]),int(fields[i+1]),float(fields[i+2])) for i in range(2,len(fields),3)]


def dump_logits(model,path):
    if model.command(f'dump {path}')!='dumped': raise RuntimeError('Invalid dump response')
    values=np.fromfile(path,dtype=np.float64)
    if values.size!=151936 or not np.isfinite(values).all(): raise RuntimeError('Invalid logits')
    return values


def compare_logits(a,b,atol=0):
    if not 0<=atol<=1e-6: raise ValueError('Tolerance must be explicit and <=1e-6')
    error=float(np.max(np.abs(a-b)))
    exact=bool(np.array_equal(a.view(np.uint64),b.view(np.uint64)))
    if (atol==0 and not exact) or error>atol:
        raise AssertionError(f'Logits differ: max error {error}, exact={exact}, allowed={atol}')
    return dict(max_absolute_error=error,bit_identical=exact,absolute_tolerance=atol)


def prefill(model,tokens,block_size):
    for start in range(0,len(tokens),block_size):
        chunk=tokens[start:start+block_size]
        if block_size==1:
            model.token(chunk[0],logits=False)
        else:
            model.block(chunk,logits=False)


def request(model,prompt,block_size,decode_count):
    model.reset()
    started=time.perf_counter()
    prefill(model,prompt[:-1],block_size)
    prefilled=time.perf_counter()
    token,position,_=model.token(prompt[-1])
    first=time.perf_counter()
    ids=[token]
    for _ in range(decode_count-1):
        token,position,_=model.token(token);ids.append(token)
    ended=time.perf_counter()
    if any(token in (151643,151645) for token in ids):
        raise RuntimeError('Early EOS rejects fixed-workload timing')
    return dict(prompt_tokens=len(prompt),prefill_tokens=len(prompt)-1,
                prefill_seconds=prefilled-started,
                prefill_tokens_per_second=(len(prompt)-1)/(prefilled-started),
                ttft_seconds=first-started,decode_seconds=ended-first,
                decode_tokens_per_second=(decode_count-1)/(ended-first),output_token_ids=ids,
                block_size=block_size,final_position=position)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--reference-binary',type=Path)
    parser.add_argument('--model')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--lanes',type=int,choices=[1,8],default=8)
    parser.add_argument('--reference-block-size',type=int,choices=[1,2,3,4],default=1)
    parser.add_argument('--block-size',type=int,choices=[2,3,4],default=4)
    parser.add_argument('--profiles',default='short,long')
    parser.add_argument('--pairs',type=int,default=3)
    parser.add_argument('--tokens',type=int,default=32)
    parser.add_argument('--gate-tokens',type=int,default=8)
    parser.add_argument('--logit-atol',type=float,default=0)
    parser.add_argument('--output',type=Path,default=Path('build/block-benchmark.json'))
    args=parser.parse_args()
    if not 1<=args.pairs<=20 or not 2<=args.tokens<=128 or not 1<=args.gate_tokens<=32:
        parser.error('Invalid workload')
    if not 0<=args.logit_atol<=1e-6:parser.error('Tolerance ceiling is1e-6; default is exact')
    profiles=args.profiles.split(',')
    if any(p not in ('short','medium','long') for p in profiles):parser.error('Invalid profile')
    reference=(args.reference_binary or args.binary).resolve()
    candidate=args.binary.resolve()
    models=[BlockProtocolModel(args.model,workers=args.workers,binary=p,lanes=args.lanes) for p in (reference,candidate)]
    protocol=ChatProtocol(models[0].model)
    report=dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                completed=False,gpu=False,pytorch=False,workers=args.workers,lanes=args.lanes,
                block_size=args.block_size,reference_block_size=args.reference_block_size,arithmetic='SystemVerilog real / FP64',
                batching='Declared layer-first RTL block execution; inspect engine implementation, not inferred from IPC counts.',
                limitation='Two resident processes, sequential inference, no OS cache eviction; timings include protocol overhead.',
                binaries=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (reference,candidate)],
                profiles=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        startup=[]
        for m in models:
            t=time.perf_counter();m.start();startup.append(time.perf_counter()-t)
        report['startup_seconds']=startup
        with tempfile.TemporaryDirectory() as directory:
            for profile in profiles:
                prompt=make_prompt(protocol,profile)
                for m in models:m.reset()
                prefill(models[0],prompt[:-1],args.reference_block_size);prefill(models[1],prompt[:-1],args.block_size)
                pending=prompt[-1];gates=[]
                for _ in range(args.gate_tokens):
                    outputs=[m.token(pending) for m in models]
                    if outputs[0][:2]!=outputs[1][:2]:raise AssertionError('Gate greedy/position mismatch')
                    arrays=[dump_logits(m,Path(directory)/f'{i}.bin') for i,m in enumerate(models)]
                    gates.append(compare_logits(*arrays,args.logit_atol));pending=outputs[0][0]
                record=dict(profile=profile,prompt_tokens=len(prompt),prompt_ids=prompt,gate=gates,pairs=[])
                report['profiles'].append(record)
                warm=[request(m,prompt,b,args.tokens) for m,b in zip(models,(args.reference_block_size,args.block_size))]
                expected=warm[0]['output_token_ids']
                if warm[1]['output_token_ids']!=expected:raise AssertionError('Warm greedy trajectories differ')
                record['warmup']=warm
                for index in range(args.pairs):
                    order=[0,1] if index%2==0 else [1,0]
                    results={i:request(models[i],prompt,args.reference_block_size if i==0 else args.block_size,args.tokens) for i in order}
                    if any(r['output_token_ids']!=expected for r in results.values()):raise AssertionError('Timed greedy trajectories differ')
                    pair=dict(index=index,order=order,serial=results[0],block=results[1],
                              prefill_speed_ratio=results[0]['prefill_seconds']/results[1]['prefill_seconds'],
                              ttft_speed_ratio=results[0]['ttft_seconds']/results[1]['ttft_seconds'])
                    record['pairs'].append(pair)
                    args.output.write_text(json.dumps(report,indent=2)+'\n')
                    print(json.dumps(dict(profile=profile,pair=index,prefill_ratio=pair['prefill_speed_ratio'],ttft_ratio=pair['ttft_speed_ratio'])),flush=True)
                record['median_prefill_speed_ratio']=statistics.median(p['prefill_speed_ratio'] for p in record['pairs'])
                record['median_ttft_speed_ratio']=statistics.median(p['ttft_speed_ratio'] for p in record['pairs'])
        report['completed']=True
    finally:
        for m in models:m.close()
        args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
