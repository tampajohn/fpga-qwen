#!/usr/bin/env python3
"""Paired suffix on/off measurements with the same B4 prefill and target sampler.

One resident simulator, empty logical KV per request, alternating order. Records
complete request, prefill, first-token, and decode wall times separately. Decode
includes the first target round, avoiding ambiguous arrival rates for multi-token
updates. Rejects any output-ID/finish mismatch. No GPU or host model arithmetic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time
try:
    from .rtl_chat import RTLChatEngine
    from .experiment import ROOT
except ImportError:
    from rtl_chat import RTLChatEngine
    from experiment import ROOT

WORKLOADS={
    'repeated_code':'Repeat this code exactly three times, without an introduction:\ndef add(a, b):\n    return a + b',
    'ordinary_prose':'Explain why leaves change color in autumn in two concise sentences.',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--pairs',type=int,default=2)
    parser.add_argument('--tokens',type=int,default=64)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--decoding',choices=['chat','greedy'],default='chat')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--output',type=Path,default=ROOT/'build/suffix-paired.json')
    args=parser.parse_args()
    if not 1<=args.pairs<=10 or not 1<=args.tokens<=512:
        parser.error('Invalid pair/token count')
    binary=args.binary.expanduser().resolve()
    engine=RTLChatEngine(binary=binary,lanes=8,workers=args.workers,block_size=4)
    report=dict(binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),workers=args.workers,
                seed=args.seed,decoding=args.decoding,max_new_tokens=args.tokens,prefill_block_size=4,
                limitation='Single resident process, exclusive agent timing window, user applications remain running. Decode includes first target round; startup excluded after warmup.',
                completed=False,workloads=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    serial=0
    def request(name,draft):
        nonlocal serial
        serial+=1
        prepared=engine.prepare(dict(session=f'suffix-benchmark-{serial}',messages=[dict(role='user',content=WORKLOADS[name])],
                                     max_new_tokens=args.tokens,max_draft=draft,decoding=args.decoding,seed=args.seed))
        start=time.perf_counter()
        final=list(engine.stream(prepared,threading.Event()))[-1]
        final['request_wall_seconds']=time.perf_counter()-start
        final['prompt_tokens']=len(prepared.ids)
        assert final['metrics']['reused_prefix_tokens']==0
        assert final['finish_reason'] in ('eos','length')
        return final
    try:
        for name,prompt in WORKLOADS.items():
            warmup=request(name,0)
            record=dict(name=name,prompt=prompt,warmup=warmup,pairs=[])
            report['workloads'].append(record)
            for index in range(args.pairs):
                order=['off','on'] if index%2==0 else ['on','off']
                results={mode:request(name,3 if mode=='on' else 0) for mode in order}
                a,b=results['off'],results['on']
                if (a['token_ids'],a['finish_reason'])!=(b['token_ids'],b['finish_reason']):
                    raise RuntimeError('Suffix mode changed target outputs; comparison rejected')
                pair=dict(order=order,results=results,exact_output_ids=True,
                          total_speed_ratio=a['request_wall_seconds']/b['request_wall_seconds'],
                          decode_speed_ratio=a['metrics']['decode_seconds']/b['metrics']['decode_seconds'],
                          ttft_speed_ratio=a['metrics']['first_token_ms']/b['metrics']['first_token_ms'])
                record['pairs'].append(pair)
                args.output.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(dict(workload=name,pair=index,total_ratio=pair['total_speed_ratio'],
                    decode_ratio=pair['decode_speed_ratio'],ttft_ratio=pair['ttft_speed_ratio'],
                    accepted=b['metrics']['accepted_tokens'],proposed=b['metrics']['proposed_tokens'],
                    output_tokens=b['metrics']['output_tokens'])),flush=True)
            for metric in ('total_speed_ratio','decode_speed_ratio','ttft_speed_ratio'):
                values=[pair[metric] for pair in record['pairs']]
                record[metric]=dict(median=statistics.median(values),minimum=min(values),maximum=max(values))
        report['completed']=True
    finally:
        engine.close()
        args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
