#!/usr/bin/env python3
"""Interleaved CPU-only SV A/B comparison with per-pair ratios.

Alternates A/B then B/A, keeping both processes resident but evaluating one at a
time. Warmup and each measured request reset logical KV. This measures current
host conditions, not a hardware latency estimate or a native-kernel comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

try:
    from .benchmark_matched import SVEngine, make_prompt
    from .chat_protocol import ChatProtocol
    from .experiment import ROOT, locate_model
except ImportError:
    from benchmark_matched import SVEngine, make_prompt
    from chat_protocol import ChatProtocol
    from experiment import ROOT, locate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--baseline-lanes',type=int,choices=[1,8],default=8)
    parser.add_argument('--candidate-lanes',type=int,choices=[1,8],default=8)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--profiles',default='short,long')
    parser.add_argument('--pairs',type=int,default=3)
    parser.add_argument('--tokens',type=int,default=32)
    parser.add_argument('--model')
    parser.add_argument('--host-load-note',default='')
    parser.add_argument('--output',type=Path,default=ROOT/'build/compare-variants.json')
    args=parser.parse_args()
    profiles=args.profiles.split(',')
    if any(profile not in ('short','medium','long') for profile in profiles):
        parser.error('Unknown context profile')
    if not 1<=args.pairs<=20 or not 2<=args.tokens<=512 or not 1<=args.workers<=32:
        parser.error('Invalid pairs, tokens or worker count')
    binaries={name:path.expanduser().resolve() for name,path in [('baseline',args.baseline),('candidate',args.candidate)]}
    for path in binaries.values():
        if not path.is_file():
            parser.error(f'Missing binary: {path}')
    checkpoint=locate_model(args.model)
    protocol=ChatProtocol(checkpoint)
    engines={name:SVEngine(checkpoint,args.workers,protocol.tokenizer,path,lanes)
             for (name,path),lanes in zip(binaries.items(),(args.baseline_lanes,args.candidate_lanes))}
    report=dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),gpu=False,pytorch=False,
                workers=args.workers,tokens=args.tokens,host_load_note=args.host_load_note,completed=False,
                limitation='Two resident models; sequential inference, alternating paired order; no OS cache eviction or host isolation.',
                settings={name:dict(binary=str(binaries[name]),sha256=hashlib.sha256(binaries[name].read_bytes()).hexdigest(),
                          **engine.metadata) for name,engine in engines.items()},profiles=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        report['startup_seconds']={name:engine.start() for name,engine in engines.items()}
        for profile in profiles:
            prompt=make_prompt(protocol,profile)
            warmup={name:engine.generate(prompt,args.tokens,True) for name,engine in engines.items()}
            reference=warmup['baseline']['output_token_ids']
            if warmup['candidate']['output_token_ids']!=reference:
                raise RuntimeError('Warmup greedy trajectories differ; speed comparison rejected')
            record=dict(profile=profile,prompt_tokens=len(prompt),
                        prompt_ids_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),warmup=warmup,pairs=[])
            report['profiles'].append(record)
            for index in range(args.pairs):
                order=['baseline','candidate'] if index%2==0 else ['candidate','baseline']
                outputs={name:engines[name].generate(prompt,args.tokens,True) for name in order}
                if any(result['output_token_ids']!=reference for result in outputs.values()):
                    raise RuntimeError('Measured greedy trajectories differ; speed comparison rejected')
                a,b=outputs['baseline'],outputs['candidate']
                pair=dict(index=index,order=order,results=outputs,unchanged_greedy_sequence=True,
                          decode_speed_ratio=b['steady_decode_tokens_per_second']/a['steady_decode_tokens_per_second'],
                          ttft_speed_ratio=a['first_token_seconds']/b['first_token_seconds'])
                record['pairs'].append(pair)
                args.output.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(dict(profile=profile,pair=index,order=order,
                    baseline_tps=a['steady_decode_tokens_per_second'],candidate_tps=b['steady_decode_tokens_per_second'],
                    decode_ratio=pair['decode_speed_ratio'],ttft_ratio=pair['ttft_speed_ratio'])),flush=True)
            for metric in ('decode_speed_ratio','ttft_speed_ratio'):
                values=[pair[metric] for pair in record['pairs']]
                record[metric+'_summary']=dict(median=statistics.median(values),minimum=min(values),maximum=max(values))
            for name in engines:
                values=[pair['results'][name]['steady_decode_tokens_per_second'] for pair in record['pairs']]
                record[name+'_decode_summary']=dict(median=statistics.median(values),minimum=min(values),maximum=max(values))
            args.output.write_text(json.dumps(report,indent=2)+'\n')
        report['completed']=True
    finally:
        for engine in engines.values():
            engine.close()
        args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
