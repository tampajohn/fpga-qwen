#!/usr/bin/env python3
"""Exploratory SV worker sweep and optional macOS sampling, outside scored benchmarks.

Uses the matched benchmark prompt and EOS policy. Sampling overhead invalidates the
sampled request's timing; unsampled requests are recorded separately. No GPU/Torch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
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
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--workers', default='4,8,12,16')
    parser.add_argument('--lanes', type=int, choices=[1,8], default=8)
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--profile', choices=['short','medium','long'], default='short')
    parser.add_argument('--sample-seconds', type=int, default=0)
    parser.add_argument('--model')
    parser.add_argument('--host-load-note', default='', help='Record competing host work; do not terminate user processes')
    parser.add_argument('--output', type=Path, default=ROOT/'build/profile-simulator.json')
    args = parser.parse_args()
    workers = [int(value) for value in args.workers.split(',')]
    if not workers or any(not 1<=value<=32 for value in workers):
        parser.error('workers must be between 1 and 32')
    if not 1<=args.repetitions<=20 or not 2<=args.tokens<=512 or not 0<=args.sample_seconds<=60:
        parser.error('invalid repetitions, tokens or sample duration')
    binary = args.binary.resolve()
    checkpoint = locate_model(args.model)
    protocol = ChatProtocol(checkpoint)
    prompt = make_prompt(protocol, args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  prompt_tokens=len(prompt), profile=args.profile, lanes=args.lanes,
                  prompt_ids_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                  exploratory=True, gpu=False, pytorch=False, runs=[],
                  timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  host_load_note=args.host_load_note,
                  limitation='Sequential worker sweep; order and host-load effects are not controlled. Sampling timings excluded.')
    reference = None
    for count in workers:
        engine = SVEngine(checkpoint,count,protocol.tokenizer,binary,args.lanes)
        try:
            startup = engine.start()
            first = engine.generate(prompt,args.tokens,True)
            for repeat in range(args.repetitions):
                result = engine.generate(prompt,args.tokens,True)
                if reference is None:
                    reference = result['output_token_ids']
                result.update(workers=count, repetition=repeat,
                              matches_reference_greedy=result['output_token_ids']==reference)
                report['runs'].append(result)
                print(json.dumps(dict(workers=count, repetition=repeat,
                      ttft=result['first_token_seconds'], tps=result['steady_decode_tokens_per_second'],
                      matches_reference_greedy=result['matches_reference_greedy'])),flush=True)
            if args.sample_seconds and count==workers[0]:
                if platform.system()!='Darwin':
                    raise RuntimeError('macOS sample requested on another platform')
                sample_path = args.output.with_suffix('.sample.txt')
                sampler = subprocess.Popen(['sample',str(engine.model.process.pid),str(args.sample_seconds),'1','-file',str(sample_path)],
                                           stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
                try:
                    while sampler.poll() is None:
                        engine.generate(prompt,args.tokens,True)
                    _, stderr = sampler.communicate()
                    if sampler.returncode:
                        raise RuntimeError(stderr)
                finally:
                    if sampler.poll() is None:
                        sampler.terminate()
                        sampler.wait()
                report['sample_file']=str(sample_path)
                report['sample_workers']=count
                report['sample_warning']='Sampling request timings excluded; stack counts include waiting threads.'
            report.setdefault('first_requests',[]).append(dict(workers=count,startup_seconds=startup,result=first))
        finally:
            engine.close()
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    if not all(run['matches_reference_greedy'] for run in report['runs']):
        raise SystemExit('Worker count changed greedy output')


if __name__=='__main__':
    main()
