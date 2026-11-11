"""Paired exact-route canonical FP64-prefix reuse versus fresh hybrid requests.

Both modes use the same binary, seed and prompt IDs. Priming requests are untimed
and start in a new session each pair. Only the follow-up is timed; binary logit
dumps and comparison are outside that interval. No native-backend comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import threading
import time
import numpy as np
try:
    from .chat_protocol import common_prefix
    from .rtl_chat import RTLChatEngine
    from .evaluate_projection_quality import read_dump_logits
except ImportError:
    from chat_protocol import common_prefix
    from rtl_chat import RTLChatEngine
    from evaluate_projection_quality import read_dump_logits

FILLER=('Memory bandwidth measures data transfer rate between a processor and memory. '
        'Caches retain recently accessed information and can reduce transfers. ')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pairs',type=int,default=3)
    parser.add_argument('--tokens',type=int,default=8)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--decodings',default='greedy')
    args=parser.parse_args()
    decodings=args.decodings.split(',')
    if not 1<=args.pairs<=10 or not 1<=args.tokens<=128 or not 1<=args.workers<=32 or any(d not in ('greedy','chat') for d in decodings):
        parser.error('Invalid workload')
    binary=args.binary.resolve()
    settings=dict(binary=binary,workers=args.workers,lanes=8,block_size=1,prefill_size=8,
                  reuse_prefix=False,precision_profile='fp32_b1_fp64_prefill')
    engines={mode:RTLChatEngine(**settings,canonical_prefix_reuse=mode=='on') for mode in ('off','on')}
    report=dict(completed=False,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        binary=str(binary),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pairs=args.pairs,output_budget=args.tokens,workers=args.workers,seed=42,workloads=[],
        policy='fp64_b8_aligned_v1',gpu=False,
        limitations=['Same hybrid executable only; no native parity claim.',
            'Two resident processes; idle workers sleep. Startup and priming excluded.',
            'Same seeded IDs and full final logits required within each pair. EOS respected.',
            'Wall timings include streaming/transport and cache bookkeeping, exclude tokenization and logit dumps.',
            'Exclusive agent timing window required; unrelated user applications may remain active.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    def execute(engine,messages,session,decoding,dump):
        request=engine.prepare(dict(session=session,messages=messages,max_new_tokens=args.tokens,
                                   max_draft=0,decoding=decoding,seed=42))
        start=time.perf_counter()
        result=list(engine.stream(request,threading.Event()))[-1]
        wall=time.perf_counter()-start
        if result['type']!='done' or result['finish_reason'] not in ('length','eos'):
            raise RuntimeError('Request did not complete')
        logits=read_dump_logits(engine.sim,dump)
        return dict(prompt_ids=request.ids,token_ids=result['token_ids'],text=result['text'],
                    finish_reason=result['finish_reason'],metrics=result['metrics'],wall_seconds=wall,
                    logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest()),logits
    try:
        with tempfile.TemporaryDirectory() as directory:
            for profile,repetitions in [('short',0),('long',16)]:
                messages=[dict(role='user',content=FILLER*repetitions+'Explain memory bandwidth and caching in four concise sentences.')]
                for pattern in ('repeat','appended_turn'):
                    for decoding in decodings:
                        record=dict(profile=profile,pattern=pattern,decoding=decoding,pairs=[])
                        report['workloads'].append(record)
                        for pair_index in range(args.pairs):
                            order=['off','on'] if pair_index%2==0 else ['on','off']
                            session=f'{profile}-{pattern}-{decoding}-{pair_index}'
                            primed={};results={};arrays={}
                            for mode in order:
                                engine=engines[mode]
                                primed[mode],_=execute(engine,messages,session,decoding,Path(directory)/f'prime-{mode}.bin')
                                primed[mode]['cache_ids']=list(engine.cache_ids)
                                if primed[mode]['metrics']['reused_prefix_tokens']:
                                    raise RuntimeError('Priming request unexpectedly reused prior session')
                                followup=messages if pattern=='repeat' else messages+[
                                    dict(role='assistant',content=primed[mode]['text']),
                                    dict(role='user',content='How does that apply when generating one token at a time? Answer briefly.')]
                                results[mode],arrays[mode]=execute(engine,followup,session,decoding,Path(directory)/f'{mode}.bin')
                            for key in ('prompt_ids','token_ids','text','finish_reason','logits_sha256'):
                                if primed['off'][key]!=primed['on'][key]:raise RuntimeError(f'Priming mismatch: {key}')
                            for key in ('prompt_ids','token_ids','text','finish_reason'):
                                if results['off'][key]!=results['on'][key]:raise RuntimeError(f'Follow-up mismatch: {key}')
                            if not np.array_equal(arrays['off'].view(np.uint64),arrays['on'].view(np.uint64)):
                                raise RuntimeError('Full logit bits differ')
                            reused=results['on']['metrics']['reused_prefix_tokens']
                            safe_end=(len(primed['on']['prompt_ids'])-1)//8*8
                            expected=min(safe_end,common_prefix(primed['on']['cache_ids'],results['on']['prompt_ids'][:-1]))//8*8
                            if results['off']['metrics']['reused_prefix_tokens'] or reused<=0 or reused!=expected:
                                raise RuntimeError('Unsafe or absent canonical reuse')
                            a,b=results['off'],results['on']
                            ratios=dict(total=a['wall_seconds']/b['wall_seconds'],
                                ttft=a['metrics']['first_token_ms']/b['metrics']['first_token_ms'],
                                prefill=a['metrics']['prefill_seconds']/b['metrics']['prefill_seconds'])
                            record['pairs'].append(dict(order=order,results=results,ratios=ratios,
                                priming_ids=primed['off']['token_ids'],exact_output_ids=True,full_logits_bit_identical=True))
                            save()
                            print(json.dumps(dict(profile=profile,pattern=pattern,decoding=decoding,pair=pair_index,
                                                  reused=reused,ratios=ratios)),flush=True)
                        record['median_ratios']={key:statistics.median(p['ratios'][key] for p in record['pairs'])
                                                 for key in ('total','ttft','prefill')}
        report['completed']=True
    finally:
        for engine in engines.values():engine.close()
        save()


if __name__=='__main__':main()
