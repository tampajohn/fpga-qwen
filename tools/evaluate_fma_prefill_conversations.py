"""Compare fresh-session fused-prefill conversations and exact same-route replay."""
import argparse
import hashlib
import json
from pathlib import Path
import threading

from tools.rtl_chat import RTLChatEngine
from tools.conversation_eval import CASES,evaluate_response,suite_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--reference-profile',choices=('fp32_b1_fp64_prefill','fp32_b1_fp32_prefill'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-prefill-size',type=int,choices=(4,8),default=4)
    parser.add_argument('--candidate-block-size',type=int,choices=(1,4),default=4)
    parser.add_argument('--disable-prefix-reuse',action='store_true',help='Reset each request; required for experimental hybrid precision')
    args=parser.parse_args()
    if args.reference_prefill_size != 8 or args.candidate_block_size != 4 or not args.disable_prefix_reuse:
        parser.error('This profile comparison requires B8 prefill, B4 verification and disabled prefix reuse')
    paths=[args.reference.resolve(),args.candidate.resolve()]
    engines=[RTLChatEngine(binary=paths[0],workers=8,lanes=8,block_size=4,prefill_size=8,
                 reuse_prefix=False,precision_profile=args.reference_profile,canonical_prefix_reuse=args.reference_profile=='fp32_b1_fp64_prefill'),
             RTLChatEngine(binary=paths[1],workers=8,lanes=8,block_size=4,prefill_size=8,
                 reuse_prefix=False,precision_profile='fp32_b1_fp32_fma_prefill')]
    report=dict(completed=False,suite_sha256=suite_fingerprint(),workers=8,decoding='greedy',
                suffix_draft=0,verification_capacities=[4,args.candidate_block_size],
                prefix_reuse_enabled=False,prefill_capacities=[8,8], precision_profiles=[e.precision_profile for e in engines],
                penalty_policy='Existing RTLChatEngine greedy path; no sampler or penalty override',
                timing_claim=False,kv_storage=['fp64','fp64'],attention_arithmetic='fp64',limitation='Nine strict regression cases, not a general quality benchmark; concurrent correctness jobs permitted.',
                binaries=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        for case in CASES:
            results=[];prompts=[]
            for engine in engines:
                engine.close()  # Every case starts empty, including reference canonical policy.
                request=engine.prepare(dict(session='prefill-quality-'+case.name,messages=list(case.messages),
                    max_new_tokens=case.max_new_tokens,max_draft=0,decoding='greedy',workers=8))
                prompts.append(request.ids)
                events=list(engine.stream(request,threading.Event()))
                if not events or events[-1]['type']!='done':raise RuntimeError('Conversation did not complete')
                final=events[-1]
                results.append(dict(text=final['text'],token_ids=final['token_ids'],grade=evaluate_response(case,final['text']),
                    prefill_size=engine.sim.prefill_size,reused_prefix_tokens=final['metrics']['reused_prefix_tokens']))
            if prompts[0]!=prompts[1]:raise AssertionError('Input token IDs differ')
            if any(r['reused_prefix_tokens'] for r in results):raise AssertionError('Fresh session unexpectedly reused KV')
            record=dict(case=case.name,messages=list(case.messages),max_new_tokens=case.max_new_tokens,
                prompt_token_ids=prompts[0],baseline=results[0],candidate=results[1],
                identical_output_ids=results[0]['token_ids']==results[1]['token_ids'],
                identical_text=results[0]['text']==results[1]['text'])
            # Repeating the same candidate request must reset and reproduce its route.
            candidate=engines[1]
            repeated=list(candidate.stream(request,threading.Event()))[-1]
            if repeated['token_ids'] != results[1]['token_ids'] or repeated['metrics']['reused_prefix_tokens']:
                raise AssertionError('Same-route replay changed output or reused KV')
            record['same_route_replay_exact_ids']=True
            report['cases'].append(record)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(dict(case=case.name,identical_ids=record['identical_output_ids'],
                passes=[r['grade']['passed'] for r in results])),flush=True)
        report['passed_counts']=[sum(c[k]['grade']['passed'] for c in report['cases']) for k in ('baseline','candidate')]
        report['all_output_ids_identical']=all(c['identical_output_ids'] for c in report['cases'])
        report['baseline_pass_regressions']=[c['case'] for c in report['cases']
            if c['baseline']['grade']['passed'] and not c['candidate']['grade']['passed']]
        report['baseline_passes_preserved']=not report['baseline_pass_regressions']
        report['completed']=True
    finally:
        for engine in engines:engine.sim.close()
        args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
