"""Compare free-running objective conversations with explicit prefill capacity."""
import argparse
import hashlib
import json
from pathlib import Path
import threading

from rtl_chat import RTLChatEngine
from conversation_eval import CASES,evaluate_response,suite_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    paths=[args.reference.resolve(),args.candidate.resolve()]
    engines=[RTLChatEngine(binary=p,workers=8,lanes=8,block_size=4,prefill_size=s) for p,s in zip(paths,(4,8))]
    report=dict(completed=False,suite_sha256=suite_fingerprint(),workers=8,decoding='greedy',
                suffix_draft=0,verification_capacity=4,prefill_capacities=[4,8],
                penalty_policy='Existing RTLChatEngine greedy path; no sampler or penalty override',
                timing_claim=False,limitation='Nine strict regression cases, not a general quality benchmark; concurrent correctness jobs permitted.',
                binaries=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        for case in CASES:
            results=[];prompts=[]
            for engine in engines:
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
            report['cases'].append(record)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(dict(case=case.name,identical_ids=record['identical_output_ids'],
                passes=[r['grade']['passed'] for r in results])),flush=True)
        report['passed_counts']=[sum(c[k]['grade']['passed'] for c in report['cases']) for k in ('baseline','candidate')]
        report['all_output_ids_identical']=all(c['identical_output_ids'] for c in report['cases'])
        report['completed']=True
    finally:
        for engine in engines:engine.sim.close()
        args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
