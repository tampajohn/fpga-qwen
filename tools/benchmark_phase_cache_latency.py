"""Same frozen phase model: warmed empty-KV versus canonical-prefix reuse."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import threading
import time
from rtl_chat import RTLChatEngine
from chat_protocol import ChatRequest,common_prefix
from benchmark_matched import make_prompt
from check_phase_pools_real import telemetry
from compare_phase_pools_deployed import read_dump_logits

FROZEN='162bed6577d91b19a57ea0f83d244d3fbb2666e329e90733f56f347267e8580d'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--pairs',type=int,default=3);p.add_argument('--tokens',type=int,default=32);a=p.parse_args()
 if not 1<=a.pairs<=5 or not 2<=a.tokens<=64:p.error('Invalid work budget')
 binary=a.binary.resolve()
 if sha(binary)!=FROZEN:raise RuntimeError('Not reviewed live phase artifact')
 engines=[RTLChatEngine(binary=binary,workers=8,lanes=8,block_size=4,prefill_size=8,reuse_prefix=False,canonical_prefix_reuse=True,precision_profile='fp32_b1_fp64_prefill')for _ in range(2)]
 report=dict(completed=False,binary_sha256=FROZEN,source_sha256=sha(__file__),tokens=a.tokens,pairs=a.pairs,selector='PREFILL_12',suffix=0,workloads=[],limitations=['Cold means empty logical KV in a warmed model, not cold process or OS cache.','Two resident models, each with persistent 12+8 pools; no core affinity or measured RSS claim.','Both modes use the same full-eight canonical policy; cold mode resets via a new session.','Appended workload uses exact prompt plus generated token IDs and one newline, not a rendered multi-turn chat template.','Decode includes processing the final prompt token; TTFT and post-first-token decoding are reported separately.'])
 a.output.parent.mkdir(parents=True,exist_ok=True)
 def save():a.output.write_text(json.dumps(report,indent=2)+'\n')
 def execute(e,ids,session,dump):
  before=telemetry(e.sim,True);start=time.perf_counter();result=list(e.stream(ChatRequest(session,ids,a.tokens,0,decoding='greedy',seed=42),threading.Event()))[-1];wall=time.perf_counter()-start;after=telemetry(e.sim,True)
  if result['type']!='done' or result['finish_reason']!='length' or len(result['token_ids'])!=a.tokens:raise RuntimeError('Fixed work ended early')
  bits=read_dump_logits(e.sim,dump).tobytes();m=result['metrics'];reused=m['reused_prefix_tokens']
  delta=[[after['dispatch_counts'][i][j]-before['dispatch_counts'][i][j]for j in range(3)]for i in range(2)]
  if delta[0][0]<=0 or delta[0][1]!=0 or delta[0][2]!=0 or delta[1][0]!=0 or delta[1][2]!=0 or (len(ids)-1-reused>0 and delta[1][1]<=0):raise RuntimeError('Unexpected phase routing')
  if len(e.cache_ids)!=len(ids)+a.tokens-1:raise RuntimeError('Consumed position mismatch')
  first=m['first_token_ms']/1000
  return dict(prompt_ids=ids,output_ids=result['token_ids'],metrics=m,wall_seconds=wall,ttft_seconds=first,post_first_token_seconds=max(0,m['elapsed_seconds']-first),recomputed_prefill_tokens=len(ids)-1-reused,scalar_final_prompt_tokens=1,reused_tokens=reused,final_position=len(e.cache_ids),logits_sha256=hashlib.sha256(bits).hexdigest(),phase_before=before,phase_after=after,phase_dispatch_delta=delta),bits
 try:
  for e in engines:e.sim.phase_pool_selector='PREFILL_12';e.sim.start();e.verify_precision(threading.Event());telemetry(e.sim,True)
  with tempfile.TemporaryDirectory()as tmp:
   for length in ('short','long'):
    prompt=make_prompt(engines[0],length)
    for pattern in ('repeat','append_generated'):
     record=dict(length=length,pattern=pattern,pairs=[]);report['workloads'].append(record)
     for pair in range(a.pairs):
      order=[0,1]if pair%2==0 else[1,0];prime={};results={};bits={}
      for i in order:
       e=engines[i];session=f'{length}-{pattern}-{pair}'
       prime[i],_=execute(e,prompt,session,Path(tmp)/f'prime{i}')
       old=list(e.cache_ids);safe=(len(prompt)-1)//8*8
       ids=prompt if pattern=='repeat' else prompt+prime[i]['output_ids']+[198]
       results[i],bits[i]=execute(e,ids,session if i else session+'-cold',Path(tmp)/f'final{i}')
       expected=min(safe,common_prefix(old,ids[:-1]),len(ids)-1)//8*8 if i else 0
       if results[i]['reused_tokens']!=expected:raise RuntimeError('Wrong reusable boundary')
       results[i]['prior_canonical_end']=safe
       results[i]['matching_prior_cache_tokens']=common_prefix(old,ids[:-1])
      for k in ('output_ids','logits_sha256'):
       if prime[0][k]!=prime[1][k]:raise RuntimeError('Priming differs')
      for k in ('prompt_ids','output_ids','final_position'):
       if results[0][k]!=results[1][k]:raise RuntimeError('Matched request differs')
      if bits[0]!=bits[1]:raise RuntimeError('Full logit bits differ')
      ratios={k:results[0][k]/results[1][k]for k in ('wall_seconds','ttft_seconds','post_first_token_seconds')}
      record['pairs'].append(dict(order=order,cold=results[0],reused=results[1],ratios=ratios,full_bits_exact=True));save()
      print(json.dumps(dict(length=length,pattern=pattern,pair=pair,reused=results[1]['reused_tokens'],recomputed=results[1]['recomputed_prefill_tokens'],ratios=ratios)),flush=True)
     record['median_ratios']={k:statistics.median(x['ratios'][k]for x in record['pairs'])for k in ratios}
  if sha(binary)!=FROZEN:raise RuntimeError('Artifact changed')
  report['completed']=True
 finally:
  for e in engines:e.close()
  save()
if __name__=='__main__':main()
