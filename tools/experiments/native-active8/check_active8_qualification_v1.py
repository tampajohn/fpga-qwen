#!/usr/bin/env python3
"""Read-only fail-closed qualification binding; no implicit release authorization."""
from pathlib import Path
import json,hashlib
B=Path(__file__).resolve().parent;TAG='decode-attention-active8-v2'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def validate(full=True):
 candidate=B/(TAG+'-build/qwen_cpu');manifest=read(candidate.parent/'manifest.json');assert manifest['completed']and sha(candidate)==manifest['binary_sha256'];binary_sha=manifest['binary_sha256'];reports={}
 def load(role,path):
  j=read(path);assert j['completed']and j['passed']and not j.get('error')and not j.get('cleanup_error');reports[role]={'path':str(path),'sha256':sha(path)};return j
 for suffix,count in [('exact-gate',56),('timeboundary-exact-gate',12),('newboundary-relative-exact-gate',8),('step-edge-exact-gate',24)]:
  j=load(suffix,B/f'{TAG}-{suffix}.json');assert j['compared_arrays']==count and j['inputs'][str(candidate)]==binary_sha
 j=load('api',B/f'{TAG}-verify-api/report.json');assert j['full_acceptance_matrix_passed']and j['workers']==[1,12,64]and j['prefixes']==[0,1,511,512,513,3966]and j['commits']==[1,2]
 for name in ('core.cpp','core.hpp'):
  path=B/f'{TAG}-source'/name;assert j['source_hashes'][str(path)]==sha(path)==manifest['source_sha256'][str(path)]
 if full:
  paths={'prefix':f'{TAG}-prefix-check/report.json','suffix':f'{TAG}-suffix-trajectories.json','cli':f'{TAG}-suffix-cli/report.json','quality':f'{TAG}-quality.json','extended':f'{TAG}-extended-quality.json','seeded':f'{TAG}-seeded-stream/report.json'}
  for role,name in paths.items():
   j=load(role,B/name);bindings=j.get('inputs',j.get('hashes',{}));assert bindings[str(candidate)]==binary_sha
   if role=='prefix':assert len(j['cases'])==28
   if role=='suffix':assert len(j['cases'])==35 and len(j['extended_cases'])==7 and j['original35_passed']and j['cancellation']['passed']and j['restart']['passed']
   if role=='cli':assert len(j['cases'])==5 and j['invalid_and_chat_commands_reset']
   if role in ('quality','extended'):assert j['exact_trajectory_gate_passed']and len(j['suite']if role=='quality'else j['cases'])==(9 if role=='quality'else 18)
   if role=='seeded':assert j['seeds']==[42,1337]and j['repeats']==2 and j['requested_tokens']==20
   if role in ('suffix','quality','extended','seeded'):assert bindings[str(B/'build-20/qwen_cpu')]==sha(B/'build-20/qwen_cpu')
 return {'completed':True,'passed':True,'candidate':str(candidate),'binary_sha256':binary_sha,'candidate_manifest_sha256':sha(candidate.parent/'manifest.json'),'baseline_binary_sha256':sha(B/'build-20/qwen_cpu'),'baseline_manifest_sha256':sha(B/'build-20/manifest.json'),'full':full,'reports':reports,'absolute_oracle_pass_claimed':False,'release_authorized':False}
if __name__=='__main__':print(json.dumps(validate(),indent=2))
