#!/usr/bin/env python3
"""Explicit final kernel comparison phase,12/6 only; never select/deploy automatically."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'build/native-cpu'
p=argparse.ArgumentParser();p.add_argument('--candidate-tag',choices=['prefill-query-row-pair','queryrow-b1-headpair-v1'],required=True);p.add_argument('--phase',choices=['vs19','vsllama'],required=True);p.add_argument('--validate-only',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
tag=a.candidate_tag;binary=B/f'{tag}-build/qwen_cpu';checker=B/('check-queryrow-relative-prerequisites.py' if tag=='prefill-query-row-pair' else 'check-headpair-relative-prerequisites.py')
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def guarded(name,args):subprocess.run([sys.executable,str(ROOT/'tools/benchmark_guard.py'),'--name',name,'--',sys.executable,*map(str,args)],cwd=ROOT,check=True)
if a.validate_only:
 sys.path.insert(0,str(ROOT/'tools'))
 from benchmark_native_phase0 import guard
 guard(ROOT/'build/benchmark-status')
 subprocess.run([sys.executable,str(checker),'final'],cwd=ROOT,check=True)
 digest=sha(binary)
 for suffix,count in [('newboundary-relative-exact-gate.json',8),('seeded-stream/report.json',None)]:
  report=read(B/f'{tag}-{suffix}');assert report['completed'] and report['passed'] and not report.get('error') and not report.get('cleanup_error')
  assert report['inputs'][str(binary)]==digest
  if count is not None:assert report['compared_arrays']==count
  else:assert report['seeds']==[42,1337] and report['repeats']==2 and report['requested_tokens']==20
  for path,value in report['inputs'].items():assert sha(path)==value
 for suite in ('standard','extended'):
  quality=read(B/f'{tag}-{"quality" if suite=="standard" else "extended-quality"}.json')
  assert quality['completed'] and quality['passed'] and quality['exact_trajectory_gate_passed']
  assert quality['inputs'][str(binary)]==digest
  assert len(quality['suite'] if suite=='standard' else quality['cases'])==(9 if suite=='standard' else 18)
 print('Explicit selected kernel relative/seed/newboundary prerequisites passed. Absolute knownfailures not waived.')
 sys.exit(0)
plan=B/f'{tag}-final-{a.phase}-plan-v1.json';assert not plan.exists()
for suite in ('standard','extended'):
 for suffix in ('confirmation.json','confirmation-logs','analysis.json'):assert not (B/f'{tag}-final-{a.phase}-{suite}-{suffix}').exists()
record=dict(completed=False,candidate_tag=tag,phase=a.phase,decode_threads=12,prefill_threads=6,runtime_defaults_inferred=False,comparison_scope='Same12/6 kernel comparison; not a serving8/6 worker result.',known_absolute_failures_retained=['boundary63','boundary64'],source_sha256={str(Path(__file__).resolve()):sha(Path(__file__).resolve()),str(checker):sha(checker)},suites=[])
def save():plan.write_text(json.dumps(record,indent=2)+'\n')
save()
try:
 for suite,tokens in [('standard',32),('extended',128)]:
  assert all(sha(path)==digest for path,digest in record['source_sha256'].items())
  guarded(f'Final {tag} {a.phase} prerequisite review',[Path(__file__).resolve(),'--candidate-tag',tag,'--phase',a.phase,'--validate-only'])
  rounds=20 if a.phase=='vs19' and suite=='standard' else 10
  output=B/f'{tag}-final-{a.phase}-{suite}-confirmation.json'
  command=[ROOT/'tools/benchmark_native_cpu.py','--binary',binary,'--weights',B/'weights.bin','--threads',12,'--prefill-threads',6,'--mode','confirmation','--rounds',rounds,'--warmups',1,'--context-suite',suite,'--tokens',tokens,'--output',output]
  if a.phase=='vs19':command+=['--reference-binary',B/'build-19/qwen_cpu'];baseline='native-reference-build-19'
  else:command+=['--baseline-build',ROOT/'build/native-phase0/kleidi-on-lto-on','--model',ROOT/'build/bench/qwen3-0.6b-f16.gguf'];baseline='llama-f16-onon-d12-p6'
  guarded(f'Final {tag} {a.phase} {suite} {rounds}pairs12/6',command)
  report=read(output);assert report['completed'] and not report.get('error') and not report.get('cleanup_error') and report['paired_sampling_complete'] and report['preflight_same_precision_ids_passed']
  assert report['rounds']==rounds and len(report['visits'])==2*rounds and report['tokens']==tokens and report['warmups_per_visit_profile']==1
  names=[c['name'] for c in report['configurations']];assert names[0]==baseline and len(names)==2
  for index in range(rounds):
   visits=report['visits'][2*index:2*index+2];assert [v['config'] for v in visits]==(names if index%2==0 else names[::-1])
   assert all(v['phase']=='scored' and v['round']==index and v['order_index']==order for order,v in enumerate(visits))
  analysis=B/f'{tag}-final-{a.phase}-{suite}-analysis.json'
  subprocess.run([sys.executable,str(ROOT/'tools/analyze_native_phase0.py'),'--report',str(output),'--baseline',baseline,'--quality-gate',str(B/f'{tag}-{"quality" if suite=="standard" else "extended-quality"}.json'),'--output',str(analysis)],cwd=ROOT,check=True)
  result=read(analysis);assert result['completed'] and result['rounds']==rounds and result['source_report_sha256']==sha(output)
  record['suites'].append(dict(suite=suite,rounds=rounds,tokens=tokens,report=str(output),report_sha256=sha(output),analysis=str(analysis),analysis_sha256=sha(analysis)));save()
 assert all(sha(path)==digest for path,digest in record['source_sha256'].items()),'Controller changed after final suite'
 record['completed']=True
except BaseException as error:record['error']=dict(type=type(error).__name__,message=str(error));raise
finally:save()
print(plan)
