#!/usr/bin/env python3
"""Root-run archive only after timing ends. No model, build, selection or git actions."""
import argparse,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'build/native-cpu'
def require(ok,message):
 if not ok:raise RuntimeError(message)
def sha(path):
 with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def clean(value):
 if isinstance(value,dict):
  result={}
  for key,item in value.items():
   if key in ('pid','ppid','target_pid','hostname','username','ps_command','old_session','new_session','benchmark_session')or key.endswith('_pid'):continue
   if key=='busiest_processes':
    rows=[]
    for line in item:
     fields=line.split()
     if len(fields)>=5 and fields[0].isdigit():
      cpu,memory=float(fields[2]),float(fields[3]);require(math.isfinite(cpu)and math.isfinite(memory),'Invalid host percentages')
      rows.append(dict(cpu_percent=cpu,memory_percent=memory,elapsed=fields[4]))
    result[key]=rows
   else:result[clean(key)]=clean(item)
  return result
 if isinstance(value,list):return [clean(v)for v in value]
 if isinstance(value,str):return value.replace(str(ROOT),'$REPO').replace(str(Path.home()),'$HOME')
 return value

def validate_pair(confirmation,analysis,phase,suite,tag):
 c,a=read(confirmation),read(analysis)
 for report in (c,a):require(report.get('completed')is True and not report.get('error')and not report.get('cleanup_error'),'Incomplete/error report')
 rounds=20 if phase=='vs19'and suite=='standard'else 10;tokens=32 if suite=='standard'else 128
 require(c['mode']=='confirmation'and c['context_suite']==suite and c['rounds']==rounds and c['tokens']==tokens,'Unexpected fixed workload')
 require(c['paired_sampling_complete']is True and c['preflight_same_precision_ids_passed']is True and c['warmups_per_visit_profile']==1,'Incomplete preflight/sampling')
 require(c['gpu']is False and c['pytorch']is False and c['precision']=='f16','Unexpected runtime')
 configs=c['configurations'];require(len(configs)==2,'Expected two configurations')
 binary=str((BASE/f'{tag}-build/qwen_cpu').resolve());digest=c['inputs'][binary]
 require(c['native_builds'][binary]['binary_sha256']==digest,'Candidate manifest/input digest mismatch')
 # Pin archived provenance without rereading large executables or model files.
 require(Path(configs[1]['binary']).resolve()==Path(binary),'Wrong candidate path')
 if phase=='worker':
  require(c['preflight_full_logits_bit_identical']is True,'Worker full-logit preflight missing')
  require([x['decode_threads']for x in configs]==[12,8]and all(x['prefill_threads']==6 and Path(x['binary']).resolve()==Path(binary)for x in configs),'Worker config mismatch')
  baseline='native-workers-d12-p6'
 else:
  require(all(x['decode_threads']==12 and x['prefill_threads']==6 for x in configs),'Final comparison must use matched12/6')
  if phase=='vs19':
   require(Path(configs[0]['binary']).resolve()==(BASE/'build-19/qwen_cpu').resolve(),'Wrong native reference');baseline='native-reference-build-19'
  else:
   require(Path(configs[0]['build_dir']).resolve()==(ROOT/'build/native-phase0/kleidi-on-lto-on').resolve(),'Wrong llama build');baseline='llama-f16-onon-d12-p6'
 names=[x['name']for x in configs];require(names[0]==baseline and len(set(names))==2,'Unexpected config names')
 profiles={'short':85,'long':452}if suite=='standard'else{'context1024':1024,'context2048':2048,'context3968':3968}
 require(len(c['visits'])==2*rounds,'Missing scored visits')
 for index in range(rounds):
  visits=c['visits'][2*index:2*index+2];require([v['config']for v in visits]==(names if index%2==0 else names[::-1]),'Order mismatch')
  for order,v in enumerate(visits):require(v['phase']=='scored'and v['round']==index and v['order_index']==order and set(v['profiles'])==set(profiles),'Visit scope mismatch')
 require(a['source_report_sha256']==sha(confirmation)and a['rounds']==rounds and a['tokens']==tokens and a['context_suite']==suite and a['baseline']==baseline and a['configurations']==configs,'Analysis binding mismatch')
 require(a['configuration_order_by_round']==[names if i%2==0 else names[::-1]for i in range(rounds)],'Analysis order mismatch')
 comparisons=a['comparisons'];require(len(comparisons)==len(profiles)and {r['profile']for r in comparisons}==set(profiles),'Analysis profiles missing')
 for row in comparisons:
  require(row['candidate']==names[1]and row['prompt_tokens']==profiles[row['profile']],'Wrong analysis candidate/profile')
  require(len(row['raw_pairs'])==rounds and sorted(p['round']for p in row['raw_pairs'])==list(range(rounds)),'Missing/duplicate raw pairs')
  require(set(row['metrics'])=={'decode_speedup','ttft_speedup','request_speedup'},'Missing metrics')
  for metric in row['metrics'].values():
   require(metric['paired_observations']==rounds and len(metric['percentile_95_ci'])==2,'Invalid interval sample count')
   require(all(math.isfinite(v)and v>0 for v in [metric['median_paired_speedup'],*metric['percentile_95_ci']]),'Nonfinite metrics')
 # No condition on whether any interval is above/below one.
 return c,a

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--candidate-tag',choices=['prefill-query-row-pair','queryrow-b1-headpair-v1'],required=True)
 p.add_argument('--phase',choices=['worker','vs19','vsllama'],required=True)
 p.add_argument('--worker-prefix',help='Exact basename before -d8-standard-confirmation.json; required only for worker phase')
 a=p.parse_args();require((a.worker_prefix is not None)==(a.phase=='worker'),'Worker prefix only for worker phase')
 if a.worker_prefix:require(a.worker_prefix and all(c.islower()or c.isdigit()or c=='-'for c in a.worker_prefix),'Invalid worker prefix')
 prefix=(a.worker_prefix+'-d8')if a.phase=='worker'else f'{a.candidate_tag}-final-{a.phase}'
 manifest=ROOT/f'reports/native-cpu-{prefix}-archive-v1.json';require(not manifest.exists(),'Archive inventory already exists')
 prepared=[];metrics=[]
 for suite in ('standard','extended'):
  conf=BASE/f'{prefix}-{suite}-confirmation.json';analysis=BASE/f'{prefix}-{suite}-analysis.json'
  c,q=validate_pair(conf,analysis,a.phase,suite,a.candidate_tag)
  metrics.extend(dict(suite=suite,profile=r['profile'],prompt_tokens=r['prompt_tokens'],metrics=r['metrics'])for r in q['comparisons'])
  for src,data in ((conf,c),(analysis,q)):
   dest=ROOT/'reports'/('native-cpu-'+src.name);digest=sha(src)
   if dest.exists():
    previous=read(dest)['archive_provenance'];require(previous.get('source_sha256',previous.get('raw_source_sha256'))==digest,'Existing archive has different source')
   prepared.append((src,dest,data,digest))
 # All suites validated before any write. Full raw reports preserve all pairs/outliers.
 records=[]
 for src,dest,data,digest in prepared:
  require(sha(src)==digest,'Source changed before archive')
  if not dest.exists():
   archived=clean(data);archived['archive_provenance']=dict(source_report=str(src.relative_to(ROOT)),source_sha256=digest,sanitization='Repository/home paths and process identifiers/names removed; host pcpu uses originalcolumn2,pmem3,elapsed4. All scores, pairs, confidence intervals and negative outcomes retained.')
   with dest.open('x')as stream:stream.write(json.dumps(archived,separators=(',',':'),allow_nan=False)+'\n')
  records.append(dict(source=str(src.relative_to(ROOT)),source_sha256=digest,archive=str(dest.relative_to(ROOT)),archive_sha256=sha(dest)))
 require(all(sha(src)==digest for src,_,_,digest in prepared),'Source changed during archive')
 record=dict(completed=True,inference_run=False,candidate_tag=a.candidate_tag,phase=a.phase,records=records,all_profiles=metrics,archive_runner_sha256=sha(Path(__file__)),absolute_numerical_qualified=False,performance_claim_eligible=False,release_eligible=False,automatic_selection=False,limitations=['Known baseline absolute63/64 oracle failures remain failures.','Intervals and every raw pair are retained regardless of sign. No pooling with prior or later runs.','Worker comparisons use samebinary12vs8; final kernel/reference comparisons use matched12/6. Worker choice is not inferred.','This is an archive integrity check, not a fresh rehash of model/binary payloads or a replacement for release gates.','No automatic win, noninferiority, quality improvement, release or deployment conclusion.'])
 with manifest.open('x')as stream:stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
 print(manifest)
if __name__=='__main__':main()
