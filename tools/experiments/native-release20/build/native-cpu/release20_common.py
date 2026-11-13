"""Local release20 templates. No automatic selection; root-authored evidence required."""
import hashlib,json,re,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'build/native-cpu'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,data):
 p=Path(p);tmp=p.with_name(p.name+'.pending');assert not tmp.exists();tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,p)
def selected(tag,prerequisites):
 assert re.fullmatch(r'[a-z0-9][a-z0-9-]+',tag)
 p=Path(prerequisites).resolve();j=read(p)
 assert j['candidate_tag']==tag and j['authorize_selection'] is True
 assert j['policy']=='exact_nonregression_with_inherited_baseline19_gap'
 assert j['known_absolute_failures']==['boundary63','boundary64']
 assert j['absolute_oracle_pass_claimed'] is False and j['baseline_release']=='build-19'
 assert j['selection_rationale'].strip()
 candidate=BASE/(tag+'-build');source=BASE/(tag+'-source');m=read(candidate/'manifest.json')
 assert m['completed'] and m['gpu'] is False and m['pytorch'] is False
 assert not (candidate/'qwen_cpu').is_symlink()
 assert sha(candidate/'qwen_cpu')==j['candidate_binary_sha256']==m['binary_sha256']
 assert sha(candidate/'manifest.json')==j['candidate_manifest_sha256']
 for name in ('core.cpp','core.hpp','main.cpp'):
  path=(source/name).resolve();assert sha(path)==m['source_sha256'][str(path)]
 roles={'relative_exact','newboundary_relative_exact','api','prefix','suffix','quality','extended_quality','standard_confirmation','standard_analysis','extended_confirmation','extended_analysis','timeboundary_exact','suffix_cli','seeded_stream'}
 evidence=j['evidence'];assert roles<=set(evidence)
 if 'packed_head' in (source/'core.cpp').read_text():assert 'head_worker40' in evidence
 reports={}
 for role,item in evidence.items():
  path=Path(item['path']).resolve();assert sha(path)==item['sha256']
  report=read(path);reports[role]=report;assert report['completed'] is True and not report.get('error') and not report.get('cleanup_error')
  if role not in {'standard_confirmation','standard_analysis','extended_confirmation','extended_analysis'}:assert report['passed'] is True
  if role=='api':
   assert report['full_acceptance_matrix_passed'] and not report['smoke'] and report['workers']==[1,12,64] and report['prefixes']==[0,1,511,512,513,3966] and report['commits']==[1,2]
   for name in ('core.cpp','core.hpp'):assert report['source_hashes'][str((source/name).resolve())]==sha(source/name)
  elif not role.endswith('_analysis'):
   bindings=report.get('inputs',report.get('hashes',{}))
   assert bindings.get(str((candidate/'qwen_cpu').resolve()))==m['binary_sha256'],'Candidate binary binding missing: '+role
  if role in {'relative_exact','timeboundary_exact','newboundary_relative_exact'}:assert report['compared_arrays']=={'relative_exact':56,'timeboundary_exact':12,'newboundary_relative_exact':8}[role]
  if role=='prefix':assert len(report['cases'])==28
  if role=='suffix_cli':assert len(report['cases'])==5 and report['invalid_and_chat_commands_reset']
  if role=='suffix':assert report['original35_passed'] and len(report['extended_cases'])==7 and report['cancellation']['passed'] and report['restart']['passed']
  if role=='quality':assert len(report['suite'])==9 and report['exact_trajectory_gate_passed'] and report['no_new_regressions']
  if role=='extended_quality':assert len(report['cases'])==18 and report['exact_trajectory_gate_passed'] and report['no_new_case_regressions'] and report['no_new_check_regressions']
  if role=='seeded_stream':assert report['seeds']==[42,1337] and report['repeats']==2 and report['requested_tokens']==20 and len(report['cases'])>=4
  if role=='head_worker40':assert report['compared_arrays']==40 and report['workers']==[1,2,6,12,64],'Require completed40-array worker-partition exact gate'
  if role.endswith('_confirmation'):
   suite=role.split('_')[0];rounds=item['expected_rounds']
   assert type(rounds)is int and rounds>=10,'Explicit predeclared expected_rounds>=10 required per timing role'
   assert report['mode']=='confirmation' and report['context_suite']==suite and report['rounds']==rounds and report['tokens']==(32 if suite=='standard' else 128)
   assert [Path(c['binary']).resolve() for c in report['configurations']]==[(BASE/'build-19/qwen_cpu').resolve(),(candidate/'qwen_cpu').resolve()]
   assert report['native_builds'][str((candidate/'qwen_cpu').resolve())]['binary_sha256']==m['binary_sha256']
   assert report['warmups_per_visit_profile']==1 and report['paired_sampling_complete'] and report['preflight_same_precision_ids_passed'] and len(report['visits'])==2*rounds
   names=[c['name'] for c in report['configurations']];assert len(names)==2 and len(set(names))==2
   profiles={'short','long'} if suite=='standard' else {'context1024','context2048','context3968'}
   for index in range(rounds):
    visits=report['visits'][2*index:2*index+2];order=names if index%2==0 else names[::-1]
    assert [v['config'] for v in visits]==order
    assert all(v['phase']=='scored' and v['round']==index and v['order_index']==position and set(v['profiles'])==profiles for position,v in enumerate(visits))
  # Root supplies role-specific schema predicates, counts and precision bindings.
  assert item['required'],'Every evidence item needs explicit reviewed predicates'
  for dotted,value in item['required'].items():
   actual=report
   for key in dotted.split('.'):actual=actual[key]
   assert actual==value,(role,dotted)
 for suite in ('standard','extended'):
  analysis=reports[suite+'_analysis'];confirmation=reports[suite+'_confirmation']
  expected=evidence[suite+'_confirmation']['expected_rounds']
  assert type(evidence[suite+'_analysis']['expected_rounds'])is int and evidence[suite+'_analysis']['expected_rounds']==expected
  assert analysis['rounds']==confirmation['rounds']==expected
  assert analysis['source_report_sha256']==evidence[suite+'_confirmation']['sha256']
  assert analysis['configurations']==confirmation['configurations'] and analysis['tokens']==confirmation['tokens']
  names=[c['name'] for c in confirmation['configurations']]
  assert analysis['configuration_order_by_round']==[names if i%2==0 else names[::-1] for i in range(expected)]
 validate_contextual_evidence(j,candidate,m)
 return p,j,candidate,source,m

def helper_hashes():
 names=['release20_common.py','stage-release20.py','promote-release20.py','run-release20-browser-isolated.py','run-release20-browser-live.py']
 paths=[BASE/n for n in names]+[ROOT/'tests/browser_native_cpu.py',ROOT/'tests/native_cpu_browser_server.py',ROOT/'tools/benchmark_guard.py',ROOT/'tools/chat_server.py']
 return {str(p):sha(p) for p in paths}
def unchanged(hashes):assert all(sha(p)==h for p,h in hashes.items()),'Release helper changed'


def validate_contextual_evidence(selection,candidate,manifest):
 """Revalidate optional contextual archives at every stage/promotion selection read."""
 contextual=selection.get('contextual_llama_evidence',{})
 if contextual:
  assert set(contextual)=={f'llama_{s}_{r}' for s in ('standard','extended') for r in ('confirmation','analysis')}
  for suite in ('standard','extended'):
   ci=contextual[f'llama_{suite}_confirmation'];ai=contextual[f'llama_{suite}_analysis']
   for item in (ci,ai):assert sha(item['path'])==item['sha256']
   c=read(ci['path']);a=read(ai['path'])
   assert all(x['completed'] is True and not x.get('error') and not x.get('cleanup_error') for x in (c,a))
   assert ci['expected_rounds']==ai['expected_rounds']==c['rounds']==a['rounds']==10
   assert c['mode']=='confirmation' and c['context_suite']==suite and c['tokens']==(32 if suite=='standard' else 128)
   configs=c['configurations'];assert len(configs)==2
   assert configs[0]['name']=='llama-f16-onon-d12-p6' and Path(configs[0]['build_dir']).resolve()==(ROOT/'build/native-phase0/kleidi-on-lto-on').resolve()
   assert all(x['decode_threads']==12 and x['prefill_threads']==6 for x in configs)
   binary=str((candidate/'qwen_cpu').resolve())
   assert Path(configs[1]['binary']).resolve()==Path(binary)
   assert c['inputs'][binary]==manifest['binary_sha256']==c['native_builds'][binary]['binary_sha256']
   assert c['warmups_per_visit_profile']==1 and c['paired_sampling_complete'] and c['preflight_same_precision_ids_passed'] and len(c['visits'])==20
   names=[x['name'] for x in configs];assert len(set(names))==2
   profiles={'short','long'} if suite=='standard' else {'context1024','context2048','context3968'}
   for i in range(10):
    visits=c['visits'][2*i:2*i+2];assert [v['config'] for v in visits]==(names if i%2==0 else names[::-1])
    assert all(v['phase']=='scored' and v['round']==i and v['order_index']==k and set(v['profiles'])==profiles for k,v in enumerate(visits))
   assert a['source_report_sha256']==ci['sha256'] and a['baseline']==names[0] and a['configurations']==configs and a['tokens']==c['tokens']
   assert a['configuration_order_by_round']==[names if i%2==0 else names[::-1] for i in range(10)]
 failures=selection.get('known_failure_evidence',{})
 if failures:
  assert set(failures)=={'boundary63','boundary64'}
  for item in failures.values():
   assert sha(item['path'])==item['sha256'];r=read(item['path'])
   assert r['completed'] is True and r['passed'] is False and not r.get('error') and not r.get('cleanup_error')
