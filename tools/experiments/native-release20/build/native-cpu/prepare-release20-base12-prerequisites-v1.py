#!/usr/bin/env python3
"""Root-authorized evidence manifest builder. Never stages, selects by speed or deploys."""
import argparse,json,os,sys
from pathlib import Path
from release20_common import ROOT,BASE,sha,read,selected
sys.path.insert(0,str(ROOT/'tools'))
from benchmark_native_phase0 import guard
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--candidate-tag',choices=['prefill-query-row-pair','queryrow-b1-headpair-v1'],required=True)
p.add_argument('--explicit-root-selection',action='store_true',required=True)
p.add_argument('--selection-rationale',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert a.selection_rationale.strip()
os.chdir(ROOT);guard(ROOT/'build/benchmark-status')
out=a.output.resolve();assert not out.exists();pending=out.with_name(out.name+'.pending');assert not pending.exists()
tag=a.candidate_tag;candidate=BASE/f'{tag}-build';evidence={}
def add(role,suffix,required,rounds=None):
 path=BASE/f'{tag}-{suffix}';r=read(path)
 assert r['completed'] is True and not r.get('error') and not r.get('cleanup_error')
 item=dict(path=str(path),sha256=sha(path),required=required)
 if rounds is not None:item['expected_rounds']=rounds
 evidence[role]=item
add('relative_exact','exact-gate.json',{'passed':True,'compared_arrays':56})
add('timeboundary_exact','timeboundary-exact-gate.json',{'passed':True,'compared_arrays':12})
add('newboundary_relative_exact','newboundary-relative-exact-gate.json',{'passed':True,'compared_arrays':8})
if tag=='queryrow-b1-headpair-v1':add('b1_step_edges','step-edge-exact-gate.json',{'passed':True,'compared_arrays':24})
add('api','verify-api/report.json',{'passed':True,'full_acceptance_matrix_passed':True,'workers':[1,12,64],'prefixes':[0,1,511,512,513,3966],'commits':[1,2]})
add('prefix','prefix-check/report.json',{'passed':True})
add('suffix','suffix-trajectories.json',{'passed':True,'original35_passed':True,'cancellation.passed':True,'restart.passed':True})
add('suffix_cli','suffix-cli/report.json',{'passed':True,'invalid_and_chat_commands_reset':True})
add('quality','quality.json',{'passed':True,'exact_trajectory_gate_passed':True,'no_new_regressions':True})
add('extended_quality','extended-quality.json',{'passed':True,'exact_trajectory_gate_passed':True,'no_new_case_regressions':True,'no_new_check_regressions':True})
add('seeded_stream','seeded-stream/report.json',{'passed':True,'seeds':[42,1337],'repeats':2,'requested_tokens':20})
for suite,rounds in [('standard',20),('extended',10)]:
 add(suite+'_confirmation',f'final-vs19-{suite}-confirmation.json',{'completed':True,'paired_sampling_complete':True,'rounds':rounds,'context_suite':suite},rounds)
 add(suite+'_analysis',f'final-vs19-{suite}-analysis.json',{'completed':True,'rounds':rounds,'baseline':'native-reference-build-19'},rounds)
# Contextual tuned-llama evidence is required too, without treating it as a release speed threshold.
for suite in ('standard','extended'):
 add('llama_'+suite+'_confirmation',f'final-vsllama-{suite}-confirmation.json',{'completed':True,'paired_sampling_complete':True,'rounds':10,'context_suite':suite},10)
 add('llama_'+suite+'_analysis',f'final-vsllama-{suite}-analysis.json',{'completed':True,'rounds':10,'baseline':'llama-f16-onon-d12-p6'},10)
# Common validator reserves *_confirmation/_analysis for its strict vs19 two-role checks.
# Keep contextual evidence separate and bind its identity/counts explicitly here.
contextual={k:evidence.pop(k) for k in list(evidence) if k.startswith('llama_')}
for suite in ('standard','extended'):
 conf=contextual[f'llama_{suite}_confirmation'];analysis=contextual[f'llama_{suite}_analysis'];c=read(conf['path']);z=read(analysis['path'])
 assert c['rounds']==10 and len(c['visits'])==20 and c['tokens']==(32 if suite=='standard' else 128) and c['warmups_per_visit_profile']==1
 assert c['preflight_same_precision_ids_passed'] and c['paired_sampling_complete']
 assert c['configurations'][0]['name']=='llama-f16-onon-d12-p6'
 assert Path(c['configurations'][1]['binary']).resolve()==(candidate/'qwen_cpu').resolve()
 assert c['inputs'][str(candidate/'qwen_cpu')]==sha(candidate/'qwen_cpu')
 assert z['completed'] and z['rounds']==10 and z['source_report_sha256']==conf['sha256'] and z['baseline']=='llama-f16-onon-d12-p6'
failed={}
for label,name in [('boundary63','build19-newboundary63-control-oracle.json'),('boundary64','build19-original-prefix-control-boundary64-oracle.json')]:
 path=BASE/name;j=read(path);assert j['completed'] and j['passed'] is False
 failed[label]=dict(path=str(path),sha256=sha(path),passed=False)
record=dict(candidate_tag=tag,authorize_selection=True,baseline_release='build-19',policy='exact_nonregression_with_inherited_baseline19_gap',known_absolute_failures=['boundary63','boundary64'],absolute_oracle_pass_claimed=False,selection_rationale=a.selection_rationale,candidate_binary_sha256=sha(candidate/'qwen_cpu'),candidate_manifest_sha256=sha(candidate/'manifest.json'),runtime_defaults=None,evidence=evidence,contextual_llama_evidence=contextual,known_failure_evidence=failed,builder_sha256=sha(Path(__file__).resolve()),serving_scope='Base12decode/6prefill only; no conditional worker factory patch authorized.')
pending.write_text(json.dumps(record,indent=2)+'\n')
try:
 selected(tag,pending)
 assert not out.exists();os.replace(pending,out)
except BaseException:
 # Preserve the pending manifest for review; do not publish it as completed selection.
 raise
print(out)
