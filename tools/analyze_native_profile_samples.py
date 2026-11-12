#!/usr/bin/env python3
"""Summarize a local exported Time Profiler table; no model execution.

The raw TOC can contain sensitive inherited environment values. Only its
start-date and duration are retained; never print or publish the raw TOC.
Run large parses under tools/benchmark_guard.py in an isolated CPU window.
"""
import argparse
import collections,datetime,hashlib,json,pathlib,xml.etree.ElementTree as ET
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-dir", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
D = args.input_dir
if args.output.exists():
 raise SystemExit("Refusing to overwrite an existing analysis output")
report=json.loads((D/'report.json').read_text())
toc=ET.parse(D/'trace-toc.xml');start_text=toc.findtext('.//summary/start-date');duration=float(toc.findtext('.//summary/duration'))
origin=round(datetime.datetime.fromisoformat(start_text).timestamp()*1e9)
root=ET.parse(D/'time-profile-samples.xml').getroot();refs={e.get('id'):e for e in root.iter() if e.get('id')}
def resolve(e):
 while e is not None and e.get('ref'):e=refs[e.get('ref')]
 return e
def child(e,tag):return resolve(resolve(e).find(tag))
stack_cache={}
def stack(e):
 e=resolve(e);key=e.get('id')
 if key not in stack_cache:
  b=child(e,'backtrace');frames=[]
  for f in b if b is not None else []:
   f=resolve(f);binary=child(f,'binary');frames.append((f.get('name','unknown'),binary.get('name','unknown') if binary is not None else 'unknown'))
  stack_cache[key]=frames or [('(no user backtrace)','unknown')]
 return stack_cache[key]
windows=[]
for i,r in enumerate(report['requests']):
 w=r['profiling_windows']['estimated_unix_ns'];windows.extend([(w['request_start']-origin,w['first_token']-origin,'prefill',i),(w['first_token']-origin,w['last_token']-origin,'decode',i)])
def phase(t,margin):
 for a,b,p,i in windows:
  if a<=t<=b:
   return ('boundary' if t-a<margin or b-t<margin else p),i
 return 'outside_requests',None
stats={m:collections.defaultdict(lambda:dict(count=0,missing_count=0,weight=0,leaves=collections.Counter(),categories=collections.Counter(),threads=collections.Counter())) for m in (0,2000000)}
states=collections.Counter();pids=collections.Counter();top_stacks=collections.Counter();first=None;last=None

def category(frames):
 names=[f[0] for f in frames];text=' '.join(names);leaf=names[0]
 if 'llama_sampler' in text or 'common_sampler' in text:return 'sampling'
 if 'ggml_barrier' in leaf:return 'explicit_worker_barrier'
 if leaf.startswith(('ggml_graph_compute_secondary_thread','ggml_graph_compute_thread')):return 'worker_graph_loop_unresolved'
 if any(x in text for x in ('flash_attn','soft_max','ggml_compute_forward_rope')):return 'attention_softmax_rope'
 if any(x in text for x in ('ggml_compute_forward_mul_mat','ggml_backend_blas_graph_compute','cblas_sgemm','kai_run_matmul','ggml::cpu::kleidiai')):return 'matrix_multiply_pack_path'
 if any(x in leaf for x in ('ggml_backend_sched','ggml_gallocr','llama_graph','ggml_new_tensor')):return 'graph_scheduling_allocation'
 if any(x in text for x in ('ggml_compute_forward_rms_norm','ggml_compute_forward_silu','ggml_compute_forward_mul','ggml_compute_forward_add','ggml_compute_forward_cont','ggml_compute_forward_cpy')):return 'other_neural_or_tensor_ops'
 return 'other_unclassified'
for row in root.iter('row'):
 children=list(row)
 if len(children)!=7:raise RuntimeError('Unexpectedrow')
 t=int(resolve(children[0]).text);proc=resolve(children[2]);pid=int(child(proc,'pid').text);pids[pid]+=1
 if pid!=report['server']['pid']:continue
 state=resolve(children[4]).text;states[state]+=1
 weight=int(resolve(children[5]).text);frames=stack(children[6]);tid=int(child(children[1],'tid').text)
 first=t if first is None else min(first,t);last=t if last is None else max(last,t)
 for margin in stats:
  p,i=phase(t,margin);v=stats[margin][p];v['count']+=1;v['weight']+=weight;v['leaves'][frames[0]]+=weight;v['categories'][category(frames)]+=weight;v['threads'][tid]+=weight
  if frames[0][0]=='(no user backtrace)':v['missing_count']+=1
 if phase(t,2000000)[0]=='decode':top_stacks[tuple(frames[:5])]+=weight
out=dict(completed=True,diagnostic_only=True,source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
 sample_export_sha256=hashlib.sha256((D/'time-profile-samples.xml').read_bytes()).hexdigest(),target_pid=report['server']['pid'],trace_start=start_text,trace_duration_seconds=duration,
 sample_time_range_seconds=[first/1e9,last/1e9],sample_pids=dict(pids),target_thread_states=dict(states),request_count=len(report['requests']),
 interpretation='CPU sample-weight shares, not wall-time fractions. Only target PID. No head attribution. Matrix-multiply/packing path may include attention matmuls where shared symbols hide tensor identity. Worker graph-loop samples are unresolved loop/inlined-work/polling, not proven removable overhead.',
 thread_count_limits='Unique observed operating-system TIDs, not simultaneous workers. Thread churn attribution requires separate evidence.',
 timestamp_limits='Trace start-date is printed to milliseconds; HTTP arrival phase boundaries are approximate. Report both raw windows and conservative2ms trimming at each phase edge.',phases={})
for margin,groups in stats.items():
 result={}
 for p,v in groups.items():
  missing_weight=v['leaves'].get(('(no user backtrace)','unknown'),0)
  result[p]=dict(samples=v['count'],sample_weight_seconds=v['weight']/1e9,unique_threads=len(v['threads']),missing_user_stack_samples=v['missing_count'],missing_user_stack_weight_seconds=missing_weight/1e9,missing_user_stack_share_percent=100*missing_weight/v['weight'],categories={k:dict(sample_weight_seconds=n/1e9,share_percent=100*n/v['weight']) for k,n in v['categories'].most_common()},
   hot_leaf_symbols=[dict(symbol=k[0],binary=k[1],sample_weight_seconds=n/1e9,share_percent=100*n/v['weight']) for k,n in v['leaves'].most_common(30)])
 out['phases'][f'edge_trim_{margin//1000000}ms']=result
out['decode_top_stacks_2ms']=[dict(frames=[dict(symbol=f[0],binary=f[1]) for f in key],sample_weight_seconds=n/1e9) for key,n in top_stacks.most_common(25)]
out['phase_wall_windows_seconds']={p:sum(max(0,min(b,duration*1e9)-max(a,0)) for a,b,kind,i in windows if kind==p)/1e9 for p in ('prefill','decode')}
out['fully_recorded_decode_windows']=sum(a>=0 and b<=duration*1e9 for a,b,p,i in windows if p=='decode')
offsets=[a['unix_time_ns']-(a['perf_counter_before_ns']+a['perf_counter_after_ns'])//2 for r in report['requests'] for a in (r['profiling_windows']['clock_before'],r['profiling_windows']['clock_after'])]
out['clock_offset_range_ns']=max(offsets)-min(offsets)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ('sample_time_range_seconds','sample_pids','target_thread_states','fully_recorded_decode_windows','phase_wall_windows_seconds')},indent=2))
for p,v in out['phases']['edge_trim_2ms'].items():
 print(p, v['samples'],v['sample_weight_seconds'],v['categories'])
 if p=='decode': print(json.dumps(v['hot_leaf_symbols'][:15],indent=2))
