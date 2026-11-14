#!/usr/bin/env python3
"""Root-run ordinary STEP PV boundary comparison; no absolute oracle claim."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'build/native-cpu'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def guarded(name,args):subprocess.run([sys.executable,str(ROOT/'tools/benchmark_guard.py'),'--name',name,'--',sys.executable,*map(str,args)],cwd=ROOT,check=True)
reference=B/'build-20/qwen_cpu';candidate=B/'decode-attention-active8-v2-build/qwen_cpu'
source=B/'extended-tutorial-fixtures/context3968.json';folder=B/'decode-attention-active8-v2-step-edge-fixtures'
output=B/'decode-attention-active8-v2-step-edge-exact-gate.json';assert not folder.exists() and not output.exists()
lengths=(30,31,32,62,63,64)
paths=[source,Path(__file__).resolve()]+[ROOT/'tools'/n for n in ('capture_native_cpu_logits.py','compare_native_cpu_captures.py','native_cpu_chat.py','benchmark_guard.py')]
for binary in (reference,candidate):
 m=read(binary.parent/'manifest.json');assert m['completed'] and m['binary_sha256']==sha(binary)
 paths.extend((binary,binary.parent/'manifest.json'))
 for path,digest in m['source_sha256'].items():assert sha(path)==digest;paths.append(Path(path))
for n in lengths:
 for arm in ('reference','candidate'):assert not (B/f'active8-v2-step-edge-{n}-{arm}-logits').exists()
folder.mkdir();ids=read(source)
for n in lengths:
 p=folder/f'prompt-{n}.json';p.write_text(json.dumps(ids[:n])+'\n');paths.append(p)
 p=folder/f'continuation-{n}.json';p.write_text(json.dumps([ids[n]])+'\n');paths.append(p)
frozen={str(p):sha(p) for p in paths}
(folder/'manifest.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),prefix_lengths=list(lengths),ordinary_step_positions=[n+1 for n in lengths],note='Everycapture usesprefill(n) thenexplicitstep(nextfrozenpromptID),2positions2repeats. Twelveof24comparedarrays areB1STEPoutputs.',inputs=frozen),indent=2)+'\n')
refs=[];cands=[]
for n in lengths:
 for arm,binary,collection in [('reference',reference,refs),('candidate',candidate,cands)]:
  assert all(sha(p)==h for p,h in frozen.items())
  directory=B/f'active8-v2-step-edge-{n}-{arm}-logits'
  guarded(f'Active8 STEPedge{n+1} {arm}',[ROOT/'tools/capture_native_cpu_logits.py','--binary',binary,'--weights',B/'weights.bin','--prompt',folder/f'prompt-{n}.json','--continuation',folder/f'continuation-{n}.json','--tokens',2,'--threads',12,'--prefill-threads',6,'--directory',directory]);collection.append(directory/'report.json')
  capture=read(directory/'report.json');assert capture['completed'] and len(capture['runs'])==2
  assert all(run['consumed_continuation_ids']==[ids[n]] and [d['context_tokens'] for d in run['logit_dumps']]==[n,n+1] for run in capture['runs'])
guarded('Active8 ordinarySTEP edges exact24',[ROOT/'tools/compare_native_cpu_captures.py','--reference',*refs,'--candidate',*cands,'--output',output])
j=read(output);assert j['completed'] and j['passed'] and j['compared_arrays']==24
assert all(sha(p)==h for p,h in frozen.items())
print('24arrays match fresh baseline20, including12B1STEP outputs atpositions31/32/33/63/64/65; absolute failures remain separate.')
