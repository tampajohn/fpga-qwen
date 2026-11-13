#!/usr/bin/env python3
"""Offline stage diagnostics; strict final-array identity gates, no invented phase tolerances."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools'))
from benchmark_native_phase0 import guard
SHAPES={'input_norm_h':[1024],'q_projection':[16,128],'k_projection':[8,128],'v_projection':[8,128],'q_rope':[16,128],'k_rope':[8,128],'canonical_keys':[63,8,128],'canonical_values':[63,8,128],'attended':[2048],'post_attention_residual':[1024],'post_norm_h':[1024],'activated_mlp':[3072],'post_mlp_residual':[1024]}
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def metrics(actual,reference):
 a=actual.astype(np.float64);b=reference.astype(np.float64);d=a-b;ad=np.abs(d);index=int(ad.argmax());rr=float(np.sqrt(np.mean(b*b)));rms=float(np.sqrt(np.mean(d*d)))
 return dict(bit_identical=actual.tobytes()==reference.tobytes(),different_values=int(np.count_nonzero(actual!=reference)),max_absolute_error=float(ad.flat[index]),rms_error=rms,reference_rms=rr,relative_rms=(rms/rr if rr else None),worst_flat_index=index,actual_at_worst=float(a.flat[index]),reference_at_worst=float(b.flat[index]))
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['oracle-trace','batch-capture','sequential-capture','batch-reference','sequential-reference','batch-trace-root','sequential-trace-root','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--reference-binary',type=Path,default=ROOT/'build/native-cpu/build-19/qwen_cpu')
 p.add_argument('--instrumentation-metadata',type=Path,default=ROOT/'build/native-cpu/native-target62-trace-v1-source/experiment.json')
 p.add_argument('--guard-state-dir',type=Path,default=ROOT/'build/benchmark-status');a=p.parse_args();guard(a.guard_state_dir)
 out=a.output.resolve();assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
 r=dict(completed=False,identity_prerequisites_passed=False,inference_run=False,diagnostic_only=True,performance_claim_eligible=False,target_position=62,comparisons=[],inputs={},limitations=['No stage tolerance, pass/fail winner, model-quality or timing claim.','Local difference may be inherited from upstream input or FP16 rounding; first nonzero difference is not proof of a bug.','Only fixed targetposition62 with its63-token causal history.'])
 def save():out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 def freeze(path,digest=None):
  path=Path(path).resolve();key=str(path)
  if key not in r['inputs']:r['inputs'][key]=sha(path)
  if digest is not None:assert r['inputs'][key]==digest,key
  return path
 def loaded(path,passed=False):
  path=freeze(path);x=json.loads(path.read_text());assert x['completed'] is True and not x.get('error') and not x.get('cleanup_error')
  if passed:assert x['passed'] is True
  for field in ['inputs','source_hashes']:
   for key,value in x.get(field,{}).items():freeze(key,value)
  return x
 def vector(path,shape,digest=None):
  path=freeze(path,digest);assert path.stat().st_size==int(np.prod(shape))*4
  v=np.fromfile(path,dtype='<f4').reshape(shape);assert np.isfinite(v).all();return v
 try:
  freeze(Path(__file__).resolve());freeze(ROOT/'tools/benchmark_native_phase0.py')
  oracle=loaded(a.oracle_trace,True);assert oracle['target_position']==62 and oracle['context_tokens']==63 and oracle['phase_shapes']==SHAPES
  prompt=oracle['prompt_ids'];assert len(prompt)==63
  origin=loaded(oracle['oracle_reference']);storage=origin['oracle_logits_storage'];assert storage['prompt_ids']==prompt
  old=next(d for d in storage['dumps'] if d['step']==0 and d['context_tokens']==63);assert old['completed'] is True
  oracle_logits=vector(oracle['logits']['path'],[151936],oracle['logits']['sha256'])
  assert oracle['logits']['bit_identical'] is True and oracle['logits']['sha256']==old['sha256']
  assert oracle_logits.tobytes()==vector(old['path'],[151936],old['sha256']).tobytes()
  reference=freeze(a.reference_binary);reference_manifest=loaded(reference.parent/'manifest.json');assert reference_manifest['binary_sha256']==sha(reference)
  meta=json.loads(freeze(a.instrumentation_metadata).read_text());assert meta['baseline']=='build-19' and meta['target_global_position']==62
  for path,digest in meta['source_sha256'].items():
   original_sources=[value for key,value in reference_manifest['source_sha256'].items() if Path(key).name==Path(path).name]
   assert original_sources==[digest],'Build19 byte-copy source binding mismatch'
  for path,digest in meta['candidate_sha256'].items():freeze(path,digest)
  captures={};native_maps={}
  for name,capture_path,pristine_path,trace_root in [('batch',a.batch_capture,a.batch_reference,a.batch_trace_root),('sequential',a.sequential_capture,a.sequential_reference,a.sequential_trace_root)]:
   cap=loaded(capture_path);pristine=loaded(pristine_path)
   assert cap['prompt_ids']==pristine['prompt_ids']==prompt and cap['requested_steps']==pristine['requested_steps']==2
   assert len(cap['runs'])==len(pristine['runs'])==2 and pristine['inputs'][str(reference)]==sha(reference)
   if name=='sequential':assert cap.get('prompt_evaluation')==pristine.get('prompt_evaluation')=='sequential_step_diagnostic'
   else:assert cap.get('prompt_evaluation','canonical_prefill')==pristine.get('prompt_evaluation','canonical_prefill')=='canonical_prefill'
   binaries=[Path(path) for path in cap['inputs'] if Path(path).name=='qwen_cpu'];assert len(binaries)==1
   manifest=loaded(binaries[0].parent/'manifest.json');assert manifest['binary_sha256']==cap['inputs'][str(binaries[0])]
   for source,digest in meta['candidate_sha256'].items():assert manifest['source_sha256'][source]==digest
   r.setdefault('final_identity',{})[name]=[]
   native_maps[name]=[]
   assert {d.name for d in trace_root.iterdir()}=={'capture-000000','capture-000001'}
   for repeat,(current,oldrun) in enumerate(zip(cap['runs'],pristine['runs'])):
    assert current['own_repeat_ids_equal'] is True and current['own_repeat_logits_bit_identical'] is True
    assert current['predicted_token_ids']==oldrun['predicted_token_ids'] and current['consumed_continuation_ids']==oldrun['consumed_continuation_ids']
    assert len(current['logit_dumps'])==len(oldrun['logit_dumps'])==2
    for step,(d,e) in enumerate(zip(current['logit_dumps'],oldrun['logit_dumps'])):
     assert d['context_tokens']==e['context_tokens']==63+step
     assert vector(d['path'],[151936],d['sha256']).tobytes()==vector(e['path'],[151936],e['sha256']).tobytes()
     r['final_identity'][name].append(dict(repeat=repeat,step=step,sha256=d['sha256'],exact_reference=True))
    directory=trace_root/f'capture-{repeat:06d}';entries=[json.loads(line) for line in freeze(directory/'manifest.jsonl').read_text().splitlines()]
    assert len(entries)==366 and entries[0]['kind']=='start' and entries[-1]=={'kind':'complete','completed':True}
    header=entries[0];assert header['target_position']==62 and header['dtype']=='float32' and header['byte_order']=='little'
    assert (header['base_position'],header['batch'],header['target_row'])==((0,63,62) if name=='batch' else (62,1,0))
    values={}
    for d in entries[1:-1]:
     key=(d['layer'],d['phase']);assert key not in values and 0<=key[0]<28 and key[1] in SHAPES
     shape=SHAPES[key[1]];assert d['kind']=='tensor' and d['shape']==shape and d['dtype']=='float32' and d['byte_order']=='little'
     assert d['elements']==int(np.prod(shape)) and d['bytes']==d['elements']*4
     assert d['file']==f'layer-{key[0]:02d}-{key[1]}.f32'
     values[key]=vector(directory/d['file'],shape)
    assert len(values)==364
    assert {d.name for d in directory.iterdir()}=={'manifest.jsonl'}|{f'layer-{l:02d}-{phase}.f32' for l,phase in values}
    native_maps[name].append(values)
   assert all(native_maps[name][0][key].tobytes()==native_maps[name][1][key].tobytes() for key in native_maps[name][0]),'Stage repeat mismatch'
   captures[name]=cap
  assert captures['batch']['runs'][0]['consumed_continuation_ids']==captures['sequential']['runs'][0]['consumed_continuation_ids']
  omap={}
  for d in oracle['traces']:
   key=(d['layer'],d['phase']);assert key not in omap and key[1] in SHAPES and d['shape']==SHAPES[key[1]]
   assert d['dtype']=='float32' and d['byte_order']=='little'
   omap[key]=vector(d['path'],d['shape'],d['sha256'])
  assert set(omap)==set(native_maps['batch'][0])==set(native_maps['sequential'][0])
  r['identity_prerequisites_passed']=True;save()
  for name,actual,expected in [('batch_vs_oracle',native_maps['batch'][0],omap),('sequential_vs_oracle',native_maps['sequential'][0],omap),('batch_vs_sequential',native_maps['batch'][0],native_maps['sequential'][0])]:
   comparison=dict(name=name,stages=[],first_bit_difference=None)
   for layer in range(28):
    for phase in SHAPES:
     key=(layer,phase);v=metrics(actual[key],expected[key]);row=dict(layer=layer,phase=phase,shape=SHAPES[phase],metrics=v)
     if phase.startswith('canonical_'):row['positions']=[dict(position=t,metrics=metrics(actual[key][t],expected[key][t])) for t in range(63)]
     if not v['bit_identical'] and comparison['first_bit_difference'] is None:comparison['first_bit_difference']=dict(layer=layer,phase=phase)
     comparison['stages'].append(row)
   r['comparisons'].append(comparison);save()
  assert all(sha(Path(path))==digest for path,digest in r['inputs'].items())
  r['completed']=True
 except BaseException as error:r['error']=dict(type=type(error).__name__,message=str(error));raise
 finally:save()
 print(out)
if __name__=='__main__':main()
