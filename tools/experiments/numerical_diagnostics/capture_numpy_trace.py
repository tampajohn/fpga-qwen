#!/usr/bin/env python3
"""Guarded observational independent-oracle trace at position62; no new tolerance contract."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools'))
from benchmark_native_phase0 import guard,sha
from numpy_trace_oracle import NumpyQwen
import numpy as np
SHAPES={'input_norm_h':[1024],'q_projection':[16,128],'k_projection':[8,128],'v_projection':[8,128],'q_rope':[16,128],'k_rope':[8,128],'canonical_keys':[63,8,128],'canonical_values':[63,8,128],'attended':[2048],'post_attention_residual':[1024],'post_norm_h':[1024],'activated_mlp':[3072],'post_mlp_residual':[1024]}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--oracle-report',type=Path,required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--guard-state-dir',type=Path,default=ROOT/'build/benchmark-status');a=p.parse_args()
 guard(a.guard_state_dir);out=a.directory.resolve();out.mkdir(exist_ok=False)
 r=dict(completed=False,passed=False,gpu=False,pytorch=False,performance_claim_eligible=False,target_position=62,context_tokens=63,phase_shapes=SHAPES,traces=[],scope='Observational copied independent NumPy oracle; original FP32 math retained. Final full logits must byte-match existing persisted oracle. Trace differences do not by themselves establish implementation errors.')
 def save():(out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
 try:
  refpath=a.oracle_report.resolve();ref=json.loads(refpath.read_text())
  assert ref['completed'] is True and not ref.get('error') and not ref.get('cleanup_error')
  storage=ref['oracle_logits_storage'];assert Path(storage['report']).resolve()==refpath
  assert storage['dtype']=='float32' and storage['byte_order']=='little' and storage['numerical_conversion'] is False
  prompt=storage['prompt_ids'];assert len(prompt)==63
  old=next(row for row in storage['dumps'] if row['step']==0 and row['context_tokens']==63)
  assert old['completed'] is True and sha(Path(old['path']))==old['sha256']
  assert Path(old['path']).stat().st_size==151936*4
  config=ref['oracle'];assert config['weight_precision']=='f16' and config['kv_precision']=='f16'
  checkpoint=Path(config['checkpoint']);provenance=Path(__file__).parent/'numpy_trace_oracle-provenance.json';source=Path(__file__).parent/'numpy_trace_oracle.py'
  origin=json.loads(provenance.read_text());assert sha(source)==origin['diagnostic_sha256'] and sha(Path(origin['original']))==origin['original_sha256']
  inputs={str(refpath):sha(refpath),str(Path(old['path'])):old['sha256']}
  for group in ('inputs','source_hashes'):
   for path,digest in ref[group].items():assert sha(Path(path))==digest;inputs[path]=digest
  for path in [Path(__file__).resolve(),source,provenance,ROOT/'tools/benchmark_native_phase0.py',ROOT/'tools/experiment.py']:
   inputs[str(path)]=sha(path)
  r.update(inputs=inputs,prompt_ids=prompt,oracle_reference=str(refpath),oracle_configuration=config,original_oracle_passed=ref.get('passed'));save()
  oracle=NumpyQwen(checkpoint,weight_precision='f16',kv_precision='f16')
  expected=None
  for position,token in enumerate(prompt):expected=oracle.token(token,logits=position==62)
  assert isinstance(expected,np.ndarray) and expected.dtype==np.dtype('<f4') and expected.shape==(151936,) and np.isfinite(expected).all()
  data=expected.tobytes();digest=hashlib.sha256(data).hexdigest()
  with (out/'logits.f32').open('xb') as f:f.write(data)
  r['logits']=dict(path=str(out/'logits.f32'),sha256=digest,reference_sha256=old['sha256'],greedy=int(expected.argmax()),bit_identical=digest==old['sha256'])
  # Preserve observations even if instrumentation identity fails, but fail the run.
  for layer in range(28):
   for phase,shape in SHAPES.items():
    values=oracle.trace_rows[(layer,phase)]
    assert values.dtype==np.dtype('<f4') and list(values.shape)==shape and np.isfinite(values).all()
    path=out/f'layer-{layer:02d}-{phase}.f32';payload=values.tobytes(order='C')
    with path.open('xb') as f:f.write(payload)
    r['traces'].append(dict(layer=layer,phase=phase,shape=shape,dtype='float32',byte_order='little',path=str(path),sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload)))
  save();assert digest==old['sha256'],'Trace oracle final logits differ from persisted original; diagnostic invalid'
  assert len(r['traces'])==28*13
  for row in r['traces']:assert sha(Path(row['path']))==row['sha256'] and Path(row['path']).stat().st_size==row['bytes']
  assert all(sha(Path(path))==digest for path,digest in inputs.items())
  r.update(completed=True,passed=True)
 except BaseException as error:r['error']=dict(type=type(error).__name__,message=str(error));raise
 finally:save()
 print(out/'report.json')
if __name__=='__main__':main()
