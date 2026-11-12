"""Final frozen fused phase profile versus CPU-only llama FP16/FP32; fresh KV."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
import benchmark_matched as matched
from benchmark_matched import SVEngine,NativeEngine,make_prompt
from chat_protocol import ChatProtocol
from experiment import locate_model
from check_phase_pools_real import telemetry
from compare_final_fma import read_dump_logits
FROZEN='6c9ab44bf49b9f35ebf1a86897c284640041fb2a1a811f38184bff306f9e1b89'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
class FinalEngine(SVEngine):
 def start(self):
  self.model.phase_pool_selector='PREFILL_12';start=time.perf_counter();self.model.start()
  actual=json.loads(self.model.command('precision'))
  expected=dict(precision_profile='fp32_b1_fp32_fma_prefill',canonical_prefix_policy='fp32_fma_b8_aligned_v1',scalar_projection='fp32',prefill_projection='fp32',verification_projection='fp32',vocabulary_projection='fp32',scalar_accumulation='separate',prefill_accumulation='fused',verification_accumulation='separate',vocabulary_accumulation='separate',attention_kv='fp64',max_verification_block=4,prefix_reuse_supported=False)
  if not isinstance(actual,dict)or any(type(actual.get(k))is not type(v)or actual[k]!=v for k,v in expected.items()):raise RuntimeError('Unexpected final precision capability')
  self.model.set_lanes(self.lanes)
  telemetry(self.model,True);self.metadata.update(runtime_capabilities=actual,prefill_workers=12,decode_workers=8,resident_worker_threads=20,phase_selector='PREFILL_12')
  return time.perf_counter()-start
 def generate(self,*args,**kwargs):
  before=telemetry(self.model,True);r=super().generate(*args,**kwargs);after=telemetry(self.model,True)
  delta=[[after['dispatch_counts'][p][k]-before['dispatch_counts'][p][k]for k in range(3)]for p in range(2)]
  if delta[0][0]<=0 or delta[1][1]<=0 or any(delta[p][k]for p,k in ((0,1),(0,2),(1,0),(1,2))):raise RuntimeError('Unexpected phase routing')
  r.update(phase_before=before,phase_after=after,phase_delta=delta)
  return r

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--native-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 binary=a.binary.resolve();matched.ROOT=a.native_root.resolve()
 if sha(binary)!=FROZEN:raise RuntimeError('Unexpected candidate')
 cache=matched.ROOT/'build/llama-cpu/CMakeCache.txt';cache_text=cache.read_text()
 for name in ('METAL','CUDA','VULKAN','OPENCL','HIP','MUSA','SYCL'):
  if f'GGML_{name}:BOOL=OFF'not in cache_text:raise RuntimeError('GPU backend must be disabled: '+name)
 native=matched.ROOT/'build/llama-cpu/bin/llama-server';checkpoint=locate_model(None);protocol=ChatProtocol(checkpoint)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 report=dict(completed=False,external_contention='Run through the serving checkout benchmark_guard to exclude browser chat. This helper does not monitor external CPU processes or establish OS-wide isolation.',gpu=False,pytorch=False,candidate_sha256=FROZEN,native_binary_sha256=sha(native),native_build_cache_sha256=sha(cache),native_build_options=[line for line in cache_text.splitlines()if line.startswith('GGML_')and ':BOOL='in line],llama_commit=subprocess.check_output(['git','-C',str(matched.ROOT/'build/llama.cpp'),'rev-parse','HEAD'],text=True).strip(),source_sha256={name:sha(Path(__file__).with_name(name))for name in ('benchmark_final_native.py','benchmark_matched.py','check_phase_pools_real.py','rtl_model.py','compare_final_fma.py')},runs=[],limitations=['Same85/452 prompt IDs,32 greedy outputs,EOS respected,empty logical KV and two warm repetitions.','SV12 prefill/8 decode workers versus native12 batch/8 decode threads; numeric counts match, implementation and placement need not.','SV BF16 weights, fusedFP32 prefill/separateFP32 decode,FP64 KV; native named weight precision,mixed kernels,FP16 KV.','Native Apple BLAS may be enabled; no GPU backend or runtime offload.','Sequential engine sessions, existing SV pipes versus nativeHTTP; no OS cache eviction or core placement.','Cache reuse and suffix disabled. No cross-precision token/logit equality assumed; output trajectories reported.'])
 def save():a.output.write_text(json.dumps(report,indent=2)+'\n')
 try:
  with tempfile.TemporaryDirectory()as tmp:
   engines=[('rtl-final',FinalEngine(checkpoint,8,protocol.tokenizer,binary,8,1,8,'fp32_b1_fp32_fma_prefill',4)),('llama-f16',NativeEngine('f16',8,Path(tmp),batch_threads=12)),('llama-f32',NativeEngine('f32',8,Path(tmp),batch_threads=12))]
   for name,e in engines:
    try:
     startup=e.start()
     for profile in ('short','long'):
      ids=make_prompt(protocol,profile);rows=[];bits=None
      for repetition in range(3):
       row=e.generate(ids,32,True)
       if row['output_token_count']!=32 or row['eos_observed']:raise RuntimeError('EOS/count mismatch')
       if name!='rtl-final' and (not isinstance(row.get('backend_timings'),dict) or row['backend_timings'].get('cache_n')!=0 or row['backend_timings'].get('prompt_n')!=len(ids)):raise RuntimeError('Native prompt cache/work mismatch')
       if rows and row['output_token_ids']!=rows[0]['output_token_ids']:raise RuntimeError('Greedy request not repeatable')
       if name=='rtl-final':
        current=read_dump_logits(e.model,Path(tmp)/'logits').tobytes()
        if bits is not None and bits!=current:raise RuntimeError('SV full logits not repeatable')
        bits=current;row['logits_sha256']=hashlib.sha256(bits).hexdigest()
       rows.append(row)
      warm=rows[1:];record=dict(engine=name,profile=profile,prompt_ids=ids,prompt_tokens=len(ids),startup_seconds=startup,metadata=e.metadata,first_request=rows[0],warm_requests=warm,median_ttft_seconds=statistics.median(r['first_token_seconds']for r in warm),median_decode_tokens_per_second=statistics.median(r['steady_decode_tokens_per_second']for r in warm),median_request_seconds=statistics.median(r['request_seconds']for r in warm))
      if name!='rtl-final':record.update(command=e.command,gguf_sha256=sha(e.model))
      report['runs'].append(record);save();print(json.dumps({k:record[k]for k in ('engine','profile','median_ttft_seconds','median_decode_tokens_per_second')}),flush=True)
    finally:
     e.close()
     if name!='rtl-final':
      log=(Path(tmp)/f'llama-{e.precision}.log').read_text();report.setdefault('native_runtime_logs',{})[name]=log
      if any(f'--{flag}'not in e.command for flag in ('no-kv-offload','no-op-offload'))or e.command[e.command.index('-ngl')+1]!='0':raise RuntimeError('Native offload flag mismatch')
   if sha(binary)!=FROZEN or sha(native)!=report['native_binary_sha256']:raise RuntimeError('Executable changed')
   report['completed']=True
 finally:save()
if __name__=='__main__':main()
