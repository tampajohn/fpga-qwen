#!/usr/bin/env python3
"""Guarded real-adapter wall attribution; diagnostic only, no kernel/protocol edits."""
import argparse,hashlib,json,math,os,threading,time
from pathlib import Path
from types import SimpleNamespace
from benchmark_native_phase0 import guard
from chat_protocol import ChatProtocol,ChatRequest
from experiment import locate_model
from native_cpu_chat import NativeCPUChat
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(ok,message):
 if not ok:raise RuntimeError(message)
def save(path,value):
 pending=path.with_suffix('.pending.json');pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');pending.replace(path)
class TimedTokenizer:
 def __init__(self,tokenizer):self.original=tokenizer;self.calls=[]
 def decode(self,ids,**kwargs):
  start=time.perf_counter_ns();result=self.original.decode(ids,**kwargs);end=time.perf_counter_ns()
  self.calls.append(dict(input_ids=len(ids),seconds=(end-start)*1e-9));return result
class TimedNative(NativeCPUChat):
 def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.commands=[]
 def command(self,command,cancel,**kwargs):
  begin=time.perf_counter_ns();response=super().command(command,cancel,**kwargs);end=time.perf_counter_ns()
  operation=command.split(' ',1)[0];elapsed=(end-begin)*1e-9;native=response.get('seconds')
  if operation in ('prefill','step','verify-greedy'):
   require(type(native)in(int,float)and math.isfinite(native)and native>=0,'Missing native elapsed seconds')
  row=dict(operation=operation,wall_seconds=elapsed,engine_reported_seconds=native,position=response.get('position'))
  if native is not None:row['wall_minus_engine_seconds']=elapsed-native
  self.commands.append(row);return response

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--binary',type=Path,required=True);p.add_argument('--weights',type=Path,default=ROOT/'build/native-cpu/weights.bin')
 p.add_argument('--prompt',type=Path,required=True);p.add_argument('--tokens',type=int,default=32)
 p.add_argument('--decoding',choices=['greedy','chat'],default='greedy');p.add_argument('--seed',type=int,default=42)
 p.add_argument('--max-draft',type=int,choices=[0,1],default=0);p.add_argument('--threads',type=int,default=12);p.add_argument('--prefill-threads',type=int,default=6)
 p.add_argument('--checkpoint');p.add_argument('--directory',type=Path,required=True)
 a=p.parse_args();require(1<=a.tokens<=256 and 1<=a.threads<=64 and 1<=a.prefill_threads<=64,'Invalid token/worker count')
 require(not a.max_draft or a.decoding=='greedy','Suffix is greedy-only')
 guarded=guard(ROOT/'build/benchmark-status');directory=a.directory.resolve();directory.mkdir(parents=True,exist_ok=False);output=directory/'report.json'
 report=dict(completed=False,passed=False,diagnostic_only=True,performance_claim_eligible=False,gpu=False,pytorch=False,guard_before=guarded,visits=[],limitations=['Wall attribution is instrumented, not a throughput comparison or measured speedup.','C++seconds includes model+consumed-history+sampler; excludes command parsing and response formatting/flushing.','Command wall minus C++seconds includes pipe/scheduling/JSON/Python overhead; it is not pure IPC.','Tokenizer and JSON serialization are directly timed; HTTP/network write/flush/backpressure and browser rendering are not measured.','No worker-wait duration is attributed to removable synchronization.','No full-logit arithmetic change: identical immutable executable; finite stream equivalence is checked, not a new absolute numerical qualification.','Two repeat controls are diagnostics, not ten-pair performance evidence.'])
 engine=None;original_env=dict(os.environ)
 try:
  binary=a.binary.resolve();weights=a.weights.resolve();prompt=a.prompt.resolve();manifest_path=binary.parent/'manifest.json';manifest=json.loads(manifest_path.read_text())
  require(manifest['completed']is True and manifest['binary_sha256']==sha(binary),'Build manifest mismatch')
  require(manifest['gpu']is False and manifest['pytorch']is False,'CPU binary required')
  ids=json.loads(prompt.read_text());require(isinstance(ids,list)and ids and all(type(t)is int and 0<=t<151936 for t in ids),'Invalid prompt IDs')
  require(len(ids)+a.tokens-1<=4096,'Context capacity exceeded')
  checkpoint=locate_model(a.checkpoint);protocol=ChatProtocol(checkpoint)
  paths=[Path(__file__).resolve(),binary,manifest_path,weights,Path(str(weights)+'.index'),Path(str(weights)+'.manifest.json'),prompt]
  paths += [ROOT/'tools'/n for n in ('native_cpu_chat.py','suffix.py','chat_protocol.py','experiment.py','benchmark_native_phase0.py','benchmark_guard.py')]
  paths += [checkpoint/n for n in ('tokenizer.json','tokenizer_config.json','config.json')]
  report['inputs']={str(path.resolve()):sha(path)for path in paths};report['prompt_ids']=ids
  report['configuration']=dict(binary=str(binary),weights=str(weights),decoding=a.decoding,seed=a.seed,max_draft=a.max_draft,tokens=a.tokens,decode_threads=a.threads,prefill_threads=a.prefill_threads)
  removed=[]
  for key in list(os.environ):
   if key.startswith(('DYLD_','GGML_','LLAMA_ARG_'))or key=='VECLIB_MAXIMUM_THREADS':removed.append(key);os.environ.pop(key)
  report['removed_environment_names']=sorted(removed);save(output,report)
  expected=None
  for repeat,order in [(0,('plain','instrumented')),(1,('instrumented','plain'))]:
   for arm in order:
    row=dict(repeat=repeat,arm=arm,completed=False);report['visits'].append(row);save(output,report)
    tokenizer=TimedTokenizer(protocol.tokenizer)
    target_protocol=SimpleNamespace(tokenizer=tokenizer)if arm=='instrumented'else protocol
    cls=TimedNative if arm=='instrumented'else NativeCPUChat
    engine=cls(target_protocol,binary=binary,weights=weights,threads=a.threads,prefill_threads=a.prefill_threads,reuse_prefix=False)
    cancel=threading.Event();start=time.perf_counter_ns();engine.start(cancel);row['startup_seconds']=(time.perf_counter_ns()-start)*1e-9;row['ready']=engine.ready
    request=ChatRequest('overhead-diagnostic',ids,a.tokens,a.max_draft,decoding=a.decoding,seed=a.seed)
    # Same full request warmup, excluded from the attribution visit.
    warm=list(engine.stream(request,cancel));require(warm[-1]['type']=='done','Warmup incomplete')
    if arm=='instrumented':engine.commands.clear();tokenizer.calls.clear()
    stream=engine.stream(request,cancel);events=[];next_calls=[];serialization=[]
    while True:
     begin=time.perf_counter_ns()
     try:event=next(stream)
     except StopIteration:break
     end=time.perf_counter_ns();next_calls.append(dict(event=event['type'],seconds=(end-begin)*1e-9))
     # Mirror server JSON encoding only, no socket or pretend network measurements.
     begin=time.perf_counter_ns();encoded=(json.dumps(event)+'\n').encode();end=time.perf_counter_ns()
     serialization.append(dict(event=event['type'],bytes=len(encoded),seconds=(end-begin)*1e-9));events.append(event)
    require(events and events[-1]['type']=='done','Stream incomplete');done=events[-1]
    signature=[(e['type'],e.get('token_ids'),e.get('text'),e.get('finish_reason'))for e in events if e['type']in('update','done')]
    warm_signature=[(e['type'],e.get('token_ids'),e.get('text'),e.get('finish_reason'))for e in warm if e['type']in('update','done')]
    require(signature==warm_signature,'Warm/observed trajectory differs')
    if expected is None:expected=signature
    require(signature==expected,'Plain/instrumented/repeat token/text/stop stream differs')
    require(done['finish_reason']in('length','eos')and done['metrics']['reused_prefix_tokens']==0,'Unexpected stop/cache reuse')
    row.update(completed=True,stream=signature,metrics=done['metrics'],next_calls=next_calls,serialization=serialization)
    if arm=='instrumented':
     commands=engine.commands;tokenizer_seconds=sum(t['seconds']for t in tokenizer.calls);command_seconds=sum(c['wall_seconds']for c in commands)
     decode=[c for c in commands if c['operation']in('step','verify-greedy')]
     row.update(commands=commands,tokenizer_calls=tokenizer.calls,attribution=dict(generator_next_seconds=sum(c['seconds']for c in next_calls),command_wall_seconds=command_seconds,tokenizer_seconds=tokenizer_seconds,other_generator_seconds=sum(c['seconds']for c in next_calls)-command_seconds-tokenizer_seconds,json_serialization_seconds=sum(c['seconds']for c in serialization),decode_command_wall_seconds=sum(c['wall_seconds']for c in decode),decode_engine_reported_seconds=sum(c['engine_reported_seconds']for c in decode),decode_command_residual_seconds=sum(c['wall_minus_engine_seconds']for c in decode)))
    engine.close();engine=None;save(output,report)
  require(all(sha(Path(path))==digest for path,digest in report['inputs'].items()),'Frozen artifact changed')
  report.update(completed=True,passed=True,stream_equivalence_passed=True,guard_after=guard(ROOT/'build/benchmark-status'))
 except BaseException as error:report['error']=dict(type=type(error).__name__,message=str(error));raise
 finally:
  if engine is not None:
   try:engine.close()
   except BaseException as error:report['cleanup_error']=str(error);report['completed']=False;report['passed']=False
  os.environ.clear();os.environ.update(original_env);save(output,report)
 print(output)
if __name__=='__main__':main()
