"""Exact real-model checks for command-purpose routing; no scored timing."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from rtl_model import RTLModel
from compare_attention_combined import read_dump_logits
from benchmark_matched import make_prompt
from chat_protocol import ChatProtocol
from experiment import locate_model
def start(model,selector):
 model.phase_pool_selector=selector
 model.start()

def telemetry(model,enabled):
 data=json.loads(model.command('phase_pools'))
 assert data['selector']==('PREFILL_12'if enabled else'OFF'),data
 assert data['pool_constructions']==2 and data['decode_workers']==8 and data['prefill_workers']==(12 if enabled else 8),data
 counts=data['dispatch_counts']
 assert counts[1][0]==counts[1][2]==0,data
 assert counts[0][1]==0 if enabled else sum(counts[1])==0,data
 aggregate=list(map(int,model.command('groupstats').split()))
 assert aggregate==[sum(row[i]for row in data['group_counters'])for i in range(4)],data
 return data

def main():
 p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 models=[RTLModel(binary=path,workers=8,lanes=8,block_size=4,prefill_size=8)for path in (args.reference,args.binary,args.binary)]
 report=dict(completed=False,cases=[],telemetry=[],binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest())
 protocol=ChatProtocol(locate_model(None))
 try:
  for m,selector in zip(models,(None,'OFF','PREFILL_12')):start(m,selector)
  with tempfile.TemporaryDirectory()as tmp:
   def equal(label,results):
    values=[read_dump_logits(m,Path(tmp)/str(i)).tobytes()for i,m in enumerate(models)]
    assert results[0]==results[1]==results[2],(label,results)
    assert values[0]==values[1]==values[2],label
    report['cases'].append(dict(label=label,exact=True,logits_sha256=hashlib.sha256(values[0]).hexdigest()))
   for count in range(1,9):
    for m in models:m.reset();m.prefill([198]*count)
    equal('prefill tail'+str(count),[m.token(40)[:2]for m in models])
   for count in range(1,5):
    for m in models:m.reset();m.prefill([198]*8)
    equal('sampled verification'+str(count),[[(i,pos)for i,pos,_ in m.block([40]*count,logits=True,uniforms=[.31]*count)]for m in models])
   for profile in ('short','long'):
    tokens=make_prompt(protocol,profile)
    for m in models:
     m.reset()
     for offset in range(0,len(tokens)-1,8):m.prefill(tokens[:-1][offset:offset+8])
    equal(profile,[m.token(tokens[-1])[:2]for m in models])
   for m,enabled in zip(models[1:],(False,True)):
    before=telemetry(m,enabled)
    assert m.command('workers 8')=='configured' and m.command('lanes 8')=='configured'
    assert telemetry(m,enabled)==before
    report['telemetry'].append(before)
 finally:
  for m in models:m.close()
 for command in ('workers 12','lanes 1'):
  m=RTLModel(binary=args.binary,workers=8,lanes=8,block_size=4,prefill_size=8)
  try:
   start(m,'PREFILL_12')
   try:m.command(command);raise AssertionError('Incompatible configuration accepted')
   except RuntimeError:pass
  finally:m.close()
 m=RTLModel(binary=args.binary,workers=8,lanes=8,block_size=4,prefill_size=8)
 try:
  try:start(m,'INVALID');raise AssertionError('Invalid selector accepted')
  except RuntimeError:pass
 finally:m.close()
 report['completed']=True;args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
