"""Check that no-logits block verification still uses the decode pool."""
import argparse,json,tempfile
from pathlib import Path
from rtl_model import RTLModel
from check_phase_pools_real import start,telemetry
from compare_attention_combined import read_dump_logits
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
models=[RTLModel(binary=b,workers=8,lanes=8,block_size=4,prefill_size=8)for b in (args.reference,args.binary)]
results=[]
try:
 for m,s in zip(models,(None,'PREFILL_12')):start(m,s)
 with tempfile.TemporaryDirectory()as tmp:
  for count in range(1,5):
   for m in models:m.reset();m.prefill([198]*8)
   before=telemetry(models[1],True)
   blocks=[m.block([40]*count,logits=False)for m in models]
   after=telemetry(models[1],True)
   assert blocks[0]==blocks[1]
   assert after['dispatch_counts'][1]==before['dispatch_counts'][1]
   assert after['dispatch_counts'][0][2]>before['dispatch_counts'][0][2]
   ids=[m.token(198)[:2]for m in models]
   values=[read_dump_logits(m,Path(tmp)/str(i)).tobytes()for i,m in enumerate(models)]
   assert ids[0]==ids[1] and values[0]==values[1]
   results.append(dict(count=count,exact=True,only_verification_pool=True))
finally:
 for m in models:m.close()
args.output.write_text(json.dumps(dict(completed=True,cases=results),indent=2)+'\n')
