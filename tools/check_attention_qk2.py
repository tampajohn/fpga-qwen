"""No host neural arithmetic: teacher-forced SV comparison across causal routes."""
import argparse,tempfile,hashlib,json
from pathlib import Path
from rtl_model import RTLModel
from evaluate_projection_quality import read_dump_logits
parser=argparse.ArgumentParser(description="Exact causal/partition gates for QK2 versus ordinary batched attention")
parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True)
parser.add_argument('--output',type=Path,default=Path('build/qk2-gates.json'))
args=parser.parse_args();candidate=args.candidate.resolve();reference=args.reference.resolve()
models=[RTLModel(binary=p,workers=8,lanes=8,block_size=4,prefill_size=8)for p in (candidate,reference)]
seed=[9707,11,358,525,264,17847,13,198];cases=[]
try:
 for m in models:m.start()
 with tempfile.TemporaryDirectory()as tmp:
  def exact(label):
   values=[read_dump_logits(m,Path(tmp)/str(i))for i,m in enumerate(models)]
   assert values[0].tobytes()==values[1].tobytes(),(label,float(abs(values[0]-values[1]).max()))
   cases.append(label)
  for prefix,count in [(0,n)for n in range(1,9)]+[(n,7)for n in (1,7,8,63,452)]:
   prompt=(seed*57)[:prefix]
   for m in models:
    m.reset()
    for i in range(0,prefix,8):m.prefill(prompt[i:i+8])
    m.prefill(seed[:count]);m.token(40)
   exact(['prefill',prefix,count])
   for m in models:m.command('truncate '+str(prefix));m.prefill(list(reversed(seed[:count])));m.token(198)
   exact(['overwrite',prefix,count])
  for workers in (1,3,17):
   for count in (1,2,3,4):
    outputs=[]
    for m in models:
     m.set_workers(workers);m.reset();m.prefill(seed)
     outputs.append(m.block([40,1097,264,1296][:count],logits=True,uniforms=[0.031,0.72,0.999,0.43][:count]))
    assert [x[:2]for x in outputs[0]]==[x[:2]for x in outputs[1]]
    exact(['sampled',workers,count])
 report=dict(completed=True,exact=True,cases=cases,binaries=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in (candidate,reference)])
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(report,indent=2)+'\n');print(len(cases),'exact full-vocabulary comparisons')
finally:
 for m in models:m.close()
