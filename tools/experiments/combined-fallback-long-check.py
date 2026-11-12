import sys,json,tempfile
from pathlib import Path
sys.path.insert(0,str(Path('tools').resolve()))
from rtl_model import RTLModel
from evaluate_projection_quality import read_dump_logits
candidate=next(Path('build/variants').glob('*qk2-head-pairs-value-pairs/full/Vqwen_behavioral')).resolve()
reference=next(Path('../fpga-qwen/build/variants').glob('*fp32-b1-verify-b4-batch-attention/full/Vqwen_behavioral')).resolve()
models=[RTLModel(binary=p,workers=8,lanes=8,block_size=4,prefill_size=8)for p in (candidate,reference)]
cases=[];seed=[9707,11,358,525,264,17847,13,198];prefix=(seed*57)[:451]
try:
 for m in models:
  m.start()
  for i in range(0,len(prefix),8):m.prefill(prefix[i:i+8])
 with tempfile.TemporaryDirectory()as tmp:
  for workers in (3,7,17):
   for m in models:m.set_workers(workers);m.command('truncate 451')
   for token in (40,198):
    outputs=[m.token(token) for m in models]
    values=[read_dump_logits(m,Path(tmp)/str(i))for i,m in enumerate(models)]
    cases.append(dict(workers=workers,token=token,position=outputs[0][1],ids_positions_equal=outputs[0][:2]==outputs[1][:2],exact=values[0].tobytes()==values[1].tobytes(),max_error=float(abs(values[0]-values[1]).max()),different_values=int((values[0].view('uint64')!=values[1].view('uint64')).sum())))
 Path('build/fallback-long-gates.json').write_text(json.dumps(cases,indent=2)+'\n');print(cases)
finally:
 for m in models:m.close()
