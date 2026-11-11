"""Three-way isolated B8 projection comparison; no full-model speed claim."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import numpy as np
from rtl_model import prepare_layout
from evaluate_projection_quality import model_provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fp64','original','candidate'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repetitions',type=int,default=1000)
    a=p.parse_args()
    if not 1<=a.repetitions<=10000:p.error('Invalid repetitions')
    model,layout=prepare_layout(None)
    paths={k:getattr(a,k).resolve()for k in ('fp64','original','candidate')}
    report=dict(completed=False,model_provenance=model_provenance(model),workers=8,batch=8,repetitions=a.repetitions,
        binaries={k:dict(path=str(v),sha256=hashlib.sha256(v.read_bytes()).hexdigest(),manifest=json.loads((v.parent/'manifest.json').read_text()))for k,v in paths.items()},
        limitation='Warmed projection-only synthetic activation fixtures with real checkpoint matrices; fixed eight workers, no model/prefill latency or quality claim. Three alternating orders, OS not isolated.',runs=[])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for name,slot,rows,width in [('q',4,2048,1024),('gate',11,3072,1024),('down',13,1024,3072)]:
        record=dict(name=name,slot=slot,rows=rows,width=width,rounds=[]);gold={}
        for order in [('fp64','original','candidate'),('candidate','original','fp64'),('fp64','original','candidate')]:
            arrays={};times={}
            for kind in order:
                dump=a.output.parent/f'{kind}.bin'
                run=subprocess.run([str(paths[kind]),str(model/'model.safetensors'),str(layout),str(slot),str(rows),str(width),'8',str(a.repetitions),str(dump)],check=True,capture_output=True,text=True)
                times[kind]=float(run.stdout)
                if not np.isfinite(times[kind])or times[kind]<=0:raise ValueError('Invalid timing')
                if dump.stat().st_size!=rows*8*8:raise ValueError('Invalid dump size')
                arrays[kind]=np.fromfile(dump,dtype=np.float64)
                if not np.isfinite(arrays[kind]).all():raise ValueError('Nonfinite result')
                if kind not in gold:gold[kind]=arrays[kind].copy()
                if not np.array_equal(gold[kind].view(np.uint64),arrays[kind].view(np.uint64)):raise AssertionError('Nondeterministic output')
            if not np.array_equal(arrays['original'].view(np.uint64),arrays['candidate'].view(np.uint64)):raise AssertionError('Changed FP32 arithmetic results')
            record['rounds'].append(dict(order=order,seconds=times,candidate_over_fp64=times['fp64']/times['candidate'],candidate_over_original=times['original']/times['candidate'],exact_fp32_outputs=True,max_absolute_error_vs_fp64=float(np.max(np.abs(arrays['candidate']-arrays['fp64'])))))
        for key in ('candidate_over_fp64','candidate_over_original'):
            values=[r[key]for r in record['rounds']];record[key]=dict(median=statistics.median(values),minimum=min(values),maximum=max(values))
        report['runs'].append(record);print(json.dumps(record),flush=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    report['completed']=True;a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
