"""Paired isolated fused/separate FP32 projections; no full-model claim."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import numpy as np
from rtl_model import prepare_layout
from evaluate_projection_quality import model_provenance


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repetitions',type=int,default=1000)
    a=p.parse_args()
    if not 100<=a.repetitions<=10000:p.error('Use100..10000 repetitions')
    gate=json.loads(a.gate.read_text())
    if not gate.get('completed') or len(gate['checks'])!=36:raise ValueError('Require complete reviewed correctness gate')
    expected={(r['batch'],r['kind']):r['sha256'] for r in gate['artifacts']}
    model,layout=prepare_layout(None)
    report=dict(completed=False,workers=8,repetitions=a.repetitions,model_provenance=model_provenance(model),gate_sha256=sha(a.gate),source_sha256=sha(__file__),artifacts=[],runs=[],
        scope='Warmed isolated full Q/gate/down matrices with synthetic activations. SV input conversion and repeated eval included; initial setup/input copies precede the timer. Workers signal completion before their output copy/teardown, so earlier worker copies may overlap a still-timed straggler; output traffic is not fully isolated. One worker-launch/completion barrier around repetitions, not per model projection dispatch. No full-model speed or quality claim.',
        policy='Same reviewed compiler and strict FP flags for separate/fused; explicit FMA only changes accumulation rounding. Separate final lane sum unchanged. B8 retains unroll-count1; no global forced vector width.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for batch in (1,4,8):
        paths={k:(a.directory/k/f'fp32-b{batch}/Vfp32_projection_kernel').resolve() for k in ('separate','fused')}
        manifests={k:json.loads((v.parent/'manifest.json').read_text()) for k,v in paths.items()}
        if manifests['separate']['compiler_binary_sha256']!=manifests['fused']['compiler_binary_sha256']:raise ValueError('Different compilers')
        for k,v in paths.items():
            if sha(v)!=expected[batch,k] or manifests[k]['binary_sha256']!=expected[batch,k]:raise ValueError('Artifact differs from correctness gate')
            report['artifacts'].append(dict(batch=batch,kind=k,path=str(v),sha256=sha(v),manifest=manifests[k]))
        for name,slot,rows,width in [('q',4,2048,1024),('gate',11,3072,1024),('down',13,1024,3072)]:
            record=dict(batch=batch,name=name,slot=slot,rows=rows,width=width,pairs=[]);warm={}
            def run(kind,reps):
                dump=a.output.parent/f'{kind}.bin'
                result=subprocess.run([str(paths[kind]),str(model/'model.safetensors'),str(layout),str(slot),str(rows),str(width),'8',str(reps),str(dump)],check=True,capture_output=True,text=True)
                seconds=float(result.stdout)
                if not np.isfinite(seconds) or seconds<=0 or dump.stat().st_size!=rows*batch*8:raise ValueError('Invalid result')
                data=np.fromfile(dump,dtype=np.float64)
                if not np.isfinite(data).all():raise ValueError('Nonfinite result')
                bits=data.tobytes()
                if kind not in warm:warm[kind]=bits
                if bits!=warm[kind]:raise AssertionError('Same-rounding output changed between runs')
                return seconds,data
            for kind in paths:run(kind,1)
            for order in [('separate','fused'),('fused','separate'),('separate','fused')]:
                output={k:run(k,a.repetitions) for k in order}
                delta=output['fused'][1]-output['separate'][1]
                record['pairs'].append(dict(order=order,seconds={k:v[0] for k,v in output.items()},speedup=output['separate'][0]/output['fused'][0],max_absolute_rounding_difference=float(np.abs(delta).max())))
            ratios=[r['speedup'] for r in record['pairs']]
            record['speedup']=dict(median=statistics.median(ratios),minimum=min(ratios),maximum=max(ratios))
            report['runs'].append(record)
            a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(batch=batch,name=name,speedup=record['speedup'])),flush=True)
    for artifact in report['artifacts']:
        if sha(artifact['path'])!=artifact['sha256']:raise ValueError('Binary changed during measurement')
    report['completed']=True;a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
