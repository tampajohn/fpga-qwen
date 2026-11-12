"""Exact tiny fused oracle and descriptive full-matrix projection differences."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import numpy as np
from rtl_model import prepare_layout
from fp32_rounding_oracle import round_binary32,as_float,add,fused_multiply_add


def fused_fixture(checkpoint,base,rows,width,batch):
    if batch not in (1,4,8) or rows not in (1,3,7) or width not in (8,16,24):raise ValueError('Tiny test oracle only')
    lanes=4 if batch==8 else 8
    inputs=[[round_binary32(Fraction.from_float(((j*7+t*11)%61-30)/17.0)) for t in range(batch)] for j in range(width)]
    with checkpoint.open('rb') as stream:
        stream.seek(base*2);weights=struct.unpack('<'+'H'*(rows*width),stream.read(rows*width*2))
    result=[]
    for row in range(rows):
        partial=[[0]*batch for _ in range(lanes)]
        for col in range(width):
            for t in range(batch):partial[col%lanes][t]=fused_multiply_add(inputs[col][t],weights[row*width+col]<<16,partial[col%lanes][t])
        for t in range(batch):
            total=0
            for lane in range(lanes):total=add(total,partial[lane][t])
            result.append(as_float(total))
    return np.array(result,dtype=np.float64)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    for b in (1,4,8):parser.add_argument(f'--legacy-b{b}',type=Path,required=True)
    args=parser.parse_args();model,layout=prepare_layout(None)
    offsets=[int(x) for x in layout.read_text().splitlines()]
    legacy={b:getattr(args,f'legacy_b{b}').resolve() for b in (1,4,8)}
    report=dict(completed=False,timing_claim=False,compiler_comparison='Both separate and fused use the same reviewed compiler; final lane sums unchanged',rounding_policy='Explicit one-round binary32 accumulation versus original separate-round multiply/add',artifacts=[],checks=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for batch in (1,4,8):
        paths={kind:args.directory/kind/f'fp32-b{batch}/Vfp32_projection_kernel' for kind in ('separate','fused')}
        paths['legacy']=legacy[batch]
        for kind,path in paths.items():report['artifacts'].append(dict(batch=batch,kind=kind,path=str(path.resolve()),sha256=sha(path),manifest=json.loads((path.parent/'manifest.json').read_text())))
        for workers in (1,8):
            for name,slot,rows,width in [('tiny1',4,1,8),('tiny3',4,3,16),('tiny7',4,7,24),('q',4,2048,1024),('gate',11,3072,1024),('down',13,1024,3072)]:
                arrays={}
                for kind,path in paths.items():
                    dump=args.output.parent/f'{kind}.bin'
                    subprocess.run([str(path.resolve()),str(model/'model.safetensors'),str(layout),str(slot),str(rows),str(width),str(workers),'1',str(dump)],capture_output=True,check=True)
                    if dump.stat().st_size!=rows*batch*8:raise AssertionError('Wrong output shape')
                    arrays[kind]=np.fromfile(dump,dtype=np.float64)
                    if not np.isfinite(arrays[kind]).all():raise AssertionError('Nonfinite output')
                if not np.array_equal(arrays['separate'].view(np.uint64),arrays['legacy'].view(np.uint64)):raise AssertionError('New compiler changed reference full-matrix bits')
                oracle=None
                if rows<=7:
                    oracle=fused_fixture(model/'model.safetensors',offsets[slot],rows,width,batch)
                    if not np.array_equal(arrays['fused'].view(np.uint64),oracle.view(np.uint64)):raise AssertionError('Fused tiny oracle mismatch')
                delta=arrays['fused']-arrays['separate']
                report['checks'].append(dict(batch=batch,workers=workers,name=name,rows=rows,width=width,separate_reference_bit_identical=True,fused_rational_oracle_passed=True if oracle is not None else None,max_absolute_rounding_difference=float(np.abs(delta).max()),changed_outputs=int(np.count_nonzero(arrays['fused'].view(np.uint64)!=arrays['separate'].view(np.uint64)))))
                print(json.dumps(report['checks'][-1]),flush=True)
    for item in report['artifacts']:
        path=Path(item['path'])
        if sha(path)!=item['sha256']:raise AssertionError('Artifact changed')
    report['completed']=True;args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
