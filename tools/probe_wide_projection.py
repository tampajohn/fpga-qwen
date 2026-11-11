"""Build/run isolated SV projection probes; no public model protocol changes."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
try:
    from .experiment import ROOT,locate_model
    from .rtl_model import prepare_layout
except ImportError:
    from experiment import ROOT,locate_model
    from rtl_model import prepare_layout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=int,choices=[4,8,16],required=True)
    parser.add_argument('--four-lanes',action='store_true')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--repetitions',type=int,default=100)
    parser.add_argument('--build-only',action='store_true')
    args=parser.parse_args()
    directory=ROOT/(f'build/wide-probe-b{args.batch}'+('-four-lanes' if args.four_lanes else ''))
    directory.mkdir(parents=True,exist_ok=True)
    flags=f'-std=c++17 -O3 -flto=thin -DWIDE_BATCH={args.batch}'+(' -DWIDE_FOUR_LANES' if args.four_lanes else '')
    command=['verilator','--cc','--exe','--build','-j','4','--top-module','projection_worker',
        '--Mdir',str(directory),'-Wall','-Wno-BLKSEQ','-Wno-fatal','-CFLAGS',flags,
        '-MAKEFLAGS','OPT_FAST=-O3','-LDFLAGS','-flto=thin','-O3',
        '+define+PARALLEL_ATTENTION','+define+BATCH_PREFILL','+define+BLOCK_UNROLL',
        '+define+UNCHECKED_WEIGHTS','+define+MASK_ACTIVATION_INDEX','+define+UNSIGNED_PROJECTION_INDEX',
        f'+define+WIDE_BATCH={args.batch}',*(['+define+WIDE_FOUR_LANES'] if args.four_lanes else []),str(ROOT/'rtl/projection_worker.sv'),str(ROOT/'sim/wide_projection_probe.cpp')]
    with (directory/'build.log').open('w') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    binary=directory/'Vprojection_worker'
    report=dict(batch=args.batch,candidate_reduction_lanes=4 if args.four_lanes else 8,
                absolute_tolerance=1e-10 if args.four_lanes else 0,activation_denominator=17 if args.four_lanes else 16,workers=args.workers,repetitions=args.repetitions,
                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),command=command,
                source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [ROOT/'rtl/projection_worker.sv',ROOT/'sim/wide_projection_probe.cpp']},
                limitation='Projection-only warmed arithmetic throughput, excludes per-job pool barriers, input/result copies and all nonprojection model work; synthetic activations with real checkpoint matrices.',runs=[])
    if args.build_only:
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n');return
    model,layout=prepare_layout(locate_model(None))
    for name,slot,rows,width in [('q',4,2048,1024),('o',7,1024,2048),('gate',11,3072,1024),('down',13,1024,3072),('tail',4,7,9)]:
        result=subprocess.run([str(binary),str(model/'model.safetensors'),str(layout),str(slot),str(rows),str(width),str(args.workers),str(args.repetitions)],check=True,capture_output=True,text=True)
        row=json.loads(result.stdout);row.update(name=name,rows=rows,width=width)
        row['median_throughput_ratio']=statistics.median(p['throughput_ratio'] for p in row['pairs'])
        report['runs'].append(row)
        print(json.dumps(dict(batch=args.batch,name=name,bit_identical=row['bit_identical'],max_error=row['max_absolute_error'],ratio=row['median_throughput_ratio'])),flush=True)
    (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (directory/'disassembly.txt').open('w') as assembly:
        subprocess.run(['otool','-tvV',str(binary)],stdout=assembly,check=True)


if __name__=='__main__':main()
