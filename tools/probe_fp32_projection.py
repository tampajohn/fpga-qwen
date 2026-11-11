"""Independent strict-FP32 SV microkernel versus untouched production FP64 worker.

Build and scoring are separate phases. No host neural kernel or generated-code edits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import numpy as np
from experiment import ROOT,locate_model,read_header
from fp32_rounding_oracle import projection_fixture
from rtl_model import prepare_layout
from evaluate_projection_quality import model_provenance


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def build(directory,batch,kind,compiler_root):
    destination=directory/f'{kind}-b{batch}';destination.mkdir(parents=True,exist_ok=True)
    baseline=kind=='fp64'
    env=os.environ.copy()
    for key in ('VERILATOR_ROOT','VERILATOR_BIN'):env.pop(key,None)
    if baseline:compiler=shutil.which('verilator')
    else:
        compiler=str(compiler_root/'bin/verilator')
        env['VERILATOR_ROOT']=str(compiler_root)
        env['VERILATOR_BIN']=str(compiler_root/'build-local/src/verilator_bin')
    top='projection_worker' if baseline else 'fp32_projection_kernel'
    flags=f'-std=c++17 -O3 -flto=thin -DKERNEL_BATCH={batch}'
    defines=[]
    if baseline:
        flags+=' -DBASELINE_FP64'
        defines=['PARALLEL_ATTENTION','BATCH_PREFILL','BLOCK_UNROLL','UNCHECKED_WEIGHTS','MASK_ACTIVATION_INDEX','UNSIGNED_PROJECTION_INDEX','WIDE_BATCH=8','WIDE_FOUR_LANES']
    else:
        flags+=' -fno-fast-math -ffp-contract=off'
        defines=[f'KERNEL_BATCH={batch}']
    source=ROOT/'rtl'/f'{top}.sv'
    command=[compiler,'--cc','--exe','--build','-j','4','--top-module',top,'--Mdir',str(destination),
             '-Wall','-Wno-BLKSEQ','-Wno-fatal','-CFLAGS',flags,'-MAKEFLAGS','OPT_FAST=-O3',
             '-LDFLAGS','-flto=thin','-O3',*['+define+'+v for v in defines],str(source),str(ROOT/'sim/fp32_projection_probe.cpp')]
    source_files=(source,ROOT/'sim/fp32_projection_probe.cpp',Path(__file__).resolve(),ROOT/'tools/fp32_rounding_oracle.py')
    before={str(p.relative_to(ROOT)):sha(p)for p in source_files}
    compiler_binary=(Path(compiler).resolve().parent/'verilator_bin') if baseline else compiler_root/'build-local/src/verilator_bin'
    compiler_hash=sha(compiler_binary)
    compiler_patch=b'' if baseline else subprocess.check_output(['git','diff','--binary','bbb60a403932a9f695cb6ff001ceb73de6ad9234'],cwd=compiler_root)
    with (destination/'build.log').open('w') as log:subprocess.run(command,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
    if before!={str(p.relative_to(ROOT)):sha(p)for p in source_files} or compiler_hash!=sha(compiler_binary):raise RuntimeError('Source/compiler changed during build')
    if not baseline and compiler_patch!=subprocess.check_output(['git','diff','--binary','bbb60a403932a9f695cb6ff001ceb73de6ad9234'],cwd=compiler_root):raise RuntimeError('Compiler source changed during build')
    binary=destination/f'V{top}'
    with (destination/'disassembly.txt').open('w') as assembly:subprocess.run(['otool','-tvV',str(binary)],check=True,stdout=assembly)
    metadata=dict(binary=str(binary),binary_sha256=sha(binary),command=command,
        version=subprocess.check_output([compiler,'--version'],env=env,text=True).strip(),
        compiler_binary_sha256=compiler_hash,
        compiler_patch_base=None if baseline else 'bbb60a403932a9f695cb6ff001ceb73de6ad9234',
        compiler_git_head=None if baseline else subprocess.check_output(['git','rev-parse','HEAD'],cwd=compiler_root,text=True).strip(),
        compiler_patch_sha256=None if baseline else hashlib.sha256(compiler_patch).hexdigest(),
        environment={k:env[k] for k in ('VERILATOR_ROOT','VERILATOR_BIN')if k in env},
        source_hashes=before)
    (destination/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root',type=Path)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--build-only',action='store_true');mode.add_argument('--run-only',action='store_true')
    parser.add_argument('--batches',default='1,4,8');parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--repetitions',type=int,default=20);parser.add_argument('--pairs',type=int,default=3)
    parser.add_argument('--directory',type=Path,default=ROOT/'build/fp32-projection')
    args=parser.parse_args();batches=[int(b)for b in args.batches.split(',')]
    if any(b not in (1,4,8)for b in batches) or not 1<=args.workers<=32 or not 1<=args.repetitions<=10000 or not 1<=args.pairs<=10:parser.error('Invalid workload')
    if args.build_only:
        if not args.compiler_root:parser.error('Explicit reviewed compiler root required')
        for batch in batches:
            for kind in ('fp64','fp32'):build(args.directory,batch,kind,args.compiler_root.resolve())
        return
    model,layout=prepare_layout(locate_model(None))
    header,start,_=read_header(model/'model.safetensors')
    offsets=[int(line)for line in layout.read_text().splitlines()]
    for name,slot,shape in [('self_attn.q_proj',4,[2048,1024]),('mlp.gate_proj',11,[3072,1024]),('mlp.down_proj',13,[1024,3072])]:
        tensor=header['model.layers.0.'+name+'.weight']
        if tensor['shape']!=shape or offsets[slot]!=(start+tensor['data_offsets'][0])//2:raise ValueError('Projection slot/shape mapping mismatch')
    report=dict(gpu=False,pytorch=False,completed=False,workers=args.workers,repetitions=args.repetitions,pairs=args.pairs,
        baseline='Production FP64 worker independently compiled with installed Verilator/O3/thin-LTO and original contraction behavior.',
        candidate='Original SV shortreal kernel, real ports/BF16 DPI, explicit FP32 conversions/add/mul; strict no contraction.',
        limitations='Synthetic deterministic activations with real full checkpoint matrices, warmed projection-only throughput. Excludes model layers, per-job dispatch barriers, input copying and output copies. FP32 conversion occurs inside each candidate SV tick. Candidate output capacity3072*B is smaller than production worker result151936 plus batch ports, potentially changing eval/cache costs. Vocabulary projection is not covered. Width must be8..3072 and divisible by8; full Q/gate/down tensor slots/shapes checked. Different compiler versions/layout/codegen and strict rounding mean this is not a pure arithmetic-width ablation; no full-model quality or speed claim.',
        model_provenance=model_provenance(model),runs=[])
    for batch in batches:
        manifests={kind:json.loads((args.directory/f'{kind}-b{batch}'/'manifest.json').read_text())for kind in ('fp64','fp32')}
        for manifest in manifests.values():
            if sha(manifest['binary'])!=manifest['binary_sha256']:raise RuntimeError('Binary changed since build')
            for source,digest in manifest['source_hashes'].items():
                if sha(ROOT/source)!=digest:raise RuntimeError('Source changed since build: '+source)
        fixture_checks=[]
        candidate=manifests['fp32']['binary']
        for rows,width in ((1,8),(3,16),(7,24)):
            path=args.directory/'oracle-output.bin'
            subprocess.run([candidate,str(model/'model.safetensors'),str(layout),'4',str(rows),str(width),str(args.workers),'1',str(path)],check=True,capture_output=True,text=True)
            actual=np.fromfile(path,dtype=np.float64)
            expected=np.asarray(projection_fixture(model/'model.safetensors',offsets[4],rows,width,batch),dtype=np.float64)
            if actual.shape!=expected.shape or not np.array_equal(actual.view(np.uint64),expected.view(np.uint64)):raise AssertionError('Exact-rational FP32 fixture mismatch')
            fixture_checks.append(dict(rows=rows,width=width,batch=batch,exact_rational_rounding_match=True))
        for rows,width in ((0,8),(3073,8),(1,7),(1,9),(1,3080)):
            run=subprocess.run([candidate,str(model/'model.safetensors'),str(layout),'4',str(rows),str(width),str(args.workers),'1',str(args.directory/'invalid.bin')],capture_output=True,text=True)
            if run.returncode!=1 or 'Invalid probe shape' not in run.stderr:raise AssertionError('Invalid shape accepted')
        report.setdefault('tiny_fixture_checks',[]).extend(fixture_checks)
        for name,slot,rows,width in [('q',4,2048,1024),('gate',11,3072,1024),('down',13,1024,3072)]:
            record=dict(batch=batch,name=name,slot=slot,rows=rows,width=width,reduction_lanes=4 if batch==8 else 8,manifests=manifests,pairs=[])
            gold=None
            for pair in range(args.pairs):
                durations={};arrays={}
                for kind in (('fp64','fp32')if pair%2==0 else('fp32','fp64')):
                    path=args.directory/f'{kind}-output.bin'
                    run=subprocess.run([manifests[kind]['binary'],str(model/'model.safetensors'),str(layout),str(slot),str(rows),str(width),str(args.workers),str(args.repetitions),str(path)],check=True,capture_output=True,text=True)
                    durations[kind]=float(run.stdout)
                    if not np.isfinite(durations[kind]) or durations[kind]<=0:raise ValueError('Invalid timing')
                    if path.stat().st_size!=rows*batch*8:raise ValueError('Wrong projection dump length')
                    arrays[kind]=np.fromfile(path,dtype=np.float64)
                    if not np.isfinite(arrays[kind]).all():raise ValueError('Nonfinite projection output')
                if gold is None:gold={k:v.copy()for k,v in arrays.items()}
                if any(not np.array_equal(v.view(np.uint64),gold[k].view(np.uint64))for k,v in arrays.items()):raise AssertionError('Repeated projection changed bits')
                delta=arrays['fp32']-arrays['fp64'];denom=float(np.linalg.norm(arrays['fp64']))
                record['pairs'].append(dict(order=['fp64','fp32']if pair%2==0 else['fp32','fp64'],seconds=durations,
                    throughput_ratio=durations['fp64']/durations['fp32'],max_absolute_error=float(np.max(np.abs(delta))),
                    rms_error=float(np.sqrt(np.mean(delta*delta))),relative_l2_error=float(np.linalg.norm(delta))/denom if denom else None))
            record['median_throughput_ratio']=statistics.median(p['throughput_ratio']for p in record['pairs']);report['runs'].append(record)
            print(json.dumps(dict(batch=batch,matrix=name,ratio=record['median_throughput_ratio'],error=record['pairs'][0]['max_absolute_error'])),flush=True)
            (args.directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['completed']=True;(args.directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
