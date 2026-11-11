"""Actual strict FP32 archive + stock default-math host/runtime thin-LTO link."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from probe_fp32_projection import build,sha
from experiment import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compiler-root',type=Path,required=True)
    p.add_argument('--directory',type=Path,default=ROOT/'build/fp32-vocab-mixed');a=p.parse_args();compiler=a.compiler_root.resolve();directory=a.directory.resolve()
    build(directory,1,'fp64',compiler,True,False)
    obj=directory/'fp32-b1';obj.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['VERILATOR_ROOT']=str(compiler);env['VERILATOR_BIN']=str(compiler/'build-local/src/verilator_bin')
    flags='-std=c++17 -O3 -flto=thin -fno-fast-math -ffp-contract=off'
    source=ROOT/'rtl/fp32_projection_kernel.sv';host=ROOT/'sim/fp32_projection_probe.cpp'
    tracked=[source,host,Path(__file__).resolve(),ROOT/'tools/probe_fp32_projection.py',ROOT/'tools/fp32_rounding_oracle.py']
    sources={str(f.relative_to(ROOT)):sha(f)for f in tracked};compiler_hash=sha(env['VERILATOR_BIN'])
    command=[str(compiler/'bin/verilator'),'--cc','--build','-j','4','--top-module','fp32_projection_kernel','--Mdir',str(obj),
        '-Wall','-Wno-BLKSEQ','-Wno-fatal','-CFLAGS',flags,'-MAKEFLAGS','OPT_FAST=-O3','-O3',
        '+define+KERNEL_BATCH=1','+define+KERNEL_ROWS=151936',str(source)]
    clean=os.environ.copy();clean.pop('VERILATOR_ROOT',None);clean.pop('VERILATOR_BIN',None)
    stock=Path(subprocess.check_output(['verilator','--getenv','VERILATOR_ROOT'],env=clean,text=True).strip())/'include'
    binary=obj/'Vfp32_projection_kernel'
    link=['c++','-std=c++17','-O3','-flto=thin','-pthread','-DKERNEL_BATCH=1','-DKERNEL_ROWS=151936','-I'+str(obj),'-I'+str(stock),'-I'+str(stock/'vltstd'),
        str(host),str(obj/'Vfp32_projection_kernel__ALL.a'),str(stock/'verilated.cpp'),str(stock/'verilated_dpi.cpp'),str(stock/'verilated_threads.cpp'),
        '-Wl,-U,__Z15vl_time_stamp64v,-U,__Z13sc_time_stampv,-U,_vlog_startup_routines','-o',str(binary)]
    with (obj/'build.log').open('w')as log:
        subprocess.run(command,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
        subprocess.run(link,env=clean,check=True,stdout=log,stderr=subprocess.STDOUT)
    if sources!={str(f.relative_to(ROOT)):sha(f)for f in tracked}or sha(env['VERILATOR_BIN'])!=compiler_hash:raise RuntimeError('Source/compiler changed during build')
    with (obj/'disassembly.txt').open('w')as stream:subprocess.run(['otool','-tvV',str(binary)],stdout=stream,check=True)
    headers=('verilated.h','verilated_types.h','verilated_threads.h')
    metadata=dict(binary=str(binary),binary_sha256=sha(binary),command=[command,link],source_hashes=sources,
        compiler_binary_sha256=compiler_hash,compiler_git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=compiler,text=True).strip(),
        compiler_patch_base='bbb60a403932a9f695cb6ff001ceb73de6ad9234',compiler_patch_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary','bbb60a403932a9f695cb6ff001ceb73de6ad9234'],cwd=compiler)).hexdigest(),
        runtime_headers={name:dict(stock=sha(stock/name),patched=sha(compiler/'include'/name))for name in headers},
        stock_runtime_root=str(stock),link_policy='Strict FP32 generated objects only; stock host/runtime and final link retain original default contraction. ThinLTO allows BF16 transport inlining.')
    (obj/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n');print(binary,flush=True)

if __name__=='__main__':main()
