"""Build the isolated finite FP32 storage fixture with the reviewed compiler fork."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root',type=Path,required=True)
    args=parser.parse_args();compiler=args.compiler_root.resolve()
    output=ROOT/'build/fp32-kv';output.mkdir(parents=True,exist_ok=True)
    sources=['rtl/fp32_kv_fixture.sv','sim/fp32_kv_fixture.cpp','tools/build_fp32_kv_fixture.py','tools/check_fp32_kv_fixture.py','tools/fp32_rounding_oracle.py']
    before={p:sha(ROOT/p)for p in sources}
    env=os.environ.copy();env.update(VERILATOR_ROOT=str(compiler),VERILATOR_BIN=str(compiler/'build-local/src/verilator_bin'))
    command=[str(compiler/'bin/verilator'),'--cc','--exe','--build','-j','4','--top-module','fp32_kv_fixture','--Mdir',str(output),'-Wall','-Wno-fatal','-Wno-BLKSEQ','-CFLAGS','-std=c++17 -O3 -fno-fast-math -ffp-contract=off','-MAKEFLAGS','OPT_FAST=-O3',str(ROOT/sources[0]),str(ROOT/sources[1])]
    subprocess.run(command,env=env,check=True)
    assert before=={p:sha(ROOT/p)for p in sources}
    binary=output/'Vfp32_kv_fixture'
    result=subprocess.run([sys.executable,str(ROOT/'tools/check_fp32_kv_fixture.py'),str(binary)],check=True,text=True,capture_output=True)
    report=json.loads(result.stdout)
    report.update(binary_sha256=sha(binary),compiler_binary_sha256=sha(env['VERILATOR_BIN']),compiler_commit=subprocess.check_output(['git','-C',str(compiler),'rev-parse','HEAD'],text=True).strip(),source_hashes=before,command=command,generated_header_sha256=sha(output/'Vfp32_kv_fixture___024root.h'))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
