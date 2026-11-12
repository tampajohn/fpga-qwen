"""Bounded SV attention correctness. No scored timing or model-quality claim."""
import argparse
from decimal import Decimal, localcontext
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@lru_cache(None)
def oracle(scenario,length,count,q):
    # Fixture K repeats every23 positions and V every29; jointly667.
    # Counting equal terms is exact for the high-precision mathematical oracle,
    # not a reproduction of either floating-point accumulation algorithm.
    with localcontext() as ctx:
        ctx.prec=80
        factor=(Decimal(1),Decimal(64),Decimal(1)/1024)[scenario]
        scores=[sum(Decimal((q*13+j*7)%19-9)/16*factor * Decimal((t*11+j*3)%23-11)/16*factor for j in range(128))/Decimal(128).sqrt() for t in range(23)]
        visible=length-count+q+1
        maximum=max(scores[t%23] for t in range(visible))
        weights=[(v-maximum).exp() for v in scores]
        multiplicities=[(visible+666-r)//667 if r<visible else 0 for r in range(667)]
        denominator=sum(weights[r%23]*n for r,n in enumerate(multiplicities))
        return tuple(sum(weights[r%23]*n*Decimal((r*5+j*17)%29-14)/8 for r,n in enumerate(multiplicities) if n)/denominator for j in range(128))

def run(binary,method,count,length,scenario):
    run=subprocess.run([str(binary),str(method),str(count),str(length),str(scenario),'1'],capture_output=True,text=True,check=True)
    result={}
    for line in run.stdout.splitlines():
        if not line.startswith('RESULT '):continue
        fields=line.split()
        assert len(fields)==4,line
        q,j=int(fields[1]),int(fields[2]);value=float(fields[3])
        assert 0<=q<count and 0<=j<128 and (q,j) not in result and math.isfinite(value),line
        result[q,j]=value
    assert len(result)==count*128
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-build',action='store_true');p.add_argument('--quick',action='store_true')
    a=p.parse_args();directory=ROOT/'build/online-attention-kernel';directory.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/'rtl/online_attention_kernel.sv',ROOT/'sim/experiments/online_attention_kernel.cpp']
    hashes={str(x.relative_to(ROOT)):sha(x) for x in sources}
    compiler=Path(shutil.which('verilator')).resolve()
    command=[str(compiler),'--cc','--exe','--build','--top-module','online_attention_kernel','--Mdir',str(directory/'obj'),'-j','2','-O3','-Wno-fatal','-CFLAGS','-O3 -flto=thin','-LDFLAGS','-flto=thin','-MAKEFLAGS','OPT_FAST=-O3 OPT_SLOW=-O3',str(sources[0]),str(sources[1])]
    manifest_path=directory/'manifest.json'
    if not a.skip_build:
        with (directory/'build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest_path.write_text(json.dumps(dict(sources=hashes,compiler=str(compiler),compiler_sha256=sha(compiler),command=command),indent=2)+'\n')
    manifest=json.loads(manifest_path.read_text());assert manifest['sources']==hashes
    binary=directory/'obj/Vonline_attention_kernel';initial=sha(binary)
    report=dict(completed=False,binary_sha256=initial,manifest=manifest,source_driver_sha256=sha(__file__),timing_claim=False,model_quality_claim=False,fixture_budget_absolute=1e-12,oracle_digits=80,cases=[],limitations=['Single head and single module, contiguous KV; excludes deployed DPI/pool/GQA addressing costs.','Q1 is a standalone query, not deployed op6 paired-head decode.','Online tiling changes rounding; cross-method bit equality is not expected.','Only bounded deterministic periodic finite fixtures; no model-quality claim.'])
    try:
        for scenario in range(3):
            for count in (1,2,8):
                for length in ((129,) if a.quick else (128,129,512,513,2048,4096)):
                    outputs=[run(binary,method,count,length,scenario) for method in (0,1)]
                    errors=[0.,0.]
                    with localcontext() as ctx:
                        ctx.prec=80
                        for q in range(count):
                            expected=oracle(scenario,length,count,q)
                            for j,target in enumerate(expected):
                                for method in (0,1):errors[method]=max(errors[method],float(abs(Decimal.from_float(outputs[method][q,j])-target)))
                    assert max(errors)<=1e-12,(scenario,count,length,errors)
                    report['cases'].append(dict(scenario=scenario,queries=count,length=length,max_abs_error=errors,max_cross_method_error=max(abs(outputs[0][key]-outputs[1][key]) for key in outputs[0]),full_outputs_checked=count*128))
                    print(json.dumps(report['cases'][-1]),flush=True)
        assert sha(binary)==initial and {str(x.relative_to(ROOT)):sha(x) for x in sources}==hashes
        report['completed']=True
    finally:(directory/'correctness.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
