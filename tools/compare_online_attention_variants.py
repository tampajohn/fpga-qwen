"""Compare two frozen standalone attention artifacts; not model throughput."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from check_online_attention_kernel import sha

FROZEN=('ef8b364d9ed4f531d833a811e5915600e68bb8d06289ffe9b2a2dfb228e6a2e1',
        '1e3ac32c2ad28ce2b15b0873b5a31a79a3ead0bf04e22aa52ccc8f3a7ca9763f')

ORACLE_HASHES=('2132a298bad1a9a19c018ebf98611395e8d669005cb8ede589a025ff31302065',
               'f0286ab88307886d21c96881f499fa48dd78a302e0a194954af77ef05da2b4cc')

def sample(binary,method,count,length,iterations):
    process=subprocess.run([str(binary),str(method),str(count),str(length),'0',str(iterations)],capture_output=True,text=True,check=True)
    seconds=[];output={}
    for line in process.stdout.splitlines():
        fields=line.split()
        if not fields:continue
        if fields[0]=='SECONDS':
            assert len(fields)==2
            seconds.append(float(fields[1]))
        elif fields[0]=='RESULT':
            assert len(fields)==4
            key=tuple(map(int,fields[1:3]));value=float(fields[3])
            assert key not in output and 0<=key[0]<count and 0<=key[1]<128 and math.isfinite(value)
            output[key]=fields[3]
    assert len(output)==count*128 and len(seconds)==1 and math.isfinite(seconds[0]) and seconds[0]>0
    digest=hashlib.sha256(json.dumps(sorted(output.items())).encode()).hexdigest()
    return dict(seconds=seconds[0],iterations=iterations,seconds_per_eval=seconds[0]/iterations,output_sha256=digest,output_elements=len(output))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True);p.add_argument('--paired',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pairs',type=int,default=3)
    p.add_argument('--minimum-seconds',type=float,default=.1)
    a=p.parse_args()
    if a.pairs!=3 or not .1<=a.minimum_seconds<=1:p.error('Reviewed budget:3pairs and0.1..1second intervals')
    paths=[a.original.resolve(),a.paired.resolve()];gates=[];provenance=[]
    for index,path in enumerate(paths):
        assert sha(path)==FROZEN[index]
        gate_path=path.parent.parent/'correctness.json'
        assert sha(gate_path)==ORACLE_HASHES[index]
        gate=json.loads(gate_path.read_text())
        assert gate['completed'] and len(gate['cases'])==54 and gate['binary_sha256']==FROZEN[index]
        assert all(max(case['max_abs_error'])<=gate['fixture_budget_absolute'] for case in gate['cases'])
        gates.append(gate)
        provenance.append(dict(path=str(path),sha256=FROZEN[index],oracle_report_sha256=sha(gate_path),compiled_source_hashes=gate['manifest']['sources'],build_command=gate['manifest']['command']))
    # Source hashes come from frozen build reports. The original compiled source
    # is intentionally preserved at4ba1f67, not the modified current worktree.
    report=dict(completed=False,artifacts=provenance,driver_sha256=sha(__file__),timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pairs=a.pairs,minimum_seconds=a.minimum_seconds,cases=[],limitations=['One head, single module/thread and contiguous Q/K/V arrays; no real model DPI, scheduling, GQA or cache-footprint equivalence.','Q1 is a standalone query, not deployed paired-head scalar decode.','Each sample starts a new process and performs one warm evaluation; only repeated clock eval calls are timed.','Fixed deterministic normal-magnitude fixture; no whole-model performance or quality claim.','Four configurations share an iteration count per shape, calibrated to at least100ms for the fastest configuration when feasible. Calibration is recorded separately and is not a result.','Original and paired binaries have separate baseline code layouts; report each baseline and do not attribute all change solely to online recurrence.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        configurations=[(0,0),(0,1),(1,0),(1,1)]
        for count in (1,2,8):
            for length in (128,512,2048,4096):
                record=dict(queries=count,length=length,calibration=[],pairs=[]);report['cases'].append(record)
                known={};iterations=10
                for attempt in range(4):
                    measured=[]
                    for b,m in configurations:
                        row=sample(paths[b],m,count,length,iterations);key=f'{b}:{m}'
                        assert key not in known or known[key]==row['output_sha256']
                        known[key]=row['output_sha256'];measured.append(row['seconds'])
                        record['calibration'].append(dict(binary=b,method=m,**row))
                    if min(measured)>=a.minimum_seconds or iterations==100000:break
                    iterations=min(100000,max(iterations+1,math.ceil(iterations*a.minimum_seconds/min(measured)*1.15)))
                record['iterations']=iterations
                for pair in range(a.pairs):
                    order=configurations if pair%2==0 else list(reversed(configurations))
                    rows={}
                    for b,m in order:
                        row=sample(paths[b],m,count,length,iterations);key=f'{b}:{m}'
                        assert row['output_sha256']==known[key]
                        row['interval_target_met']=row['seconds']>=a.minimum_seconds
                        rows[key]=row
                    ratios=dict(original_online_speedup=rows['0:0']['seconds']/rows['0:1']['seconds'],paired_online_speedup=rows['1:0']['seconds']/rows['1:1']['seconds'],paired_vs_original_online=rows['0:1']['seconds']/rows['1:1']['seconds'],paired_vs_original_baseline=rows['0:0']['seconds']/rows['1:0']['seconds'])
                    record['pairs'].append(dict(order=order,measurements=rows,ratios=ratios))
                    a.output.write_text(json.dumps(report,indent=2)+'\n')
                record['median_ratios']={k:statistics.median(row['ratios'][k]for row in record['pairs'])for k in ratios}
                print(json.dumps(dict(queries=count,length=length,median_ratios=record['median_ratios'])),flush=True)
        for path,h in zip(paths,FROZEN):assert sha(path)==h
        report['completed']=True
    finally:a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
