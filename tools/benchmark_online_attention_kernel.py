"""Explicitly requested standalone timing; requires prior full oracle gate."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from check_online_attention_kernel import ROOT,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--pairs',type=int,default=3)
    a=p.parse_args()
    if not 1<=a.iterations<=10000 or not 1<=a.pairs<=5:p.error('Invalid work budget')
    directory=ROOT/'build/online-attention-kernel';binary=directory/'obj/Vonline_attention_kernel'
    gate=json.loads((directory/'correctness.json').read_text())
    assert gate['completed'] and len(gate['cases'])==54 and sha(binary)==gate['binary_sha256']
    for name,h in gate['manifest']['sources'].items():assert sha(ROOT/name)==h
    report=dict(completed=False,binary_sha256=sha(binary),driver_sha256=sha(__file__),oracle_report_sha256=sha(directory/'correctness.json'),iterations=a.iterations,pairs=a.pairs,cases=[],limitations=gate['limitations']+['Single model thread. Each sample starts a fresh process, initializes identical arrays and performs one untimed warm evaluation; only repeated clock eval calls are timed.','Both branches use same compiled binary/O3 ThinLTO/default contraction. Branch layout and compiler vectorization differ. No host dispatch/DPI access, whole-model cache footprint or decode-throughput claim.'])
    try:
        for count in (1,2,8):
            for length in (128,512,2048,4096):
                record=dict(queries=count,length=length,pairs=[]);report['cases'].append(record)
                known={}
                for pair in range(a.pairs):
                    order=[0,1] if pair%2==0 else [1,0];times={}
                    for method in order:
                        run=subprocess.run([str(binary),str(method),str(count),str(length),'0',str(a.iterations)],capture_output=True,text=True,check=True)
                        lines=run.stdout.splitlines();seconds=[float(x.split()[1]) for x in lines if x.startswith('SECONDS ')]
                        assert len(seconds)==1 and seconds[0]>0
                        outputs='\n'.join(x for x in lines if x.startswith('RESULT '))
                        assert len(outputs.splitlines())==count*128
                        digest=hashlib.sha256(outputs.encode()).hexdigest()
                        assert method not in known or digest==known[method]
                        known[method]=digest;times[method]=seconds[0]
                    record['pairs'].append(dict(order=order,three_pass_seconds=times[0],online_seconds=times[1],speedup=times[0]/times[1]))
                record['median_speedup']=statistics.median(x['speedup'] for x in record['pairs'])
                record['output_hashes']=known
                print(json.dumps(record),flush=True)
        assert sha(binary)==report['binary_sha256']
        report['completed']=True
    finally:(directory/'timing.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
