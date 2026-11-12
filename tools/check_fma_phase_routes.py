"""Exact composition checks for FMA prefill and persistent purpose pools."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from rtl_model import RTLModel
from compare_attention_combined import read_dump_logits
from check_phase_pools_real import telemetry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    paths=[a.reference.resolve(),a.candidate.resolve()]
    hashes=[hashlib.sha256(x.read_bytes()).hexdigest() for x in paths]
    models=[RTLModel(binary=x,workers=8,lanes=8,block_size=4,prefill_size=8) for x in paths]
    models[1].phase_pool_selector='PREFILL_12'
    report=dict(completed=False,timing_claim=False,artifacts=[dict(path=str(x),sha256=h) for x,h in zip(paths,hashes)],cases=[])
    try:
        for m in models:m.start()
        caps=[json.loads(m.command('precision')) for m in models]
        for cap in caps:
            for k,v in dict(precision_profile='fp32_b1_fp32_fma_prefill',canonical_prefix_policy='none',prefill_accumulation='fused',scalar_accumulation='separate',verification_accumulation='separate').items():
                assert cap.get(k)==v,(k,cap)
        report['capabilities']=caps
        with tempfile.TemporaryDirectory() as directory:
            def equal(label,results):
                assert results[0]==results[1],(label,results)
                raw=[read_dump_logits(m,Path(directory)/str(i)).tobytes() for i,m in enumerate(models)]
                assert raw[0]==raw[1],label
                report['cases'].append(dict(label=label,ids_positions=results[0],bit_exact=True,logits_sha256=hashlib.sha256(raw[0]).hexdigest()))
            def feed(m,ids):
                for i in range(0,len(ids),8):m.prefill(ids[i:i+8])
            for length in (63,64,65,127,128,129,451):
                ids=([9707,11,358,40,1097,264,1296,198]*57)[:length]
                before=telemetry(models[1],True)['dispatch_counts']
                for m in models:m.reset();feed(m,ids)
                equal(f'prefill{length}',[m.token(198)[:2] for m in models])
                after=telemetry(models[1],True)['dispatch_counts']
                assert after[1][1]>before[1][1] and after[0][0]>before[0][0]
                assert after[0][2]==before[0][2]
                for m in models:
                    assert m.command(f'truncate {length-1}')=='truncated'
                    m.prefill([40])
                equal(f'replace{length}',[m.token(198)[:2] for m in models])
            for count in range(1,5):
                before=telemetry(models[1],True)['dispatch_counts']
                for m in models:m.reset();feed(m,[9707]*8)
                outputs=[[(i,pos) for i,pos,_ in m.block([40]*count,logits=False)] for m in models]
                assert outputs[0]==outputs[1]
                equal(f'no-logits-verification{count}',[m.token(198)[:2] for m in models])
                after=telemetry(models[1],True)['dispatch_counts']
                assert after[0][2]>before[0][2] and after[1][2]==0
            report['telemetry']=telemetry(models[1],True)
        for path,h in zip(paths,hashes):assert hashlib.sha256(path.read_bytes()).hexdigest()==h
        report['artifact_hashes_stable']=True
        report['completed']=True
    finally:
        for m in models:m.close()
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',len(report['cases']),'exact composition cases')

if __name__=='__main__':main()
