"""Exact scalar/verification preservation from identical non-prefill KV states."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from rtl_model import RTLModel
from evaluate_projection_quality import read_dump_logits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    paths=[a.reference.resolve(),a.candidate.resolve()]
    models=[RTLModel(binary=x,workers=8,lanes=8,block_size=4,prefill_size=8) for x in paths]
    report=dict(completed=False,timing_claim=False,initial_state='reset; optional shared prefix fed ONLY by separate-rounded scalar operations, never prefill',artifacts=[dict(path=str(x),sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in paths],cases=[])
    try:
        for m in models:m.start()
        caps=[json.loads(m.command('precision')) for m in models]
        if [x['precision_profile'] for x in caps]!=['fp32_b1_fp32_prefill','fp32_b1_fp32_fma_prefill']:raise AssertionError('Wrong profiles')
        for key in ('scalar_accumulation','verification_accumulation','vocabulary_accumulation'):
            if caps[1][key]!='separate':raise AssertionError('Wrong preserved arithmetic')
        report['capabilities']=caps
        with tempfile.TemporaryDirectory() as directory:
            def equal(label,outputs):
                if outputs[0]!=outputs[1]:raise AssertionError((label,'IDs/positions'))
                arrays=[read_dump_logits(m,Path(directory)/str(i)).tobytes() for i,m in enumerate(models)]
                if arrays[0]!=arrays[1]:raise AssertionError((label,'full logits'))
                report['cases'].append(dict(label=label,outputs=outputs[0],full_logit_sha256=hashlib.sha256(arrays[0]).hexdigest(),bit_exact=True))
            for workers in (1,3,8,17):
                for m in models:m.set_workers(workers);m.reset()
                for t in (9707,11,358):equal(f'workers{workers}-scalar-{t}',[m.token(t)[:2] for m in models])
                for count in (1,2,3,4):
                    for sampled in (False,True):
                        for m in models:
                            m.reset()
                            for token in (9707,11):m.token(token,logits=False)
                        tokens=[40,1097,264,1296][:count]
                        uniforms=[.031,.72,.999,.43][:count] if sampled else None
                        outputs=[[(v[0],v[1]) for v in m.block(tokens,logits=True,uniforms=uniforms)] for m in models]
                        equal(f'workers{workers}-verification{count}-sampled{sampled}',outputs)
                        equal(f'workers{workers}-continuation{count}-sampled{sampled}',[m.token(198)[:2] for m in models])
        for path,item in zip(paths,report['artifacts']):
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise AssertionError('Artifact changed')
        report['completed']=True
    finally:
        for m in models:m.close()
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',len(report['cases']),'exact scalar/verification/continuation comparisons')

if __name__=='__main__':main()
