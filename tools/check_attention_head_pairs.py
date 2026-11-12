"""Exact scalar decode checks for optional GQA head-pair SIMD; no host model math."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from rtl_model import RTLModel
from evaluate_projection_quality import read_dump_logits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('build/head-pair-gates.json'))
    args=parser.parse_args();paths=[args.reference.resolve(),args.candidate.resolve()]
    models=[RTLModel(binary=p,workers=8,lanes=8,block_size=4,prefill_size=8) for p in paths]
    report=dict(completed=False,cases=[],binaries=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in paths])
    seed=[9707,11,358,525,264,17847,13,198]
    try:
        for model in models:model.start()
        with tempfile.TemporaryDirectory() as directory:
            def exact(label):
                values=[read_dump_logits(model,Path(directory)/str(i))for i,model in enumerate(models)]
                if values[0].tobytes()!=values[1].tobytes():raise AssertionError(('Changed logit bits',label))
                report['cases'].append(dict(label=label,bit_identical=True))
            for workers,prefix in [(8,n)for n in (0,1,7,8,63,452)]+[(n,17)for n in (1,2,3,4,7,17)]:
                prompt=(seed*57)[:prefix]
                for model in models:
                    model.set_workers(workers);model.reset()
                    for i in range(0,prefix,8):model.prefill(prompt[i:i+8])
                for token,uniform in zip((40,1097,264),(.031,.72,.999)):
                    outputs=[model.token(token,uniform=uniform)for model in models]
                    if outputs[0][:2]!=outputs[1][:2]:raise AssertionError('Sampled IDs differ')
                    exact(['sample',workers,prefix,token])
                for model in models:
                    model.command('truncate '+str(prefix));model.token(198)
                exact(['overwrite',workers,prefix])
            # Restart and eligibility transitions retain the original route semantics.
            for model in models:model.close();model.start();model.token(40)
            exact(['restart'])
        report['completed']=True
    finally:
        for model in models:model.close()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(len(report['cases']),'exact full-vocabulary comparisons')

if __name__=='__main__':main()
