"""Verify bit-for-bit preservation of complete repeated full-logit trajectories."""
import argparse
import json
from pathlib import Path
from benchmark_native_phase0 import guard, sha
from experiment import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,nargs='+',required=True)
    parser.add_argument('--candidate',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    guard(ROOT/'build/benchmark-status')
    if len(args.reference)!=len(args.candidate) or args.output.exists():
        parser.error('Supply paired captures and a new output path')
    report=dict(completed=False,passed=False,scope='Bit-identical full logits and predicted/consumed IDs on all supplied repeated trajectories only.',source_sha256=sha(Path(__file__)),inputs={},arrays=[])
    def verify(path,digest=None):
        key=str(Path(path).resolve())
        if key not in report['inputs']:report['inputs'][key]=sha(Path(key))
        if digest is not None and report['inputs'][key]!=digest:raise RuntimeError('Artifact changed: '+key)
        return report['inputs'][key]
    try:
        for left,right in zip(args.reference,args.candidate):
            verify(left);verify(right)
            a,b=[json.loads(p.read_text()) for p in (left,right)]
            for r in (a,b):
                if not r['completed'] or r['gpu'] or r['pytorch']:raise RuntimeError('Incomplete/non-CPU capture')
                for path,digest in r['inputs'].items():verify(path,digest)
            for key in ('prompt_ids','continuation_ids','requested_steps','mode','configuration'):
                if a[key]!=b[key]:raise RuntimeError('Different trajectory/configuration: '+key)
            if len(a['runs'])!=len(b['runs']) or len(a['runs'])<2:raise RuntimeError('Missing repeated capture')
            for ar,br in zip(a['runs'],b['runs']):
                for key in ('index','warmup','predicted_token_ids','consumed_continuation_ids','eog_prediction_steps'):
                    if ar[key]!=br[key]:raise RuntimeError('Different run: '+key)
                if not all(r[k] for r in (ar,br) for k in ('fixed_work_valid','own_repeat_ids_equal','own_repeat_logits_bit_identical')):
                    raise RuntimeError('Capture repeat validation failed')
                if len(ar['logit_dumps'])!=a['requested_steps'] or len(br['logit_dumps'])!=b['requested_steps']:
                    raise RuntimeError('Missing full-vocabulary dump')
                for ad,bd in zip(ar['logit_dumps'],br['logit_dumps']):
                    for key in ('step','dtype','byte_order','shape','bytes','context_tokens'):
                        if ad[key]!=bd[key]:raise RuntimeError('Different dump metadata: '+key)
                    if ad['shape']!=[151936] or ad['bytes']!=607744 or ad['dtype']!='float32' or ad['byte_order']!='little':
                        raise RuntimeError('Unexpected dump representation')
                    if any(Path(d['path']).stat().st_size!=607744 for d in (ad,bd)):
                        raise RuntimeError('Unexpected dump file length')
                    ah=verify(ad['path'],ad['sha256']);bh=verify(bd['path'],bd['sha256'])
                    equal=ah==bh
                    report['arrays'].append(dict(reference=ad['path'],candidate=bd['path'],context_tokens=ad['context_tokens'],bit_identical=equal,reference_sha256=ah,candidate_sha256=bh))
                    if not equal:raise RuntimeError('Full-logit bytes differ')
        report.update(completed=True,passed=True,compared_arrays=len(report['arrays']))
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error));raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',report['compared_arrays'],'full-vocabulary arrays')


if __name__=='__main__':main()
