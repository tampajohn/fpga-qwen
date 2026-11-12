"""Explicit no-cache FP32 KV storage policy wrapper for the held-out scorer."""
import hashlib
import json
from pathlib import Path
import sys
from tools.rtl_model import RTLModel
from tools import evaluate_projection_quality


def main():
    args=sys.argv[1:]
    def argument(name):return args[args.index(name)+1]
    expected=[('baseline','fp32_b1_fp64_prefill','fp64'),('candidate','fp32_b1_fp64_prefill_fp32_kv','fp64')]
    precision=[]
    for role,profile,prefill in expected:
        binary=Path(argument('--'+role)).resolve()
        model=RTLModel(binary=binary,workers=8,lanes=8,block_size=4,prefill_size=8)
        try:
            model.start();value=json.loads(model.command('precision'))
            if value['precision_profile']!=profile or value['prefill_projection']!=prefill:
                raise RuntimeError('Unexpected '+role+' precision capability')
            if role=='baseline' and value.get('attention_kv')!='fp64':
                raise RuntimeError('Baseline requires FP64 attention and KV')
            if role=='candidate' and (value.get('attention_arithmetic')!='fp64' or value.get('kv_storage')!='fp32'):
                raise RuntimeError('Candidate requires FP64 arithmetic and FP32 KV storage')
            if value.get('scalar_projection')!='fp32' or value.get('verification_projection')!='fp32' or value.get('max_verification_block')!=4:
                raise RuntimeError('Unexpected scalar/verification projection route')
            if role=='candidate' and (value.get('canonical_prefix_policy')!='none' or value['prefix_reuse_supported']):
                raise RuntimeError('FP32 KV evaluation requires no prefix cache policy')
            precision.append(dict(role=role,sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),capability=value))
        finally:model.close()
    for name in ('--baseline-block-size','--candidate-block-size'):
        if argument(name)!='8':raise ValueError('This evaluation requires canonical B8 segmentation on both engines')
    evaluate_projection_quality.main()
    output=Path(argument('--output'));report=json.loads(output.read_text())
    for item,record in zip(precision,report['binaries']):
        if item['sha256']!=record['sha256']:raise RuntimeError('Artifact changed after capability check')
    report['runtime_precision']=precision
    report['quality_scope']='Descriptive FP32 KV numerical differences; no invented tolerance or quality improvement claim'
    report['prefix_cache_policy']='none; reset every example, B8 chunks then B4 tail, final prompt/decode B1'
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
