"""Descriptive fused-prefill corpus comparison with explicit numerical profiles."""
import hashlib
import json
from pathlib import Path
import sys
from tools.rtl_model import RTLModel
from tools import evaluate_projection_quality


def main():
    args=sys.argv[1:]
    def arg(name):return args[args.index(name)+1]
    for name in ('--baseline-block-size','--candidate-block-size'):
        if arg(name)!='8':raise ValueError('Canonical B8 segmentation required')
    precision=[]
    for role in ('baseline','candidate'):
        binary=Path(arg('--'+role)).resolve()
        model=RTLModel(binary=binary,workers=8,lanes=8,block_size=4,prefill_size=8)
        try:
            model.start();value=json.loads(model.command('precision'))
            if role=='baseline':
                if value['precision_profile'] not in ('fp32_b1_fp64_prefill','fp32_b1_fp32_prefill'):raise ValueError('Unexpected baseline profile')
            else:
                expected=dict(precision_profile='fp32_b1_fp32_fma_prefill',prefill_projection='fp32',prefill_accumulation='fused',scalar_accumulation='separate',verification_accumulation='separate',vocabulary_projection='fp32',vocabulary_accumulation='separate',canonical_prefix_policy='none')
                if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Unexpected fused-prefill capability')
            for k,v in dict(scalar_projection='fp32',verification_projection='fp32',attention_kv='fp64',max_verification_block=4,prefix_reuse_supported=False).items():
                if type(value.get(k)) is not type(v) or value[k]!=v:raise ValueError('Unexpected scalar/verification/KV route')
            precision.append(dict(role=role,sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),capability=value))
        finally:model.close()
    evaluate_projection_quality.main()
    output=Path(arg('--output'));report=json.loads(output.read_text())
    for item,record in zip(precision,report['binaries']):
        if item['sha256']!=record['sha256']:raise RuntimeError('Artifact changed')
    report['runtime_precision']=precision
    report['rounding_scope']='Cross-profile numerical differences are descriptive; no invented tolerance, bit-equality requirement or quality-improvement claim'
    report['prefix_cache_policy']='none: reset each example, canonical B8 chunks plus B4 tail, separate B1 final prompt/decode'
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
