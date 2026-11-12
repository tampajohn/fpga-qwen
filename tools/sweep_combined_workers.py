"""Balanced frozen-artifact worker sweep; kernel routing differs by worker count."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import numpy as np
from benchmark_matched import SVEngine,make_prompt
from compare_attention_dot import request
from compare_attention_combined import read_dump_logits
from chat_protocol import ChatProtocol
from experiment import locate_model
from evaluate_projection_quality import model_provenance

FROZEN='f17ae654f0c64ac47b63225a3406edc8bcd114cc86e67d56610090bf5d7c075d'
COUNTS=(4,6,8,12)
ORDERS=((4,6,12,8),(6,8,4,12),(8,12,6,4),(12,4,8,6))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pool-policy',choices=('resident-per-count','switch-and-rewarm'),required=True)
    args=parser.parse_args();binary=args.binary.resolve()
    if sha(binary)!=FROZEN:raise ValueError('Use the reviewed frozen f17 artifact')
    manifest=json.loads((binary.parent.parent/'manifest.json').read_text())
    if manifest['binary_sha256']!=FROZEN:raise ValueError('Manifest mismatch')
    for key in ('attention_qk2','attention_head_pairs','attention_value_pairs','static_kv','batch_attention'):
        if manifest['options'].get(key) is not True:raise ValueError('Unexpected attention routing')
    checkpoint=locate_model(None);protocol=ChatProtocol(checkpoint)
    keys=COUNTS if args.pool_policy=='resident-per-count' else (8,)
    engines={n:SVEngine(checkpoint,n,protocol.tokenizer,binary,8,block_size=1,prefill_size=8,
                       precision_profile='fp32_b1_fp64_prefill',verification_capacity=4) for n in keys}
    report=dict(completed=False,binary_sha256=FROZEN,variant_options=manifest['options'],
        model_provenance=model_provenance(checkpoint),pool_policy=args.pool_policy,
        worker_counts=COUNTS,orders=ORDERS,tokens=32,prefill_size=8,lanes=8,
        routing={str(n):'op6 paired-head QK/V' if n in (4,8) else 'op1 scalar-head fallback' for n in COUNTS},
        hardware_inventory={},startup_seconds={},gpu=False,pytorch=False,profiles=[],
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_hashes={name:sha(Path(__file__).with_name(name)) for name in
            ('sweep_combined_workers.py','benchmark_matched.py','compare_attention_dot.py','compare_attention_combined.py','rtl_model.py')},
        limitations='Worker count changes both scheduling and scalar-attention kernel eligibility. No affinity/pinning or placement guarantee; do not infer which core class ran a worker. Coordinator exists in addition to workers. No native comparison in this sweep. Logical KV resets do not evict OS caches. Full tested output equality does not imply identical FP64 intermediate rounding on fallback paths.')
    for key in ('hw.physicalcpu','hw.perflevel0.name','hw.perflevel0.physicalcpu','hw.perflevel1.name','hw.perflevel1.physicalcpu'):
        result=subprocess.run(['sysctl','-n',key],text=True,capture_output=True)
        report['hardware_inventory'][key]=result.stdout.strip() if result.returncode==0 else None
    report['residency']='Four resident models; 7GiB KV payload alone, not measured RSS' if len(engines)==4 else 'One resident model; pool switched and request re-warmed before every scored slot; reconfiguration/warmup excluded from latency'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def model_for(n):
        model=engines[n if len(engines)==4 else 8].model
        model.set_workers(n)
        return model
    try:
        report['startup_seconds']={str(n):e.start() for n,e in engines.items()}
        report['initial_engine_metadata']={str(n):e.metadata for n,e in engines.items()}
        with tempfile.TemporaryDirectory() as directory:
            dump=Path(directory)/'logits.bin'
            for profile in ('short','long'):
                prompt=make_prompt(protocol,profile)
                record=dict(profile=profile,prompt_token_ids=prompt,prompt_tokens=len(prompt),warmups=[],rounds=[])
                report['profiles'].append(record)
                expected_ids=None;expected_bits=None
                def checked(n):
                    nonlocal expected_ids,expected_bits
                    model=model_for(n);result=request(model,prompt,32)
                    logits=read_dump_logits(model,dump);bits=logits.view(np.uint64)
                    if len(result['output_token_ids'])!=32 or result['final_position']!=len(prompt)+31:
                        raise AssertionError('Consumed work differs')
                    if int(np.argmax(logits))!=result['output_token_ids'][-1]:raise AssertionError('Misaligned dump')
                    if expected_ids is None:
                        expected_ids=result['output_token_ids'];expected_bits=bits.copy()
                    if result['output_token_ids']!=expected_ids or not np.array_equal(bits,expected_bits):
                        raise AssertionError(f'Worker count {n} changed tested greedy IDs or full logit bits')
                    result['full_logits_sha256']=hashlib.sha256(logits.tobytes()).hexdigest()
                    result['workers']=n;result['exact_output_gate']=True
                    return result
                # Establish the comparison against the deployed eight-worker route.
                record['warmups'].append(checked(8))
                if len(engines)==4:
                    for n in (4,6,12):record['warmups'].append(checked(n))
                for index,order in enumerate(ORDERS):
                    row=dict(index=index,order=order,results={})
                    record['rounds'].append(row)
                    for n in order:
                        if len(engines)==1:record['warmups'].append(checked(n))
                        row['results'][str(n)]=checked(n)
                        save()
                    print(json.dumps(dict(profile=profile,round=index,decode_tps={n:r['decode_tokens_per_second'] for n,r in row['results'].items()})),flush=True)
                record['summary']={}
                for n in COUNTS:
                    record['summary'][str(n)]={}
                    for metric in ('prefill_seconds','ttft_seconds','decode_seconds','decode_tokens_per_second','total_seconds'):
                        values=[r['results'][str(n)][metric] for r in record['rounds']]
                        record['summary'][str(n)][metric]=dict(median=statistics.median(values),minimum=min(values),maximum=max(values))
                    record['summary'][str(n)]['paired_speedups_vs8']={metric:[r['results']['8'][metric]/r['results'][str(n)][metric] for r in record['rounds']] for metric in ('prefill_seconds','ttft_seconds','decode_seconds','total_seconds')}
        if sha(binary)!=FROZEN:raise RuntimeError('Artifact changed during evaluation')
        report['completed']=True
    finally:
        for engine in engines.values():engine.close()
        save()

if __name__=='__main__':main()
