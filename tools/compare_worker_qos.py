"""Same-binary DEFAULT versus USER_INITIATED worker QoS experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np
from benchmark_matched import SVEngine,make_prompt
from compare_attention_dot import request
from chat_protocol import ChatProtocol
from experiment import locate_model
from check_worker_qos_real import start_with_selector,validate_telemetry
from evaluate_projection_quality import validate_logits,model_provenance


def read_dump_logits(model,path):
    # Both evaluated artifacts close/check the file before acknowledging.
    # Historical lane barriers recreate pools and would invalidate warmup.
    path=Path(path)
    if model.command(f'dump {path}')!='dumped':raise RuntimeError('Invalid dump acknowledgement')
    if path.stat().st_size!=151936*8:raise ValueError('Wrong full-vocabulary dump byte count')
    return validate_logits(np.fromfile(path,dtype=np.float64))


def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def diagnostics(models,outputs,directory):
    arrays=[read_dump_logits(m,Path(directory)/f'{i}.bin')for i,m in enumerate(models)]
    greedy=[int(np.argmax(a))for a in arrays]
    if greedy!=[r['output_token_ids'][-1]for r in outputs]:raise AssertionError('Dump alignment differs from final prediction')
    if arrays[0].tobytes()!=arrays[1].tobytes():raise AssertionError('QoS selection changed full-vocabulary logit bits')
    delta=arrays[1]-arrays[0]
    return dict(max_absolute_logit_error=float(np.max(np.abs(delta))),rms_logit_error=float(np.sqrt(np.mean(delta*delta))),
                full_vocabulary_size=len(delta),top1_ids=greedy,bit_identical=bool(np.array_equal(arrays[0].view(np.uint64),arrays[1].view(np.uint64))),
                interpretation='Exact full-vocabulary equality required for unchanged per-query arithmetic.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True);parser.add_argument('--expected-sha',required=True)
    parser.add_argument('--pairs',type=int,default=3);parser.add_argument('--tokens',type=int,default=32)
    parser.add_argument('--profiles',default='short,long');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();profiles=args.profiles.split(',')
    if not 1<=args.pairs<=10 or not 2<=args.tokens<=128 or any(p not in ('short','long')for p in profiles):parser.error('Invalid workload')
    checkpoint=locate_model(None);protocol=ChatProtocol(checkpoint)
    paths=[args.binary.resolve(),args.binary.resolve()]
    if sha(paths[0])!=args.expected_sha:raise RuntimeError('Unexpected artifact SHA')
    manifests=[json.loads((p.parent.parent/'manifest.json').read_text())for p in paths]
    if not manifests[0]['options'].get('worker_qos_experiment'):raise RuntimeError('QoS experiment not compiled')
    if manifests[0]['binary_sha256']!=args.expected_sha:raise RuntimeError('Manifest artifact mismatch')
    engines=[SVEngine(checkpoint,8,protocol.tokenizer,paths[0],8,block_size=1,prefill_size=8,precision_profile='fp32_b1_fp64_prefill',verification_capacity=4),
             SVEngine(checkpoint,8,protocol.tokenizer,paths[1],8,block_size=1,prefill_size=8,precision_profile='fp32_b1_fp64_prefill',verification_capacity=4)]
    report=dict(variant_options=[m["options"]for m in manifests],completed=False,gpu=False,pytorch=False,workers=8,tokens=args.tokens,
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        binaries=[dict(path=str(p),sha256=sha(p))for p in paths],model_provenance=model_provenance(checkpoint),
        source_hashes={p.name:sha(p)for p in (Path(__file__),Path(__file__).with_name('compare_attention_dot.py'),Path(__file__).with_name('benchmark_matched.py'),Path(__file__).with_name('rtl_model.py'),Path(__file__).with_name('check_worker_qos_real.py'))},
        selectors=['DEFAULT','USER_INITIATED'],
        policy='Both requests reset KV and prefill prompt[:-1] in canonical chunks8 with B4 tails; Both use FP64 prefill projections and FP32 scalar final-prompt/decode. Both retain batched prefill attention; Identical compiled math and8workers; only worker startup QoS selector changes DEFAULT to USER_INITIATED. Coordinator untouched, no core placement guarantee. No prefix reuse or suffix verification.',
        limitations='Two resident models; sequential warmed inference in alternating order. Direct acknowledged/closed dumps preserve worker pools after warmup. No OS cache eviction or hard host isolation. Full-logit bits and fixed greedy IDs must match; separate causal/cache/sampled verification gates required.',profiles=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        report['startup_seconds']=[]
        report['qos_readback']=[]
        for engine,selector in zip(engines,report['selectors']):
            start=time.perf_counter();start_with_selector(engine,selector)
            report['startup_seconds'].append(time.perf_counter()-start)
            report['qos_readback'].append(validate_telemetry(engine,selector))
        if report['qos_readback'][0]['coordinator_before']!=report['qos_readback'][1]['coordinator_before']:raise RuntimeError('Coordinator QoS differs across processes')
        report['engines']=[engine.metadata for engine in engines]
        models=[engine.model for engine in engines]
        def guarded_request(index,prompt):
            before=validate_telemetry(engines[index],report['selectors'][index])
            result=request(models[index],prompt,args.tokens)
            after=validate_telemetry(engines[index],report['selectors'][index])
            result['qos_before']=before;result['qos_after']=after
            result['qos_scope']='worker observations are cached startup requested classes; coordinator_now is queried live; no effective in-flight scheduling or placement measurement'
            return result
        with tempfile.TemporaryDirectory()as directory:
            for profile in profiles:
                prompt=make_prompt(protocol,profile)
                warm=[guarded_request(i,prompt)for i in range(len(models))]
                if any(r['final_position']!=len(prompt)+args.tokens-1 for r in warm):raise AssertionError('Warm consumed work differs')
                expected=warm[0]['output_token_ids']
                if len(expected)!=args.tokens or warm[1]['output_token_ids']!=expected:raise AssertionError('Warm fixed-count greedy IDs differ')
                record=dict(profile=profile,prompt_ids=prompt,prompt_tokens=len(prompt),warmup=warm,
                            warm_full_logits=diagnostics(models,warm,directory),pairs=[])
                report['profiles'].append(record)
                for pair in range(args.pairs):
                    order=[0,1]if pair%2==0 else[1,0]
                    results={i:guarded_request(i,prompt)for i in order}
                    if any(r['output_token_ids']!=expected or len(r['output_token_ids'])!=args.tokens for r in results.values()):raise AssertionError('Scored fixed-count greedy IDs differ')
                    if any(r['final_position']!=len(prompt)+args.tokens-1 for r in results.values()):raise AssertionError('Scored consumed work differs')
                    ratios={k:results[0][k]/results[1][k]for k in ('prefill_seconds','ttft_seconds','decode_seconds','total_seconds')}
                    record['pairs'].append(dict(index=pair,order=order,baseline=results[0],candidate=results[1],ratios=ratios))
                    args.output.write_text(json.dumps(report,indent=2)+'\n')
                    print(json.dumps(dict(profile=profile,pair=pair,ratios=ratios)),flush=True)
                record['median_ratios']={k:statistics.median(p['ratios'][k]for p in record['pairs'])for k in ratios}
                record['ratio_ranges']={k:[min(p['ratios'][k]for p in record['pairs']),max(p['ratios'][k]for p in record['pairs'])]for k in ratios}
                record['final_full_logits']=diagnostics(models,[results[0],results[1]],directory)
        report['qos_final_readback']=[validate_telemetry(e,s)for e,s in zip(engines,report['selectors'])]
        if any(sha(path)!=record['sha256'] for path,record in zip(paths,report['binaries'])):raise RuntimeError('Binary changed during evaluation')
        report['completed']=True
    finally:
        for engine in engines:engine.close()
        args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
