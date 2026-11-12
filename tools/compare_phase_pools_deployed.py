"""Direct frozen deployed f17 versus persistent12-prefill/8-decode candidate."""
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
from check_phase_pools_real import telemetry
from evaluate_projection_quality import validate_logits,model_provenance


def start_with_selector(engine,selector):
    engine.model.phase_pool_selector=selector
    engine.start()

def validate_telemetry(engine,selector):
    return None if selector is None else telemetry(engine.model,selector=='PREFILL_12')


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
    if arrays[0].tobytes()!=arrays[1].tobytes():raise AssertionError('Pool selection changed full-vocabulary logit bits')
    delta=arrays[1]-arrays[0]
    return dict(max_absolute_logit_error=float(np.max(np.abs(delta))),rms_logit_error=float(np.sqrt(np.mean(delta*delta))),
                full_vocabulary_size=len(delta),top1_ids=greedy,bit_identical=bool(np.array_equal(arrays[0].view(np.uint64),arrays[1].view(np.uint64))),
                interpretation='Exact full-vocabulary equality required for unchanged per-query arithmetic.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--pairs',type=int,default=3);parser.add_argument('--tokens',type=int,default=32)
    parser.add_argument('--profiles',default='short,long');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();profiles=args.profiles.split(',')
    if not 1<=args.pairs<=10 or not 2<=args.tokens<=128 or any(p not in ('short','long')for p in profiles):parser.error('Invalid workload')
    checkpoint=locate_model(None);protocol=ChatProtocol(checkpoint)
    paths=[args.reference.resolve(),args.candidate.resolve()]
    frozen=('f17ae654f0c64ac47b63225a3406edc8bcd114cc86e67d56610090bf5d7c075d','162bed6577d91b19a57ea0f83d244d3fbb2666e329e90733f56f347267e8580d')
    if tuple(sha(p)for p in paths)!=frozen:raise RuntimeError('Unexpected frozen artifact pair')
    manifests=[json.loads((p.parent.parent/'manifest.json').read_text())for p in paths]
    if not manifests[1]['options'].get('phase_pools_experiment') or manifests[0]['options'].get('phase_pools_experiment',False):raise RuntimeError('Wrong phase-pool build selection')
    if any(m['binary_sha256']!=h for m,h in zip(manifests,frozen)):raise RuntimeError('Manifest mismatch')
    engines=[SVEngine(checkpoint,8,protocol.tokenizer,paths[0],8,block_size=1,prefill_size=8,precision_profile='fp32_b1_fp64_prefill',verification_capacity=4),
             SVEngine(checkpoint,8,protocol.tokenizer,paths[1],8,block_size=1,prefill_size=8,precision_profile='fp32_b1_fp64_prefill',verification_capacity=4)]
    report=dict(variant_options=[m["options"]for m in manifests],completed=False,gpu=False,pytorch=False,workers=8,tokens=args.tokens,
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        binaries=[dict(path=str(p),sha256=sha(p))for p in paths],model_provenance=model_provenance(checkpoint),
        source_hashes={p.name:sha(p)for p in (Path(__file__),Path(__file__).with_name('compare_attention_dot.py'),Path(__file__).with_name('benchmark_matched.py'),Path(__file__).with_name('rtl_model.py'),Path(__file__).with_name('check_phase_pools_real.py'))},
        selectors=[None,'PREFILL_12'],
        policy='Both requests reset KV and prefill prompt[:-1] in canonical chunks8 with B4 tails; Both use FP64 prefill projections and FP32 scalar final-prompt/decode. Both retain batched prefill attention; Reference is deployed f17 with one8-worker pool; candidate uses12 for explicit prefill commands and8 for scalar/verification with two resident pools. Source arithmetic unchanged; separately linked executables. One shared mainmodel/KV. No core placement guarantee. No prefix reuse or suffix verification.',
        limitations='Two resident models: reference8 worker threads, candidate20 resident worker threads, each with one KV store. Sequential warmed inference in alternating order. This measures full deployment composition including host/layout/pool overhead, not isolated worker-count causality. Separate same-binary OFF comparison exists; do not multiply its ratios. Direct acknowledged/closed dumps preserve worker pools after warmup. No OS cache eviction or hard host isolation. Full-logit bits and fixed greedy IDs must match; separate causal/cache/sampled verification gates required.',profiles=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        report['startup_seconds']=[]
        report['phase_readback']=[]
        for engine,selector in zip(engines,report['selectors']):
            start=time.perf_counter();start_with_selector(engine,selector)
            report['startup_seconds'].append(time.perf_counter()-start)
            report['phase_readback'].append(validate_telemetry(engine,selector))
            engine.metadata.update(resident_worker_threads=20 if selector else 8,prefill_worker_count=12 if selector=='PREFILL_12' else 8,scalar_verification_worker_count=8,cpu_thread_note=('Two resident pools8+12 plus coordinator; only one pool receives a synchronous job at a time' if selector else 'One8-worker pool plus coordinator'))
        report['engines']=[engine.metadata for engine in engines]
        models=[engine.model for engine in engines]
        def guarded_request(index,prompt):
            before=validate_telemetry(engines[index],report['selectors'][index])
            result=request(models[index],prompt,args.tokens)
            after=validate_telemetry(engines[index],report['selectors'][index])
            if index==0:
                result['phase_scope']='Deployed reference has no purpose telemetry; fixed eight-worker pool verified by frozen source/artifact'
                return result
            delta=[[after['dispatch_counts'][p][k]-before['dispatch_counts'][p][k]for k in range(3)]for p in range(2)]
            prefill_index=1 if report['selectors'][index]=='PREFILL_12' else 0
            if delta[0][0]<=0 or delta[prefill_index][1]<=0 or delta[0][2]!=0 or delta[1][2]!=0 or delta[1][0]!=0 or delta[1-prefill_index][1]!=0:
                raise AssertionError('Request phase routing counters differ from intended work')
            result['phase_dispatch_delta']=delta
            result['phase_before']=before;result['phase_after']=after
            result['phase_scope']='dispatch/group counters are coordinator logical routing counts; no actual CPU affinity or per-core residency measurement'
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
                    # Verify every scored pair outside the request clocks, not
                    # only warmup and the final pair of each prompt profile.
                    full_logits=diagnostics(models,[results[0],results[1]],directory)
                    record['pairs'].append(dict(index=pair,order=order,baseline=results[0],candidate=results[1],ratios=ratios,full_logits=full_logits))
                    args.output.write_text(json.dumps(report,indent=2)+'\n')
                    print(json.dumps(dict(profile=profile,pair=pair,ratios=ratios)),flush=True)
                record['median_ratios']={k:statistics.median(p['ratios'][k]for p in record['pairs'])for k in ratios}
                record['ratio_ranges']={k:[min(p['ratios'][k]for p in record['pairs']),max(p['ratios'][k]for p in record['pairs'])]for k in ratios}
                record['final_full_logits']=diagnostics(models,[results[0],results[1]],directory)
        report['phase_final_readback']=[validate_telemetry(e,s)for e,s in zip(engines,report['selectors'])]
        if any(sha(path)!=record['sha256'] for path,record in zip(paths,report['binaries'])):raise RuntimeError('Binary changed during evaluation')
        report['completed']=True
    finally:
        for engine in engines:engine.close()
        args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
