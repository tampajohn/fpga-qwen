"""Teacher-forced distribution comparison of CPU-only SV inference executables.

Host work is tokenization, transport and scoring; no host neural execution.
This descriptive held-out probe has no automatic approximation pass threshold.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
VOCAB=151936
DEFAULT_CORPUS=ROOT/'tests/fixtures/projection_quality_v1.json'


def hash_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def model_provenance(directory):
    directory=Path(directory)
    checkpoint=directory/'model.safetensors'
    names=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
           'special_tokens_map.json','added_tokens.json','vocab.json','merges.txt','chat_template.jinja']
    files=[directory/name for name in names if (directory/name).is_file()]
    files+=sorted((directory/'chat_templates').glob('*.jinja'))
    return dict(same_original_checkpoint_for_both_engines=True,
                original_checkpoint=dict(filename=checkpoint.name,bytes=checkpoint.stat().st_size,sha256=hash_file(checkpoint)),
                config_tokenizer_template_files={str(path.relative_to(directory)):hash_file(path)for path in files},
                chat_protocol_source_sha256=hash_file(Path(__file__).with_name('chat_protocol.py')),
                candidate_artifact_limitation='External quantized weights/calibration artifacts are not discovered automatically; supply their separate manifest before claiming full candidate reproducibility.')


def validate_logits(values,vocab_size=VOCAB):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=1 or values.size!=vocab_size:
        raise ValueError(f'Expected exactly {vocab_size} scalar logits, got {values.shape}')
    if not np.isfinite(values).all():raise ValueError('Nonfinite model logits')
    return values


def read_dump_logits(model,path,vocab_size=VOCAB):
    path=Path(path)
    if model.command(f'dump {path}')!='dumped':raise RuntimeError('Invalid dump acknowledgement')
    # Historical binaries acknowledged before the output stream destructor ran.
    # A completed next command guarantees closure, without sleeps or retries.
    if model.command(f'lanes {model.lanes}')!='configured':raise RuntimeError('Invalid dump barrier acknowledgement')
    actual=path.stat().st_size
    if actual!=vocab_size*8:
        raise ValueError(f'Wrong full-vocabulary dump byte count: expected {vocab_size*8}, got {actual}')
    return validate_logits(np.fromfile(path,dtype=np.float64),vocab_size)


def log_softmax(values):
    # Subtract first: no exp(raw_logits), and low-probability log values survive
    # even if their probabilities underflow during the KL-weight calculation.
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        shifted=values-np.max(values)
        result=shifted-math.log(float(np.exp(shifted).sum(dtype=np.float64)))
    if not np.isfinite(result).all():raise ValueError('Nonfinite normalized log probabilities')
    return result


def score_logits(reference,candidate,target,vocab_size=VOCAB):
    a,b=validate_logits(reference,vocab_size),validate_logits(candidate,vocab_size)
    if type(target) is not int or not 0<=target<vocab_size:raise ValueError('Target token out of range')
    ap,bp=log_softmax(a),log_softmax(b)
    with np.errstate(over='raise',invalid='raise'):
        kl=float(np.sum(np.exp(ap)*(ap-bp),dtype=np.float64))
        error=float(np.max(np.abs(a-b)))
    at,bt=int(np.argmax(a)),int(np.argmax(b))
    return dict(target_id=target,reference_nll_nats=float(-ap[target]),candidate_nll_nats=float(-bp[target]),
                delta_nll_nats=float(ap[target]-bp[target]),reference_top1=at,candidate_top1=bt,
                top1_agreement=at==bt,reference_target_is_top1=at==target,candidate_target_is_top1=bt==target,
                kl_reference_to_candidate_nats=kl,max_absolute_logit_error=error,
                bit_identical_logits=bool(np.array_equal(a.view(np.uint64),b.view(np.uint64))))


def aggregate(rows):
    if not rows:raise ValueError('Cannot aggregate zero scored tokens')
    count=len(rows)
    reference=math.fsum(r['reference_nll_nats'] for r in rows)/count
    candidate=math.fsum(r['candidate_nll_nats'] for r in rows)/count
    def exp_or_none(value):
        try:return math.exp(value)
        except OverflowError:return None
    return dict(scored_tokens=count,reference_mean_nll_nats=reference,candidate_mean_nll_nats=candidate,
                delta_mean_nll_nats=candidate-reference,
                reference_perplexity=exp_or_none(reference),candidate_perplexity=exp_or_none(candidate),
                perplexity_ratio_candidate_over_reference=exp_or_none(candidate-reference),
                top1_agreement=sum(r['top1_agreement'] for r in rows)/count,
                reference_target_top1_rate=sum(r['reference_target_is_top1'] for r in rows)/count,
                candidate_target_top1_rate=sum(r['candidate_target_is_top1'] for r in rows)/count,
                mean_kl_reference_to_candidate_nats=math.fsum(r['kl_reference_to_candidate_nats'] for r in rows)/count,
                max_kl_reference_to_candidate_nats=max(r['kl_reference_to_candidate_nats'] for r in rows),
                max_absolute_logit_error=max(r['max_absolute_logit_error'] for r in rows),
                all_logits_bit_identical=all(r['bit_identical_logits'] for r in rows),
                delta_nll_quantiles_nats={str(q):float(np.quantile([r['delta_nll_nats'] for r in rows],q)) for q in (.5,.9,.99)})


def feed_prefix(model,ids,block_size):
    if block_size not in (1,4):raise ValueError('Supported prefill block sizes are1 and4')
    for start in range(0,len(ids),block_size):
        chunk=ids[start:start+block_size]
        outputs=[model.token(chunk[0],logits=False)] if block_size==1 else model.block(chunk,logits=False)
        expected=list(range(start+1,start+len(chunk)+1))
        if [x[1] for x in outputs]!=expected:raise RuntimeError('Prefix position mismatch')


def teacher_force(reference,candidate,prompt_ids,target_ids,fetch_logits,reference_block_size=1,candidate_block_size=1,vocab_size=VOCAB):
    if not prompt_ids or not target_ids:raise ValueError('Prompt and target must be nonempty')
    if len(prompt_ids)+len(target_ids)>4096:raise ValueError('Example exceeds context capacity')
    if any(type(t) is not int or not 0<=t<vocab_size for t in [*prompt_ids,*target_ids]):
        raise ValueError('Prompt/target token out of range')
    for model,size in ((reference,reference_block_size),(candidate,candidate_block_size)):
        model.reset();feed_prefix(model,prompt_ids[:-1],size)
    rows=[]
    pending=prompt_ids[-1]
    for index,target in enumerate(target_ids):
        outputs=[model.token(pending,logits=True) for model in (reference,candidate)]
        position=len(prompt_ids)+index
        if any(result[1]!=position for result in outputs):raise RuntimeError('Teacher-forcing position mismatch')
        arrays=[fetch_logits(model,slot) for slot,model in enumerate((reference,candidate))]
        row=score_logits(*arrays,target,vocab_size)
        if [r[0] for r in outputs]!=[row['reference_top1'],row['candidate_top1']]:
            raise RuntimeError('Dump logits disagree with simulator greedy IDs')
        row.update(target_index=index,input_token_id=pending,context_position=position)
        rows.append(row)
        # The authored next token, never either model's prediction, advances both.
        pending=target
    return rows


def load_corpus(path,training_report=None):
    corpus=json.loads(Path(path).read_text())
    if corpus.get('schema_version')!=1 or corpus.get('role')!='evaluation_only':raise ValueError('Invalid corpus schema/role')
    examples=corpus.get('examples',[])
    if not examples or len({e['id'] for e in examples})!=len(examples):raise ValueError('Empty corpus or duplicate IDs')
    training=[]
    if training_report and Path(training_report).is_file():
        for case in json.loads(Path(training_report).read_text()).get('training_cases',[]):
            training.extend([case.get('prompt',''),case.get('text','')])
    normalized={' '.join(t.split()) for t in training if t}
    for e in examples:
        if not e.get('category') or not e.get('messages') or not isinstance(e.get('target'),str) or not e['target'].strip():
            raise ValueError('Invalid example')
        for text in [e['target'],*(m['content'] for m in e['messages'])]:
            if ' '.join(text.split()) in normalized:raise ValueError('Evaluation example duplicates recorded local training text')
    return corpus,dict(recorded_training_report_checked=bool(training),check='Exact whitespace-normalized prompt/target equality only; not a general contamination test.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--corpus',type=Path,default=DEFAULT_CORPUS)
    parser.add_argument('--model')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--lanes',type=int,choices=[1,8],default=8)
    parser.add_argument('--baseline-block-size',type=int,choices=[1,4],default=1)
    parser.add_argument('--candidate-block-size',type=int,choices=[1,4],default=1)
    parser.add_argument('--case-ids',default='')
    parser.add_argument('--max-target-tokens',type=int)
    parser.add_argument('--require-exact',action='store_true',help='Sanity check for exact engines; not an approximate-quality threshold')
    parser.add_argument('--output',type=Path,default=ROOT/'build/projection-quality.json')
    args=parser.parse_args()
    if not 1<=args.workers<=32 or (args.max_target_tokens is not None and args.max_target_tokens<1):parser.error('Invalid workload')
    try:
        from .rtl_model import RTLModel
        from .chat_protocol import ChatProtocol
    except ImportError:
        from rtl_model import RTLModel
        from chat_protocol import ChatProtocol
    corpus,overlap=load_corpus(args.corpus,ROOT/'reports/arm-pgo-training.json')
    requested=set(args.case_ids.split(',')) if args.case_ids else {e['id'] for e in corpus['examples']}
    if not requested<={e['id'] for e in corpus['examples']}:parser.error('Unknown case ID')
    selected=[e for e in corpus['examples'] if e['id'] in requested]
    binaries=[args.baseline.resolve(),args.candidate.resolve()]
    models=[RTLModel(args.model,workers=args.workers,lanes=args.lanes,binary=p,block_size=size)
            for p,size in zip(binaries,(args.baseline_block_size,args.candidate_block_size))]
    if models[0].model.resolve()!=models[1].model.resolve():raise RuntimeError('Checkpoint paths differ')
    protocol=ChatProtocol(models[0].model)
    corpus_target_counts={e['id']:len(protocol.tokenizer.encode(e['target'],add_special_tokens=False).ids)for e in corpus['examples']}
    provenance=model_provenance(models[0].model)
    report=dict(completed=False,decision='descriptive_only_no_quality_pass_threshold',gpu=False,pytorch=False,
                corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest(),corpus_name=corpus['name'],
                corpus_provenance=corpus['provenance'],local_training_overlap_check=overlap,
                tokenization='ChatProtocol generation prefix; target encoded separately without special tokens; no EOS target appended.',
                logit_dtype='float64',vocab_size=VOCAB,workers=args.workers,lanes=args.lanes,
                baseline_block_size=args.baseline_block_size,candidate_block_size=args.candidate_block_size,
                max_target_tokens=args.max_target_tokens,selected_case_ids=[e['id'] for e in selected],
                corpus_examples_available=len(corpus['examples']),corpus_authored_target_tokens_available=sum(corpus_target_counts.values()),
                model_provenance=provenance,scoring_source_sha256=hash_file(__file__),examples=[],
                binaries=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in binaries],
                limitations=['Small locally authored corpus; not broad capability validation.','Upstream pretraining membership unknown.',
                    'Likelihood of one chosen reference does not measure correctness of all valid alternative responses.',
                    'Teacher forcing does not measure accumulated free-running errors or long-context behavior.',
                    'Do not tune/calibrate on this evaluation corpus. Host scoring/dump overhead is not inference throughput.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    try:
        for m in models:m.start()
        with tempfile.TemporaryDirectory() as directory:
            def fetch(model,slot):
                path=Path(directory)/f'logits-{slot}.bin'
                return read_dump_logits(model,path)
            for case in selected:
                request=protocol.prepare(dict(session='heldout-quality-'+case['id'],messages=case['messages'],max_new_tokens=128,decoding='greedy'))
                all_targets=protocol.tokenizer.encode(case['target'],add_special_tokens=False).ids
                targets=all_targets[:args.max_target_tokens] if args.max_target_tokens else all_targets
                rows=teacher_force(*models,request.ids,targets,fetch,args.baseline_block_size,args.candidate_block_size)
                if args.require_exact and not all(r['bit_identical_logits'] for r in rows):raise AssertionError('Exact-engine sanity gate failed')
                record=dict(id=case['id'],category=case['category'],prompt_tokens=len(request.ids),prompt_ids=request.ids,
                            authored_target_tokens=len(all_targets),scored_target_tokens=len(targets),target_ids=targets,
                            target_truncated=len(targets)!=len(all_targets),per_token=rows,aggregate=aggregate(rows))
                report['examples'].append(record)
                args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                print(json.dumps(dict(case=case['id'],**record['aggregate']),allow_nan=False),flush=True)
        flat=[r for case in report['examples'] for r in case['per_token']]
        report['aggregate_token_weighted']=aggregate(flat)
        report['mean_of_example_mean_loss_deltas_nats']=float(np.mean([e['aggregate']['delta_mean_nll_nats'] for e in report['examples']]))
        report['by_category']={category:aggregate([r for e in report['examples'] if e['category']==category for r in e['per_token']])
                               for category in sorted({e['category'] for e in report['examples']})}
        report['coverage']=dict(selected_examples=len(selected),scored_examples=len(report['examples']),
            selected_authored_target_tokens=sum(e['authored_target_tokens']for e in report['examples']),
            scored_target_tokens=len(flat),total_corpus_authored_target_tokens=sum(corpus_target_counts.values()),
            fraction_of_corpus_target_tokens_scored=len(flat)/sum(corpus_target_counts.values()),unique_scored_target_ids=len({r['target_id']for r in flat}),
            total_prompt_tokens=sum(e['prompt_tokens']for e in report['examples']),
            categories=sorted(report['by_category']))
        report['completed']=True
    except Exception as error:
        report['error']=dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        for m in models:m.close()
        report['evaluation_seconds_not_throughput']=time.perf_counter()-started
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
