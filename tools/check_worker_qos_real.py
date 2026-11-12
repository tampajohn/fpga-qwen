"""Real requested-QoS telemetry and exact output checks, never a timing report."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from benchmark_matched import SVEngine,make_prompt
from chat_protocol import ChatProtocol
from compare_attention_dot import request
from compare_attention_combined import read_dump_logits
from experiment import locate_model
KEY='QWEN_EXPERIMENTAL_WORKER_QOS'
def start_with_selector(engine,selector):
    previous=os.environ.get(KEY)
    try:
        if selector is None:os.environ.pop(KEY,None)
        else:os.environ[KEY]=selector
        engine.start()
    finally:
        if previous is None:os.environ.pop(KEY,None)
        else:os.environ[KEY]=previous

def validate_telemetry(engine,selector):
    actual=json.loads(engine.model.command('worker_qos'))
    assert actual['selector']==('UNCHANGED' if selector is None else selector),actual
    assert actual['coordinator_before']==actual['coordinator_now'],actual
    expected=25 if selector=='USER_INITIATED' else 21
    assert len(actual['workers'])==8,actual
    assert [w['id']for w in actual['workers']]==list(range(8)),actual
    assert all(w['requested_class']==expected and w['relative_priority']==0 for w in actual['workers']),actual
    return actual

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--binary',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();checkpoint=locate_model(None);protocol=ChatProtocol(checkpoint)
    def engine(path):return SVEngine(checkpoint,8,protocol.tokenizer,path,8,block_size=1,prefill_size=8,precision_profile='fp32_b1_fp64_prefill',verification_capacity=4)
    engines=[engine(args.reference)]+[engine(args.binary)for _ in range(3)]
    report=dict(completed=False,cases=[],telemetry=[],binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest())
    try:
        start_with_selector(engines[0],None)
        for e,selector in zip(engines[1:],(None,'DEFAULT','USER_INITIATED')):
            start_with_selector(e,selector);report['telemetry'].append(validate_telemetry(e,selector))
        with tempfile.TemporaryDirectory()as directory:
            for profile in ('short','long'):
                prompt=make_prompt(protocol,profile);expected=None
                for i,e in enumerate(engines):
                    result=request(e.model,prompt,4);logits=read_dump_logits(e.model,Path(directory)/f'{i}.bin')
                    current=(result['output_token_ids'],result['final_position'],logits.tobytes())
                    if expected is None:expected=current
                    assert current==expected,(profile,i)
                    report['cases'].append(dict(profile=profile,engine=i,ids=current[0],position=current[1],full_logits_sha256=hashlib.sha256(current[2]).hexdigest(),exact=True))
        for e,selector in zip(engines[1:],(None,'DEFAULT','USER_INITIATED')):validate_telemetry(e,selector)
    finally:
        for e in engines:e.close()
    invalid=engine(args.binary)
    try:
        try:start_with_selector(invalid,'INVALID');raise AssertionError('Invalid selector accepted')
        except RuntimeError:report['invalid_selector_rejected']=True
    finally:invalid.close()
    report['completed']=True;args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
