#!/usr/bin/env python3
"""Prepare complete-tokenized frozen prompts before any held-out measurements."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from chat_protocol import ChatProtocol
from experiment import locate_model
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--checkpoint');a=p.parse_args()
out=a.directory.resolve();assert not out.exists();checkpoint=locate_model(a.checkpoint);protocol=ChatProtocol(checkpoint)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows={}
for length in (192,768,1536,2560,3328):
 padding=0
 for attempt in range(10):
  messages=[dict(role='user',content='Background notes:'+' x'*padding+'\nIgnore the background notes. Write a detailed tutorial of at least 600 words explaining CPU memory bandwidth, cache levels, cache locality, and matrix multiplication. Use ten numbered paragraphs with concrete examples. Begin with paragraph 1.')]
  ids=protocol.prepare(dict(session='active8-heldout',messages=messages,max_new_tokens=128,decoding='greedy')).ids
  if len(ids)==length:break
  padding+=length-len(ids);assert padding>=0
 else:raise RuntimeError('Could not prepare exact complete prompt length')
 rows[f'context{length}']=dict(ids=ids,count=length,messages=messages)
paths=[Path(__file__).resolve(),ROOT/'tools/chat_protocol.py',ROOT/'tools/experiment.py']+[checkpoint/n for n in ('tokenizer.json','tokenizer_config.json','config.json')]
frozen={str(p):sha(p)for p in paths};out.mkdir()
for name,row in rows.items():
 path=out/(name+'.json');path.write_text(json.dumps(row,indent=2)+'\n');frozen[str(path)]=sha(path)
record=dict(completed=True,inference_run=False,stage='exploratory3pairs',lengths=[192,768,1536,2560,3328],tokens=128,rounds=3,warmups=1,starting_arm='baseline',alternating=True,decoding='greedy',seed=42,source_sha256=frozen,profiles=rows,policy='No context threshold chosen; no context/outlier removal. Triages budget only; no statistical/performance claim.')
(out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(out/'manifest.json')
