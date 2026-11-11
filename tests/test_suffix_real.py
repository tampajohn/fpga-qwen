"""Opt-in seeded suffix parity against a real block-enabled SystemVerilog model."""
import hashlib
import json
import os
from pathlib import Path
import threading
import unittest
from tools.rtl_chat import RTLChatEngine
from tools.experiment import ROOT

BINARY=os.environ.get('SV_BLOCK_BINARY')


@unittest.skipUnless(BINARY and Path(BINARY).is_file(),'Set SV_BLOCK_BINARY to a tested B4 executable')
class RealSuffixTests(unittest.TestCase):
    def test_seeded_on_off_text_ids_and_cache_match(self):
        rows=[]
        for decoding,seed in [('greedy',42),('chat',42),('chat',43)]:
            outputs=[]
            for block_size,max_draft in [(1,0),(4,7)]:
                engine=RTLChatEngine(binary=BINARY,lanes=8,workers=8,block_size=block_size)
                try:
                    payload=dict(session='suffix-test',messages=[dict(role='user',content='Continue this sequence for another ten repetitions. Output only the sequence: red blue red blue red blue red blue red blue')],max_new_tokens=40,max_draft=max_draft,decoding=decoding,seed=seed)
                    result=list(engine.stream(engine.prepare(payload),threading.Event()))[-1]
                    outputs.append(dict(result=result,cache_ids=list(engine.cache_ids)))
                finally:
                    engine.close()
            for key in ('text','token_ids','finish_reason'):
                self.assertEqual(outputs[0]['result'][key],outputs[1]['result'][key])
            self.assertEqual(outputs[0]['cache_ids'],outputs[1]['cache_ids'])
            rows.append(dict(decoding=decoding,seed=seed,outputs=outputs,exact_text_ids_and_cache=True))
        self.assertGreater(sum(row['outputs'][1]['result']['metrics']['accepted_tokens'] for row in rows),0)
        report=dict(binary_sha256=hashlib.sha256(Path(BINARY).read_bytes()).hexdigest(),
                    purpose='Correctness only; timing fields are not scored performance evidence.',cases=rows)
        output=ROOT/'build/suffix-optimized-real.json'
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': unittest.main()
