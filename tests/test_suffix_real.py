"""Opt-in seeded suffix parity against a real block-enabled SystemVerilog model."""
import hashlib
import json
import os
from pathlib import Path
import threading
import tempfile

import numpy as np
from tools.chat_protocol import ChatRequest
from tools.evaluate_projection_quality import read_dump_logits
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


HYBRID_BINARY=os.environ.get('SV_HYBRID_BINARY')


@unittest.skipUnless(HYBRID_BINARY and Path(HYBRID_BINARY).is_file(),
                     'Set SV_HYBRID_BINARY to the opt-in FP64 prefill / FP32 decode executable')
class RealHybridRouteTests(unittest.TestCase):
    def test_canonical_prefill_replay_restart_and_suffix_rejection(self):
        # Bit identity is required only between repetitions of this SAME route.
        # No cross-precision tolerance or suffix equivalence is assumed.
        engine=RTLChatEngine(binary=HYBRID_BINARY,workers=8,lanes=8,
                             block_size=1,prefill_size=8,reuse_prefix=False)
        self.addCleanup(engine.close)
        with tempfile.TemporaryDirectory() as temporary:
            dump=Path(temporary)/'logits.bin'
            for tail in range(1,9):
                ids=[9707]*(8+tail)+[11]
                request=ChatRequest('same-session',ids,3,0,decoding='chat',seed=42)
                snapshots=[]
                for repetition in range(3):
                    if repetition==2:engine.close()
                    final=list(engine.stream(request,threading.Event()))[-1]
                    self.assertEqual(final['type'],'done')
                    self.assertIn(final['finish_reason'],('length','eos'))
                    self.assertEqual(final['metrics']['reused_prefix_tokens'],0)
                    self.assertEqual(engine.cache_ids[:len(ids)],ids)
                    snapshots.append((final['token_ids'],list(engine.cache_ids),
                                      read_dump_logits(engine.sim,dump).view(np.uint64).copy()))
                for replay in snapshots[1:]:
                    self.assertEqual(replay[:2],snapshots[0][:2])
                    np.testing.assert_array_equal(replay[2],snapshots[0][2])
                previous=list(engine.cache_ids)
                with self.assertRaises(ValueError):
                    list(engine.stream(ChatRequest('same-session',ids,3,1),threading.Event()))
                self.assertEqual(engine.cache_ids,previous)


if __name__=='__main__': unittest.main()
