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
from unittest.mock import patch
from tools.rtl_chat import RTLChatEngine
from tools.experiment import ROOT

KV_BINARY=os.environ.get('SV_FMA_PREFILL_BINARY')

@unittest.skipUnless(KV_BINARY and Path(KV_BINARY).is_file(),'Set SV_FMA_PREFILL_BINARY to the frozen experimental fused-prefill artifact')
class RealFMAPrefillSuffixTests(unittest.TestCase):
    def test_all_prefill_tails_exact_reset_and_restart(self):
        engine=RTLChatEngine(binary=KV_BINARY,workers=8,lanes=8,block_size=4,prefill_size=8,
                             reuse_prefix=False,precision_profile='fp32_b1_fp32_fma_prefill')
        self.addCleanup(engine.close)
        with tempfile.TemporaryDirectory() as directory:
            for tail in range(1,9):
                request=ChatRequest('same-session',[9707]*(8+tail)+[11],3,0,decoding='chat',seed=42)
                snapshots=[]
                for repeat in range(3):
                    if repeat==2:engine.close()
                    final=list(engine.stream(request,threading.Event()))[-1]
                    self.assertEqual(final['type'],'done')
                    self.assertEqual(final['metrics']['reused_prefix_tokens'],0)
                    snapshots.append((final['token_ids'],list(engine.cache_ids),
                        read_dump_logits(engine.sim,Path(directory)/'logits').view(np.uint64).copy()))
                for actual in snapshots[1:]:
                    self.assertEqual(actual[:2],snapshots[0][:2])
                    np.testing.assert_array_equal(actual[2],snapshots[0][2])

    def test_seeded_greedy_cache_rejection_and_cancel_replay(self):
        settings=dict(binary=KV_BINARY,workers=8,lanes=8,block_size=4,prefill_size=8,
                      reuse_prefix=False,precision_profile='fp32_b1_fp32_fma_prefill')
        serial=RTLChatEngine(**settings);drafted=RTLChatEngine(**settings)
        self.addCleanup(serial.close);self.addCleanup(drafted.close)
        for engine in (serial,drafted):
            engine.last_logit_position=None
            token_impl,block_impl=engine.sim.token,engine.sim.block
            def token(*args,_engine=engine,_impl=token_impl,**kwargs):
                result=_impl(*args,**kwargs)
                if kwargs.get('logits',True):_engine.last_logit_position=result[1]
                return result
            def block(*args,_engine=engine,_impl=block_impl,**kwargs):
                result=_impl(*args,**kwargs)
                if kwargs.get('logits',False):_engine.last_logit_position=result[-1][1]
                return result
            engine.sim.token=token;engine.sim.block=block
        prompt='Continue this sequence for another ten repetitions. Output only the sequence: red blue red blue red blue red blue red blue'
        rows=[]
        with tempfile.TemporaryDirectory() as directory:
            def compare(name,decoding,seed,draft,session=None):
                outputs=[];logits=[]
                for index,(engine,limit) in enumerate(((serial,0),(drafted,draft))):
                    request=engine.prepare(dict(session=session or name,messages=[dict(role='user',content=prompt)],
                        max_new_tokens=40,max_draft=limit,decoding=decoding,seed=seed))
                    result=list(engine.stream(request,threading.Event()))[-1]
                    self.assertEqual(result['type'],'done')
                    raw=read_dump_logits(engine.sim,Path(directory)/str(index))
                    position=len(engine.cache_ids)
                    raw_position=engine.last_logit_position
                    # A rejected draft may leave the dump buffer at a discarded
                    # future position. Probe the SAME retained KV in both modes.
                    continuation=engine.sim.token(198)
                    self.assertEqual(continuation[1],position+1)
                    logits.append(read_dump_logits(engine.sim,Path(directory)/str(index)))
                    self.assertEqual(engine.sim.command(f'truncate {position}'),'truncated')
                    outputs.append(dict(result=result,cache_ids=list(engine.cache_ids),canonical_end=None,
                        raw_final_logits_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),raw_logit_position=raw_position,
                        retained_position=position,continuation=continuation[:2]))
                for key in ('text','token_ids','finish_reason'):
                    self.assertEqual(outputs[0]['result'][key],outputs[1]['result'][key],name)
                self.assertEqual(outputs[0]['cache_ids'],outputs[1]['cache_ids'])
                self.assertEqual(outputs[0]['canonical_end'],outputs[1]['canonical_end'])
                self.assertEqual(outputs[0]['continuation'],outputs[1]['continuation'])
                if outputs[0]['raw_final_logits_sha256']!=outputs[1]['raw_final_logits_sha256']:
                    self.assertNotEqual(outputs[0]['raw_logit_position'],outputs[1]['raw_logit_position'])
                    self.assertGreater(outputs[1]['raw_logit_position'],outputs[1]['retained_position'])
                np.testing.assert_array_equal(logits[0].view(np.uint64),logits[1].view(np.uint64))
                rows.append(dict(name=name,decoding=decoding,seed=seed,draft=draft,outputs=outputs,
                    exact_logits=True,logits_sha256=hashlib.sha256(logits[0].tobytes()).hexdigest()))
            for decoding,seed in [('greedy',42),('chat',42),('chat',43)]:
                for draft in (1,3):
                    name=f'{decoding}-{seed}-{draft}'
                    compare(name,decoding,seed,draft)
                    compare(name+'-cached',decoding,seed,draft,session=name)
                    self.assertEqual(rows[-1]['outputs'][1]['result']['metrics']['reused_prefix_tokens'],0)
            self.assertGreater(sum(r['outputs'][1]['result']['metrics']['accepted_tokens'] for r in rows),0)
            # Force wrong but valid proposals to exercise truncation/RNG rollback.
            with patch('tools.rtl_chat.SuffixDrafter.propose',side_effect=lambda history,count:[0]*count):
                compare('forced-rejection','chat',42,3)
            metrics=rows[-1]['outputs'][1]['result']['metrics']
            self.assertGreater(metrics['proposed_tokens'],metrics['accepted_tokens'])
            cancel=threading.Event()
            request=drafted.prepare(dict(session='cancel',messages=[dict(role='user',content=prompt)],
                max_new_tokens=40,max_draft=3,decoding='chat',seed=42))
            cancelled=False
            for event in drafted.stream(request,cancel):
                if event['type']=='update' and event['metrics']['proposed_tokens']:
                    cancel.set();cancelled=True
            self.assertTrue(cancelled)
            self.assertIsNone(drafted.sim.process)
            self.assertEqual(drafted.cache_ids,[])
            self.assertIsNone(drafted.canonical_prefix)
            compare('after-cancel','chat',42,3)
        report=dict(binary_sha256=hashlib.sha256(Path(KV_BINARY).read_bytes()).hexdigest(),
                    completed=True,cancellation_recovery=True,timing_claim=False,
                    logit_gate='Fixed token198 continuation on retained accepted KV, then truncate probe; raw final dump may reference discarded speculative position',cases=rows)
        output=Path(os.environ.get('FMA_PREFILL_SUFFIX_REPORT',str(ROOT/'build/fma-prefill-suffix-real.json')))
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': unittest.main()
