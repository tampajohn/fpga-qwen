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


VERIFY_BINARY=os.environ.get('SV_FP32_VERIFY_BINARY')
VERIFY_PROFILE=os.environ.get('SV_FP32_VERIFY_PROFILE','fp32_b1_fp64_prefill')
if VERIFY_PROFILE not in ('fp32_b1_fp64_prefill','fp32_b1_fp32_fma_prefill'):raise ValueError('Unsupported suffix test profile')


@unittest.skipUnless(VERIFY_BINARY and Path(VERIFY_BINARY).is_file(),'Set SV_FP32_VERIFY_BINARY to the isolated FP32 B4 verification artifact')
class RealFP32SuffixTests(unittest.TestCase):
    def test_seeded_greedy_cache_rejection_and_cancel_replay(self):
        settings=dict(binary=VERIFY_BINARY,workers=8,lanes=8,block_size=4,prefill_size=8,
                      reuse_prefix=False,precision_profile=VERIFY_PROFILE,canonical_prefix_reuse=True)
        serial=RTLChatEngine(**dict(settings,canonical_prefix_reuse=VERIFY_PROFILE=='fp32_b1_fp64_prefill'));drafted=RTLChatEngine(**settings)
        for instance in (serial,drafted):instance.sim.phase_pool_selector=os.environ.get('SV_PHASE_POOL_SELECTOR')
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
                    outputs.append(dict(result=result,cache_ids=list(engine.cache_ids),canonical_end=engine.canonical_prefix.end if engine.canonical_prefix else 0,
                        raw_final_logits_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),raw_logit_position=raw_position,
                        retained_position=position,continuation=continuation[:2]))
                for key in ('text','token_ids','finish_reason'):
                    self.assertEqual(outputs[0]['result'][key],outputs[1]['result'][key],name)
                self.assertEqual(outputs[0]['cache_ids'],outputs[1]['cache_ids'])
                if VERIFY_PROFILE=='fp32_b1_fp64_prefill':
                    self.assertEqual(outputs[0]['canonical_end'],outputs[1]['canonical_end'])
                else:
                    self.assertEqual(outputs[0]['canonical_end'],0)
                    self.assertEqual(outputs[0]['result']['metrics']['reused_prefix_tokens'],0)
                self.assertEqual(outputs[0]['continuation'],outputs[1]['continuation'])
                if outputs[0]['raw_final_logits_sha256']!=outputs[1]['raw_final_logits_sha256']:
                    self.assertNotEqual(outputs[0]['raw_logit_position'],outputs[1]['raw_logit_position'])
                    self.assertGreater(outputs[1]['raw_logit_position'],outputs[1]['retained_position'])
                self.assertTrue(all(len(a)==151936 and np.isfinite(a).all() for a in logits))
                np.testing.assert_array_equal(logits[0].view(np.uint64),logits[1].view(np.uint64))
                rows.append(dict(name=name,decoding=decoding,seed=seed,draft=draft,outputs=outputs,
                    exact_logits=True,logits_sha256=hashlib.sha256(logits[0].tobytes()).hexdigest()))
            for decoding,seed in [('greedy',42),('chat',42),('chat',43)]:
                for draft in (1,3):
                    name=f'{decoding}-{seed}-{draft}'
                    compare(name,decoding,seed,draft)
                    compare(name+'-cached',decoding,seed,draft,session=name)
                    self.assertGreater(rows[-1]['outputs'][1]['result']['metrics']['reused_prefix_tokens'],0)
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
            self.assertEqual(drafted.canonical_prefix.end,0)
            compare('after-cancel','chat',42,3)
        report=dict(precision_profile=VERIFY_PROFILE,cache_off_reference=VERIFY_PROFILE=='fp32_b1_fp32_fma_prefill',phase_pool_selector=os.environ.get('SV_PHASE_POOL_SELECTOR'),
                    binary_sha256=hashlib.sha256(Path(VERIFY_BINARY).read_bytes()).hexdigest(),
                    completed=True,cancellation_recovery=True,timing_claim=False,
                    logit_gate='Fixed token198 continuation on retained accepted KV, then truncate probe; raw final dump may reference discarded speculative position',cases=rows)
        output=Path(os.environ.get('FP32_SUFFIX_REPORT',str(ROOT/'build/fp32-suffix-real.json')))
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': unittest.main()
