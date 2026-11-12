"""Exact fresh-versus-cached hybrid-route gates; opt-in real simulator only."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
import numpy as np
from tools.rtl_chat import RTLChatEngine
from tools.chat_protocol import ChatRequest, common_prefix
from tools.evaluate_projection_quality import read_dump_logits

BINARY=os.environ.get('SV_CANONICAL_BINARY')
PROFILE=os.environ.get('SV_CANONICAL_PROFILE','fp32_b1_fp64_prefill')
POLICIES={'fp32_b1_fp64_prefill':'fp64_b8_aligned_v1','fp32_b1_fp32_fma_prefill':'fp32_fma_b8_aligned_v1'}
if PROFILE not in POLICIES:raise ValueError('Unsupported canonical test profile')
POLICY=POLICIES[PROFILE]
FRESH_CANONICAL=PROFILE=='fp32_b1_fp64_prefill'
BLOCK_SIZE=int(os.environ.get('SV_CANONICAL_BLOCK_SIZE','1'))
FIXED_WORKERS=os.environ.get('SV_CANONICAL_FIXED_WORKERS')
if FIXED_WORKERS not in (None,'8'):raise ValueError('Fixed canonical worker fixture supports only8')

@unittest.skipUnless(BINARY and Path(BINARY).is_file(),'Set SV_CANONICAL_BINARY to policy-capable experimental hybrid')
class RealCanonicalPrefixTests(unittest.TestCase):
    def test_matching_fp32_decode_tokens_are_recomputed(self):
        settings=dict(binary=BINARY,workers=8,lanes=8,block_size=BLOCK_SIZE,prefill_size=8,
                      reuse_prefix=False,precision_profile=PROFILE)
        cached=RTLChatEngine(**settings,canonical_prefix_reuse=True)
        fresh=RTLChatEngine(**settings,canonical_prefix_reuse=FRESH_CANONICAL and BLOCK_SIZE==4)
        for instance in (cached,fresh):instance.sim.phase_pool_selector=os.environ.get('SV_PHASE_POOL_SELECTOR')
        self.addCleanup(cached.close);self.addCleanup(fresh.close)
        ids=[9707,11,1917,0]*8+[358]
        list(cached.stream(ChatRequest('s',ids,4,0,decoding='chat',seed=42),threading.Event()))
        boundary=cached.canonical_prefix.end
        next_ids=list(cached.cache_ids)+[358]
        shared=common_prefix(cached.cache_ids,next_ids[:-1])
        self.assertGreater(shared,boundary)
        outputs=[list(e.stream(ChatRequest('s',next_ids,4,0,decoding='chat',seed=42),threading.Event()))[-1]
                 for e in (cached,fresh)]
        self.assertEqual(outputs[0]['metrics']['reused_prefix_tokens'],boundary)
        self.assertEqual(outputs[1]['metrics']['reused_prefix_tokens'],0)
        self.assertEqual(outputs[0]['token_ids'],outputs[1]['token_ids'])
        self.assertEqual(cached.cache_ids,fresh.cache_ids)
        with tempfile.TemporaryDirectory() as d:
            arrays=[read_dump_logits(e.sim,Path(d)/str(i)) for i,e in enumerate((cached,fresh))]
        self.assertTrue(all(np.isfinite(a).all() and len(a)==151936 for a in arrays))
        np.testing.assert_array_equal(arrays[0].view(np.uint64),arrays[1].view(np.uint64))
        report=dict(completed=True,binary_sha256=hashlib.sha256(Path(BINARY).read_bytes()).hexdigest(),
                    phase_pool_selector=os.environ.get('SV_PHASE_POOL_SELECTOR'),
                    precision_profile=PROFILE,policy=POLICY,shared_prefix_tokens=shared,prior_canonical_end=boundary,
                    reused_tokens=outputs[0]['metrics']['reused_prefix_tokens'],
                    recomputed_matching_fp32_tokens=shared-boundary,
                    output_ids=outputs[0]['token_ids'],full_logits_bit_identical=True,timing_claim=False)
        path=Path(os.environ.get('CANONICAL_DECODE_REPORT','build/canonical-prefix-decode-exclusion.json'))
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')

    def test_fresh_cached_exact_logits_outputs_and_lifecycle(self):
        settings=dict(binary=BINARY,workers=8,lanes=8,block_size=BLOCK_SIZE,prefill_size=8,
                      reuse_prefix=False,precision_profile=PROFILE)
        cached=RTLChatEngine(**settings,canonical_prefix_reuse=True)
        fresh=RTLChatEngine(**settings,canonical_prefix_reuse=FRESH_CANONICAL and BLOCK_SIZE==4)
        for instance in (cached,fresh):instance.sim.phase_pool_selector=os.environ.get('SV_PHASE_POOL_SELECTOR')
        self.addCleanup(cached.close);self.addCleanup(fresh.close)
        report=dict(completed=False,binary_sha256=hashlib.sha256(Path(BINARY).read_bytes()).hexdigest(),
                    precision_profile=PROFILE,policy=POLICY,timing_claim=False,cases=[])
        report_path=Path(os.environ.get('CANONICAL_REPORT','build/canonical-prefix-gates.json'))
        report_path.parent.mkdir(parents=True,exist_ok=True)
        base=[9707,11,1917,0,358,11,9707,1917]*12
        with tempfile.TemporaryDirectory() as directory:
            def compare(name,ids,session='s',workers=8,max_new=4,decoding='chat'):
                shared=common_prefix(cached.cache_ids,ids[:-1])
                expected=cached.canonical_prefix.reuse(shared,len(ids)-1)
                if cached.cache_session!=session or not cached.sim.process or cached.sim.process.poll() is not None:expected=0
                results=[];arrays=[]
                for i,engine in enumerate((cached,fresh)):
                    # B4 profile requires canonical capability; unique sessions
                    # make this reference fresh without bypassing the public API.
                    request_session=(f'{session}-fresh-{len(report["cases"])}' if i==1 else session)
                    request=ChatRequest(request_session,ids,max_new,0,workers=workers,decoding=decoding,seed=42)
                    result=list(engine.stream(request,threading.Event()))[-1]
                    self.assertEqual(result['type'],'done');self.assertIn(result['finish_reason'],('length','eos'))
                    results.append(result);arrays.append(read_dump_logits(engine.sim,Path(directory)/f'{i}.bin'))
                self.assertTrue(all(np.isfinite(a).all() and len(a)==151936 for a in arrays))
                np.testing.assert_array_equal(arrays[0].view(np.uint64),arrays[1].view(np.uint64))
                self.assertEqual(results[0]['token_ids'],results[1]['token_ids'])
                self.assertEqual(cached.cache_ids,fresh.cache_ids)
                self.assertEqual(results[0]['metrics']['reused_prefix_tokens'],expected)
                self.assertEqual(results[1]['metrics']['reused_prefix_tokens'],0)
                self.assertEqual(cached.canonical_prefix.end,(len(ids)-1)//8*8)
                report['cases'].append(dict(name=name,prompt_tokens=len(ids),workers=workers,
                    decoding=decoding,expected_reused=expected,actual_reused=results[0]['metrics']['reused_prefix_tokens'],
                    canonical_end=cached.canonical_prefix.end,output_ids=results[0]['token_ids'],
                    full_logits_bit_identical=True,logits_sha256=hashlib.sha256(arrays[0].tobytes()).hexdigest()))
                report_path.write_text(json.dumps(report,indent=2)+'\n')
            compare('initial',base[:65]);compare('same_after_fp32_decode',base[:65])
            for tail in range(8):compare(f'shrink_grow_tail_{tail}',base[:25+tail])
            compare('tail_becomes_full_block',base[:33])
            for position in (7,8,9,15,16,17):
                compare(f'restore_before_change_{position}',base[:41])
                changed=base[:41];changed[position]=7
                compare(f'change_{position}',changed)
            compare('new_turn_reprocess_generated',base[:49]+cached.cache_ids[-3:]+[358,11])
            compare('shorten_inside_first_block',base[:8],max_new=1,decoding='greedy')
            compare('one_token_prompt',base[:1],max_new=1,decoding='greedy')
            compare('new_session',base[:49],session='new')
            for workers in ((8,) if FIXED_WORKERS else (1,3,17)):compare(f'worker_change_{workers}',base[:49],session='new',workers=workers)
            cached.sim.close() # Simulate process loss without clearing engine metadata.
            compare('process_restart',base[:49],session='new')
            cached.close()
            cancel=threading.Event()
            for event in cached.stream(ChatRequest('s',base[:41],4,0),cancel):
                if event['type']=='prefill':cancel.set()
            self.assertEqual(cached.canonical_prefix.end,0);self.assertIsNone(cached.sim.process)
            compare('after_cancel',base[:41])
            stream=cached.stream(ChatRequest('s',base[:65],4,0),threading.Event())
            for event in stream:
                if event['type']=='prefill':break
            stream.close();self.assertEqual(cached.canonical_prefix.end,0);self.assertIsNone(cached.sim.process)
            compare('after_generator_close',base[:41])
        report['phase_pool_selector']=os.environ.get('SV_PHASE_POOL_SELECTOR')
        report['worker_change_scope']='fixed8; incompatible changes tested separately' if FIXED_WORKERS else '1/3/17 worker changes'
        report['completed']=True
        report['total_reused_tokens']=sum(c['actual_reused'] for c in report['cases'])
        report_path.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':unittest.main()
