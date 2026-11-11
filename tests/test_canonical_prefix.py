"""Route-aware bookkeeping tests; no model arithmetic."""
import json
import threading
import unittest
from unittest.mock import patch
from tools.canonical_prefix import CanonicalPrefix, POLICY
from tools.rtl_chat import RTLChatEngine
from tools.chat_protocol import ChatRequest
from tests.test_hybrid_backend import hybrid, EXPECTED
from tests.test_suffix_chat import run


def target(cached):
    e=hybrid();e.canonical_prefix=CanonicalPrefix() if cached else None
    m=e.sim;m.routes=[]
    old_token,old_prefill,old_reset,old_close,old_command=m.token,m.prefill,m.reset,m.close,m.command
    def token(*a,**k):
        result=old_token(*a,**k);m.routes.append('decode');return result
    def prefill(tokens,**k):
        start=len(m.routes);result=old_prefill(tokens,**k)
        m.routes[start:]=['prefill'+str(len(tokens))]*len(tokens)
        return result
    def reset(*a,**k):m.routes=[];return old_reset(*a,**k)
    def close():m.routes=[];return old_close()
    def command(text,cancel=None):
        if text=='precision':return json.dumps(dict(EXPECTED,canonical_prefix_policy=POLICY))
        result=old_command(text,cancel)
        if text.startswith('truncate '):m.routes=m.routes[:int(text.split()[1])]
        return result
    m.token=token;m.prefill=prefill;m.reset=reset;m.close=close;m.command=command
    return e


class CanonicalPrefixTests(unittest.TestCase):
    def compare(self,cached,fresh,ids,session='s'):
        a=run(cached,ids=ids,max_new=4,decoding='chat',session=session)
        b=run(fresh,ids=ids,max_new=4,decoding='chat',session=session)
        self.assertEqual(a['token_ids'],b['token_ids'])
        self.assertEqual(cached.cache_ids,fresh.cache_ids)
        self.assertEqual(cached.sim.routes,fresh.sim.routes)
        self.assertEqual(cached.canonical_prefix.end,(len(ids)-1)//8*8)
        return a

    def test_all_tail_lengths_growth_shrink_and_matching_decode_ids(self):
        ids=([1,2,3]*20)
        for length in range(1,35):
            cached,fresh=target(True),target(False)
            self.compare(cached,fresh,ids[:26])
            result=self.compare(cached,fresh,ids[:length])
            self.assertEqual(result['metrics']['reused_prefix_tokens'],min(24,length-1)//8*8)
            result=self.compare(cached,fresh,ids[:35])
            self.assertEqual(result['metrics']['reused_prefix_tokens'],(length-1)//8*8)

    def test_mismatch_inside_block_and_multi_turn(self):
        ids=[1,2,3]*15
        for position in (0,7,8,9,15,16,17,23,24,25):
            cached,fresh=target(True),target(False)
            self.compare(cached,fresh,ids[:34])
            changed=ids[:35];changed[position]=7
            result=self.compare(cached,fresh,changed)
            self.assertEqual(result['metrics']['reused_prefix_tokens'],position//8*8)
            self.compare(cached,fresh,changed+[1,2,3,1,2])
            result=self.compare(cached,fresh,changed,session='different')
            self.assertEqual(result['metrics']['reused_prefix_tokens'],0)

    def test_cancel_generator_close_restart_and_bad_ack_clear_boundary(self):
        e=target(True);fresh=target(False);ids=[1,2,3]*11
        self.compare(e,fresh,ids)
        e.close();self.assertEqual(e.canonical_prefix.end,0)
        self.assertEqual(self.compare(e,fresh,ids)['metrics']['reused_prefix_tokens'],0)
        cancel=threading.Event();events=e.stream(ChatRequest('s',ids,4,0),cancel)
        for event in events:
            if event['type']=='start':cancel.set()
        self.assertIsNone(e.sim.process);self.assertEqual(e.canonical_prefix.end,0)
        events=e.stream(ChatRequest('s',ids,4,0),threading.Event())
        for event in events:
            if event['type']=='prefill':break
        events.close();self.assertEqual(e.canonical_prefix.end,0)
        self.assertIsNone(e.sim.process)
        original=e.sim.prefill
        e.sim.prefill=lambda *a,**k: [(0,999,0.0)]*len(a[0])
        with self.assertRaises(RuntimeError):run(e,ids=ids)
        self.assertEqual(e.canonical_prefix.end,0);e.sim.prefill=original

    def test_policy_capability_required_and_default_unchanged(self):
        e=hybrid();e.canonical_prefix=CanonicalPrefix()
        with self.assertRaises(RuntimeError):run(e,ids=[1]*17)
        self.assertEqual(e.sim.prefills,[]);self.assertIsNone(e.sim.process)
        with self.assertRaises(ValueError):RTLChatEngine(canonical_prefix_reuse=True)
        with self.assertRaises(ValueError):RTLChatEngine(canonical_prefix_reuse=1)
        default=target(False)
        for _ in range(2):self.assertEqual(run(default,ids=[1]*17)['metrics']['reused_prefix_tokens'],0)

    def test_context_limit_and_eos_do_not_expand_safe_boundary(self):
        cached,fresh=target(True),target(False)
        ids=[1,2,3]*1364
        self.compare(cached,fresh,ids)
        self.assertEqual(cached.canonical_prefix.end,4088)
        before=list(cached.cache_ids)
        with self.assertRaises(ValueError):run(cached,ids=ids+[1],max_new=4)
        self.assertEqual(before,cached.cache_ids)
        self.assertEqual(cached.canonical_prefix.end,4088)
        cached,fresh=target(True),target(False)
        for model in (cached,fresh):model.sim.prediction=lambda history,uniform:99
        first=self.compare(cached,fresh,[1]*18)
        self.assertEqual(first['finish_reason'],'eos')
        second=self.compare(cached,fresh,list(cached.cache_ids)+[1])
        self.assertEqual(second['metrics']['reused_prefix_tokens'],16)

    def test_helper_epoch_and_ack_invariants(self):
        p=CanonicalPrefix();epoch=object();p.bind(epoch);p.acknowledge(0,8);p.acknowledge(8,3)
        self.assertEqual(p.end,8);self.assertEqual(p.reuse(7,30),0)
        p.bind(epoch);self.assertEqual(p.end,8)
        p.bind(object());self.assertEqual(p.end,0)
        with self.assertRaises(RuntimeError):p.acknowledge(1,8)
        with self.assertRaises(ValueError):p.truncate(1)

if __name__=='__main__':unittest.main()
