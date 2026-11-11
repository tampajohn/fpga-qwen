"""Host orchestration invariants with a causal target double, no model math tests."""
import threading
import unittest
from unittest.mock import patch

from tools.chat_protocol import ChatRequest
from tools.rtl_chat import RTLChatEngine
from tools.rtl_model import RTLModel, SimulationCancelled


class TargetDouble:
    def __init__(self, prediction, block_size=4):
        self.prediction=prediction
        self.block_size=block_size
        self.prefill_size=block_size
        self.workers=8
        self.lanes=8
        self.process=None
        self.history=[]
        self.blocks=[]
        self.truncations=[]
        self.after_block=None
        self.prefills=[]
        self.resets=0

    def start(self,cancel=None):
        self.process=self

    def poll(self):
        return None

    def set_workers(self,workers,cancel=None):
        self.workers=workers

    def close(self):
        self.process=None
        self.history=[]

    def reset(self,cancel=None):
        self.resets+=1
        self.history=[]

    def command(self,command,cancel=None):
        self.check_cancel(cancel)
        kind,length=command.split()
        assert kind=='truncate'
        length=int(length)
        assert 0<=length<=len(self.history)
        self.history=self.history[:length]
        self.truncations.append(length)
        return 'truncated'

    def check_cancel(self,cancel):
        if cancel is not None and cancel.is_set():
            raise SimulationCancelled()

    def token(self,token,logits=True,cancel=None,uniform=None):
        self.check_cancel(cancel)
        self.history.append(token)
        predicted=self.prediction(self.history,uniform) if logits else 0
        return predicted,len(self.history),0.0

    def prefill(self,tokens,cancel=None):
        self.prefills.append(list(tokens))
        return [self.token(t,logits=False,cancel=cancel) for t in tokens]

    def block(self,tokens,logits=False,uniforms=None,cancel=None):
        self.blocks.append((list(tokens),logits,list(uniforms) if uniforms is not None else None))
        result=[self.token(token,logits,cancel,uniform) for token,uniform in
                zip(tokens,uniforms if uniforms is not None else [None]*len(tokens))]
        if self.after_block:
            self.after_block()
        return result


class TokenizerDouble:
    def decode(self,tokens,skip_special_tokens=True):
        return ','.join(map(str,tokens))


def engine(prediction,block_size=4):
    result=RTLChatEngine.__new__(RTLChatEngine)
    result.sim=TargetDouble(prediction,block_size)
    result.tokenizer=TokenizerDouble()
    result.precision_profile="fp64"
    result.precision_verified=False
    result.reuse_prefix=True
    result.canonical_prefix=None
    result.cache_ids=[]
    result.cache_session=None
    result.eos={99}
    return result


def cycle(history,uniform):
    return {1:2,2:3,3:1}.get(history[-1],1)


def run(model,ids=None,max_new=12,draft=0,decoding='greedy',seed=42,session='s'):
    request=ChatRequest(session,ids or [1,2,3,1,2,3,1],max_new,draft,decoding=decoding,seed=seed)
    return list(model.stream(request,threading.Event()))[-1]


class SuffixChatTests(unittest.TestCase):
    def test_hybrid_prefill_tails_reset_and_suffix_rejection(self):
        for tail in range(1,9):
            with self.subTest(tail=tail):
                model=engine(cycle,1)
                model.sim.prefill_size=8
                model.reuse_prefix=False
                ids=[1,2,3]*6
                ids=ids[:8+tail]+[2]
                first=run(model,ids=ids,max_new=3,decoding='chat')
                second=run(model,ids=ids,max_new=3,decoding='chat')
                self.assertEqual(first['token_ids'],second['token_ids'])
                self.assertEqual(model.sim.resets,2)
                self.assertEqual(model.sim.prefills,[ids[:8],ids[8:-1]]*2)
                self.assertEqual(second['metrics']['reused_prefix_tokens'],0)
                self.assertEqual(model.sim.blocks,[])
                self.assertEqual(model.cache_ids[:len(ids)],ids)
                self.assertFalse(model.status()['suffix_supported'])
                self.assertFalse(model.status()['prefix_reuse_enabled'])
                with self.assertRaises(ValueError):run(model,ids=ids,draft=1)
                self.assertEqual(model.sim.resets,2)

    def test_prefill_result_count_mismatch_is_not_silently_zipped(self):
        model=engine(cycle,1)
        model.sim.prefill_size=8
        model.sim.prefill=lambda *args,**kwargs: []
        with self.assertRaises(RuntimeError):run(model)

    def test_real_ngram_drafts_accept_bonus_and_match_sequential(self):
        sequential=engine(cycle,1)
        blocked=engine(cycle)
        expected=run(sequential)
        actual=run(blocked,draft=7)
        self.assertEqual(actual['text'],expected['text'])
        self.assertEqual(blocked.cache_ids,sequential.cache_ids)
        self.assertEqual(blocked.sim.history,blocked.cache_ids)
        self.assertGreater(actual['metrics']['accepted_tokens'],0)
        self.assertGreater(actual['metrics']['tokens_per_round'],1)
        self.assertEqual(actual['metrics']['effective_max_draft'],3)
        self.assertTrue(all(len(tokens)<=4 for tokens,_,_ in blocked.sim.blocks))
        self.assertTrue(any(len(tokens)==4 and not logits for tokens,logits,_ in blocked.sim.blocks))

    def test_sampled_rejections_preserve_rng_sequence_and_history(self):
        def sampled(history,uniform):
            return 7 if uniform is not None and uniform<.4 else cycle(history,uniform)
        baseline=engine(sampled,1)
        blocked=engine(sampled)
        with patch('tools.rtl_chat.SuffixDrafter.propose',return_value=[2,3,1]):
            # Fixed drafts deliberately cause rejection; proposals are never emitted blindly.
            expected=run(baseline,max_new=16,decoding='chat')
            actual=run(blocked,max_new=16,draft=3,decoding='chat')
        self.assertEqual(actual['text'],expected['text'])
        self.assertEqual(actual['metrics']['output_tokens'],16)
        self.assertEqual(baseline.sim.blocks,[])
        self.assertEqual(blocked.cache_ids,baseline.cache_ids)
        self.assertEqual(blocked.sim.history,baseline.sim.history)
        self.assertGreater(actual['metrics']['proposed_tokens'],actual['metrics']['accepted_tokens'])
        self.assertGreater(actual['metrics']['verified_tokens'],actual['metrics']['output_tokens'])
        self.assertGreater(len(blocked.sim.truncations),0)

    def test_eos_within_accepted_draft_truncates_unconsumed_suffix(self):
        target=lambda history,uniform: {1:2,2:99}.get(history[-1],3)
        blocked=engine(target)
        with patch('tools.rtl_chat.SuffixDrafter.propose',return_value=[2,99,3]):
            result=run(blocked,ids=[1],max_new=4,draft=3)
        self.assertEqual(result['text'],'2,99')
        self.assertEqual(result['finish_reason'],'eos')
        self.assertEqual(blocked.cache_ids,[1,2])
        self.assertEqual(blocked.sim.history,[1,2])
        self.assertEqual(result['metrics']['accepted_tokens'],2)

    def test_output_budget_and_context_boundary(self):
        model=engine(cycle)
        result=run(model,ids=[1]*4094,max_new=2,draft=7)
        self.assertEqual(result['metrics']['output_tokens'],2)
        self.assertEqual(len(model.cache_ids),4095)
        self.assertTrue(all(len(tokens)<=4 for tokens,_,_ in model.sim.blocks))
        with self.assertRaises(ValueError):
            run(model,ids=[1]*4095,max_new=2,draft=7)

    def test_prefix_reuse_after_rejection_and_new_session(self):
        model=engine(cycle)
        first=run(model,max_new=5,draft=3)
        ids=[1,2,3,1,2,3,1]+[int(t) for t in first['text'].split(',')]+[2]
        cached=run(model,ids=ids,max_new=5,draft=3)
        fresh=run(engine(cycle),ids=ids,max_new=5,draft=3,session='fresh')
        self.assertEqual(cached['text'],fresh['text'])
        self.assertGreater(cached['metrics']['reused_prefix_tokens'],0)
        restarted=run(model,ids=[3,1],max_new=2,draft=3,session='different')
        self.assertEqual(restarted['metrics']['reused_prefix_tokens'],0)

    def test_cancel_after_committed_block_exposes_no_unverified_output(self):
        model=engine(cycle)
        cancel=threading.Event()
        model.sim.after_block=cancel.set
        request=ChatRequest('s',[1,2,3,1],8,3)
        result=list(model.stream(request,cancel))[-1]
        self.assertEqual(result['finish_reason'],'stopped')
        self.assertEqual(result['text'],'')
        self.assertIsNone(model.sim.process)
        self.assertEqual(model.cache_ids,[])

    def test_generator_disconnect_clears_partial_session(self):
        model=engine(cycle)
        stream=model.stream(ChatRequest('s',[1,2,3,1],8,3),threading.Event())
        next(stream);next(stream);next(stream)
        stream.close()
        self.assertEqual(model.cache_ids,[])
        self.assertIsNone(model.sim.process)


class BlockTransportTests(unittest.TestCase):
    def test_supported_prefill_decode_pairs(self):
        with patch('tools.rtl_model.prepare_layout',return_value=(None,None)):
            for prefill,block in ((1,1),(4,4),(8,4),(8,1)):
                model=RTLModel(prefill_size=prefill,block_size=block)
                self.assertEqual((model.prefill_size,model.block_size),(prefill,block))
            with self.assertRaises(ValueError):RTLModel(prefill_size=4,block_size=1)

    def test_wire_and_response_validation(self):
        model=RTLModel.__new__(RTLModel)
        model.block_size=4
        calls=[]
        model.command=lambda command,cancel: calls.append(command) or 'block 2 2 11 0.5 3 12 0.25'
        self.assertEqual(model.block([1,2],logits=True,uniforms=[.25,.5]),[(2,11,.5),(3,12,.25)])
        self.assertEqual(calls[0],'block 2 1 1 1.5 1 2 0.25 0.5')
        for tokens in ([],[1]*5,[True],[-1],[151936]):
            with self.assertRaises(ValueError): model.block(tokens)
        with self.assertRaises(ValueError): model.block([1,2],uniforms=[.5])
        model.command=lambda command,cancel: 'block 2 1 2 3'
        with self.assertRaises(RuntimeError): model.block([1,2])
        model.block_size=1
        with self.assertRaises(ValueError): model.block([1])


if __name__=='__main__':
    unittest.main()
