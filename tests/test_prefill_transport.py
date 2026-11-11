"""Separate prefill capacity must not enlarge verified sampling blocks."""
import unittest
from unittest.mock import Mock, patch
from tools.rtl_model import RTLModel
from tests.test_suffix_chat import engine, cycle, run


class PrefillTransportTests(unittest.TestCase):
    def model(self):
        m=RTLModel.__new__(RTLModel)
        m.prefill_size=8;m.block_size=4
        m.command=Mock(return_value='prefill 8 13')
        return m

    def test_invalid_capacity_pair_rejected_before_model_load(self):
        with self.assertRaises(ValueError):RTLModel(block_size=1,prefill_size=4)
        with patch("tools.rtl_model.prepare_layout",return_value=(None,None)):
            hybrid=RTLModel(block_size=1,prefill_size=8)
            self.assertEqual((hybrid.prefill_size,hybrid.block_size),(8,1))
        with self.assertRaises(ValueError):RTLModel(block_size=4,prefill_size=1)

    def test_wire_and_positions(self):
        m=self.model()
        self.assertEqual(m.prefill(range(8)),[(0,p,0.0) for p in range(6,14)])
        m.command.assert_called_once_with('prefill 8 0 1 2 3 4 5 6 7',None)

    def test_invalid_inputs_do_not_send(self):
        for ids in ([],range(9),[True],[-1],[151936]):
            m=self.model()
            with self.assertRaises(ValueError):m.prefill(ids)
            m.command.assert_not_called()
        with self.assertRaises(ValueError):self.model().block(range(5),logits=True)

    def test_invalid_response_rejected(self):
        for response in ('block 8 13','prefill 7 13','prefill 8 7','prefill 8 4097'):
            m=self.model();m.command.return_value=response
            with self.assertRaises(RuntimeError):m.prefill(range(8))

    def test_wide_prefill_does_not_enlarge_sampled_verification(self):
        a=engine(cycle);b=engine(cycle)
        b.sim.prefill_size=8
        wide=[]
        def prefill(tokens,cancel=None):
            wide.append(list(tokens))
            return [b.sim.token(t,logits=False,cancel=cancel) for t in tokens]
        b.sim.prefill=prefill
        ids=[1,2,3]*6+[1]
        expected=run(a,ids=ids,draft=3,decoding='chat')
        actual=run(b,ids=ids,draft=3,decoding='chat')
        self.assertEqual(actual['token_ids'],expected['token_ids'])
        self.assertEqual(a.cache_ids,b.cache_ids)
        self.assertEqual([len(x) for x in wide],[8,8,2])
        self.assertTrue(all(len(tokens)<=4 for tokens,_,_ in b.sim.blocks))
        self.assertEqual(actual['metrics']['prefill_block_size'],8)
        self.assertEqual(actual['metrics']['effective_max_draft'],3)

if __name__=='__main__':unittest.main()
