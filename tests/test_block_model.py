"""Independent opt-in real RTL block correctness gates.

Set SV_BLOCK_BINARY and optionally SV_BLOCK_REFERENCE. Exact is default;
SV_BLOCK_ATOL must explicitly request a tolerance no larger than1e-6.
RUN_BLOCK_CAPACITY=1 enables the expensive real4096-position boundary test.
"""
import os
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np

from tools.benchmark_blocks import BlockProtocolModel,compare_logits,dump_logits,prefill
from tools.rtl_model import SimulationCancelled

BINARY=os.environ.get('SV_BLOCK_BINARY')
REFERENCE=os.environ.get('SV_BLOCK_REFERENCE',BINARY)
ATOL=float(os.environ.get('SV_BLOCK_ATOL','0'))
if not 0<=ATOL<=1e-6:raise ValueError('SV_BLOCK_ATOL must be in0..1e-6')


@unittest.skipUnless(BINARY and Path(BINARY).is_file(),'Set SV_BLOCK_BINARY to real block engine')
class BlockModelTests(unittest.TestCase):
    def model(self,reference=False):
        m=BlockProtocolModel(workers=8,binary=REFERENCE if reference else BINARY,lanes=8)
        self.addCleanup(m.close);m.start();return m

    def assert_logits(self,a,b):
        with tempfile.TemporaryDirectory() as directory:
            values=[dump_logits(m,Path(directory)/f'{i}.bin') for i,m in enumerate((a,b))]
        compare_logits(*values,ATOL)
        return values

    def compare_feed(self,serial,block,tokens,size,uniforms=None):
        for start in range(0,len(tokens),size):
            chunk=tokens[start:start+size]
            chosen=uniforms[start:start+size] if uniforms is not None else None
            expected=[serial.token(token,uniform=None if chosen is None else chosen[i]) for i,token in enumerate(chunk)]
            actual=block.block(chunk,logits=True,uniforms=chosen)
            self.assertEqual([x[:2] for x in expected],[x[:2] for x in actual])
            # Legacy token diagnostics print only six significant digits; gate
            # actual numerical precision using the full binary logits below.
            self.assertTrue(np.isfinite([x[2] for x in actual]).all())
            values=self.assert_logits(serial,block)
            self.assertEqual(actual[-1][2],values[1][actual[-1][0]])

    def test_sizes_tails_greedy_and_sample_uniforms(self):
        serial,block=self.model(True),self.model()
        tokens=[9707,11,1917,0,9707,11,358]
        for size in (1,2,3,4):
            for sampled in (False,True):
                with self.subTest(size=size,sampled=sampled):
                    serial.reset();block.reset()
                    self.compare_feed(serial,block,tokens,size,[.01,.42,.99,.7,.1,.9,.5] if sampled else None)

    def test_later_tokens_do_not_change_earlier_block_predictions(self):
        model=self.model()
        a=model.block([9707,11,1917,0],logits=True,uniforms=[.2,.4,.6,.8])
        model.reset()
        b=model.block([9707,11,1917,358],logits=True,uniforms=[.2,.4,.6,.8])
        self.assertEqual(a[:3],b[:3],'Future KV leaked into preceding block positions')

    def test_prefix_reuse_truncate_replace_reset_and_restart(self):
        serial,block=self.model(True),self.model()
        original=[9707,11,1917,0,358,11,9707,0,1917]
        self.compare_feed(serial,block,original,4)
        for m in (serial,block):self.assertEqual(m.command('truncate 3'),'truncated')
        replacement=[358,0,9707,11,1917,358,0]
        self.compare_feed(serial,block,replacement,3,[.42]*len(replacement))
        cached=self.assert_logits(serial,block)
        for m in (serial,block):m.reset()
        self.compare_feed(serial,block,original[:3]+replacement,4,[.42]*(3+len(replacement)))
        fresh=self.assert_logits(serial,block)
        for a,b in zip(cached,fresh):np.testing.assert_array_equal(a.view(np.uint64),b.view(np.uint64))
        for m in (serial,block):m.close();m.start()
        self.compare_feed(serial,block,[9707,11,1917],2)

    def test_invalid_wire_counts_tokens_and_uniforms_fail(self):
        commands=[
            'block 0 1 0 1.5',
            'block 5 1 0 1.5 1 2 3 4 5 0 0 0 0 0',
            'block 2 1 0 1.5 9707 151936 0 0',
            'block 2 1 1 1.5 9707 11 0.5 1',
            'block 1 1 1 1.5 9707 nan',
        ]
        for command in commands:
            with self.subTest(command=command):
                model=self.model()
                with self.assertRaises(RuntimeError):model.command(command)
                self.assertEqual(model.process.wait(timeout=2),1)
                self.assertIn("Invalid block",Path(model.log.name).read_text())
                model.close()

    def test_inflight_cancellation_discards_process_and_restart_is_fresh(self):
        model=self.model();cancel=threading.Event();timer=threading.Timer(.003,cancel.set)
        timer.start()
        try:
            with self.assertRaises(SimulationCancelled):model.block([9707,11,1917,0],logits=True,cancel=cancel)
        finally:timer.cancel()
        self.assertIsNone(model.process)
        model.start();self.assertEqual(model.token(9707)[1],1)

    @unittest.skipUnless(os.environ.get('RUN_BLOCK_CAPACITY')=='1','Expensive actual4096-position boundary gate')
    def test_real_context_capacity_and_crossing_block_rejection(self):
        model=self.model();prefill(model,[9707]*4092,4)
        out=model.block([9707]*4,logits=False)
        self.assertEqual([x[1] for x in out],[4093,4094,4095,4096])
        self.assertEqual(model.command('truncate 4095'),'truncated')
        with self.assertRaises(RuntimeError):model.block([9707,9707],logits=False)
        self.assertEqual(model.process.wait(timeout=2),1)
        self.assertIn("Invalid block request",Path(model.log.name).read_text())


if __name__=='__main__':unittest.main()
