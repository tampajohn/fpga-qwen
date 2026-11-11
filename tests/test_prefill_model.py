"""Independent B8 protocol/causality gates; explicit 1e-6 maximum error."""
import os
import json
from pathlib import Path
import tempfile
import threading
import unittest

from tools.prefill_protocol import PrefillProtocolModel
from tools.evaluate_projection_quality import read_dump_logits
from tools.benchmark_blocks import compare_logits,prefill
from tools.benchmark_matched import make_prompt
from tools.chat_protocol import ChatProtocol
from tools.rtl_model import SimulationCancelled

BINARY=os.environ.get('SV_PREFILL_BINARY')
REFERENCE=os.environ.get('SV_PREFILL_REFERENCE')

@unittest.skipUnless(BINARY and REFERENCE,'Set SV_PREFILL_BINARY and SV_PREFILL_REFERENCE')
class PrefillModelTests(unittest.TestCase):
    def model(self,reference=False):
        model=PrefillProtocolModel(binary=REFERENCE if reference else BINARY,workers=8,lanes=8)
        self.addCleanup(model.close);model.start();return model

    def compare(self,a,b):
        with tempfile.TemporaryDirectory() as directory:
            arrays=[read_dump_logits(m,Path(directory)/str(i))for i,m in enumerate((a,b))]
        result=compare_logits(*arrays,1e-6)
        print(json.dumps(dict(gate=self.id(),**result)),flush=True)
        return result

    def test_sizes_tails_and_truncated_prefix_causality(self):
        a,b=self.model(True),self.model()
        tokens=[9707,11,1917,0,358,11,9707,0,1917,358,11]
        for size in range(1,9):
            with self.subTest(size=size):
                a.reset();b.reset();prefill(a,tokens,4)
                for start in range(0,len(tokens),size):
                    out=b.prefill(tokens[start:start+size])
                    self.assertEqual(out[-1][1],min(start+size,len(tokens)))
                self.assertEqual(a.token(9707)[:2],b.token(9707)[:2]);self.compare(a,b)
                for m in (a,b):self.assertEqual(m.command('truncate 3'),'truncated')
                replacement=[358,1917,11,0,9707,358,11,0]
                prefill(a,replacement,4);b.prefill(replacement)
                self.assertEqual(a.token(1917)[:2],b.token(1917)[:2]);self.compare(a,b)
        # Fresh replay must agree with the retained prefix after future KV removal.
        a.reset();prefill(a,tokens[:3]+replacement,4);a.token(1917);self.compare(a,b)

    def test_short_long_continuation_and_sampled_block(self):
        a,b=self.model(True),self.model();protocol=ChatProtocol(a.model)
        for profile in ('short','long'):
            tokens=make_prompt(protocol,profile)
            a.reset();b.reset();prefill(a,tokens[:-1],4)
            for start in range(0,len(tokens)-1,8):b.prefill(tokens[:-1][start:start+8])
            pending=tokens[-1]
            for _ in range(8):
                x,y=a.token(pending),b.token(pending)
                self.assertEqual(x[:2],y[:2]);self.compare(a,b);pending=x[0]
            x=a.block([9707,11,1917,0],logits=True,uniforms=[.01,.42,.7,.99])
            y=b.block([9707,11,1917,0],logits=True,uniforms=[.01,.42,.7,.99])
            self.assertEqual([v[:2]for v in x],[v[:2]for v in y]);self.compare(a,b)

    def test_invalid_wire_and_cancel_restart(self):
        for command in ('prefill 0','prefill 9 1 2 3 4 5 6 7 8 9','prefill 2 9707 151936','prefill 1 -1'):
            model=self.model()
            with self.assertRaises(RuntimeError):model.command(command)
            self.assertEqual(model.process.wait(timeout=2),1);model.close()
        model=self.model();cancel=threading.Event();timer=threading.Timer(.003,cancel.set);timer.start()
        try:
            with self.assertRaises(SimulationCancelled):model.prefill([9707]*8,cancel)
        finally:timer.cancel()
        model.close();model.start();a=self.model(True)
        self.assertEqual(model.token(9707)[:2],a.token(9707)[:2]);self.compare(a,model)

    @unittest.skipUnless(os.environ.get('RUN_PREFILL_CAPACITY')=='1','Explicit expensive actual 4096 boundary gate')
    def test_actual_capacity_boundary(self):
        model=self.model()
        for start in range(0,4088,8):self.assertEqual(model.prefill([9707]*8)[-1][1],start+8)
        self.assertEqual(model.prefill([9707]*7)[-1][1],4095)
        self.assertEqual(model.prefill([9707])[-1][1],4096)
        with self.assertRaises(RuntimeError):model.command('prefill 1 9707')
        self.assertEqual(model.process.wait(timeout=2),1)
