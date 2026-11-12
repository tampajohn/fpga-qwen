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
    def model(self,reference=False,workers=8):
        model=PrefillProtocolModel(binary=REFERENCE if reference else BINARY,workers=workers,lanes=8)
        self.addCleanup(model.close);model.start();return model

    def compare(self,a,b):
        with tempfile.TemporaryDirectory() as directory:
            arrays=[read_dump_logits(m,Path(directory)/str(i))for i,m in enumerate((a,b))]
        result=compare_logits(*arrays,1e-6)
        print(json.dumps(dict(gate=self.id(),**result)),flush=True)
        return result

    def test_serial_and_uneven_worker_partitions(self):
        tokens=[9707,11,1917,0,358,11,9707,0]
        for workers in (0,1,3,17):
            with self.subTest(workers=workers):
                a,b=self.model(True,workers),self.model(False,workers)
                try:
                    prefill(a,tokens,4);b.prefill(tokens)
                    self.assertEqual(a.token(1917)[:2],b.token(1917)[:2])
                    self.compare(a,b)
                finally:
                    a.close();b.close()

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
        exact=os.environ.get('RUN_PREFILL_CAPACITY_EXACT')=='1'
        models=[model,self.model(True)] if exact else [model]
        for start in range(0,4088,8):
            for current in models:self.assertEqual(current.prefill([9707]*8)[-1][1],start+8)
        for current in models:self.assertEqual(current.prefill([9707]*8)[-1][1],4096)
        report=dict(completed=False,fp64_b8_prefill_reached=4096,reference_checked=exact,
                    timing_claim=False,max_causal_position=4095)
        if exact:
            # The prefill wire intentionally has no logits. Explicit verification
            # observes operation5 at the same maximum causal position with a head.
            predictions=[]
            for current in models:
                self.assertEqual(current.command('truncate 4092'),'truncated')
                predictions.append(current.block([9707]*4,logits=True))
            self.assertEqual([r[:2]for r in predictions[0]],[r[:2]for r in predictions[1]])
            self.assertEqual([r[1]for r in predictions[0]],[4093,4094,4095,4096])
            with tempfile.TemporaryDirectory()as directory:
                arrays=[read_dump_logits(current,Path(directory)/str(i))for i,current in enumerate(models)]
            self.assertEqual(arrays[0].tobytes(),arrays[1].tobytes())
            report.update(max_position_fp32_b4_verification_ids_positions=[r[:2]for r in predictions[0]],
                          full_logits_bit_identical=True)
        for current in models:
            self.assertEqual(current.command('truncate 4095'),'truncated')
            self.assertEqual(current.prefill([9707])[-1][1],4096)
            self.assertEqual(current.command('truncate 4092'),'truncated')
            with self.assertRaises(RuntimeError):current.command('prefill 5 9707 9707 9707 9707 9707')
            self.assertEqual(current.process.wait(timeout=2),1)
        report.update(completed=True,valid_single_token_tail=True,overflow_rejected=True)
        if exact:
            output=Path(os.environ.get('PREFILL_CAPACITY_REPORT','build/prefill-capacity-exact.json'))
            output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
