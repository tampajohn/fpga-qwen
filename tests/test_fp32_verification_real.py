"""Opt-in exact verification-route gates; no host model arithmetic."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from tools.rtl_model import RTLModel
from tools.evaluate_projection_quality import read_dump_logits


@unittest.skipUnless(os.environ.get('SV_FP32_VERIFY_BINARY'), 'explicit experimental binary required')
class VerificationParity(unittest.TestCase):
    def test_block_matches_sequential_and_preserves_prefill(self):
        binary=os.environ['SV_FP32_VERIFY_BINARY']
        models=[RTLModel(binary=binary,workers=8,lanes=8,block_size=4,prefill_size=8) for _ in range(2)]
        block,serial=models
        cases=[]
        try:
            for m in models:m.start()
            self.assertEqual(json.loads(block.command('precision'))['max_verification_block'],4)
            with tempfile.TemporaryDirectory() as temporary:
                for tail in (0,1,2,3,4,7):
                    for count in (1,2,3,4):
                        for m in models:
                            m.reset()
                            m.prefill([9707,11,358,525,264,17847,13,198])
                            if tail:m.prefill([791,374,264,1296,315,279,1519][:tail])
                        tokens=[40,1097,264,1296][:count]
                        uniforms=[0.031,0.72,0.999,0.43][:count]
                        expected=[serial.token(token,uniform=u) for token,u in zip(tokens,uniforms)]
                        actual=block.block(tokens,logits=True,uniforms=uniforms)
                        self.assertEqual([x[:2] for x in actual],[x[:2] for x in expected])
                        a=read_dump_logits(block,Path(temporary)/'block')
                        b=read_dump_logits(serial,Path(temporary)/'serial')
                        self.assertEqual(a.tobytes(),b.tobytes(),(tail,count))
                        # Continued scalar decode checks every layer's retained KV.
                        self.assertEqual(block.token(198)[:2],serial.token(198)[:2])
                        a=read_dump_logits(block,Path(temporary)/'block')
                        b=read_dump_logits(serial,Path(temporary)/'serial')
                        self.assertEqual(a.tobytes(),b.tobytes(),('continuation',tail,count))
                        cases.append(dict(prefill_tail=tail,count=count,exact=True))
            path=Path('build/fp32-verification-gates.json');path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(dict(cases=cases),indent=2)+'\n')
        finally:
            for m in models:m.close()

    @unittest.skipUnless(os.environ.get('SV_FP32_REFERENCE_BINARY'), 'explicit B1 reference required')
    def test_prefill_route_unchanged_and_partition_replay(self):
        candidate=RTLModel(binary=os.environ['SV_FP32_VERIFY_BINARY'],workers=3,lanes=8,block_size=4,prefill_size=8)
        reference=RTLModel(binary=os.environ['SV_FP32_REFERENCE_BINARY'],workers=3,lanes=8,block_size=1,prefill_size=8)
        models=(candidate,reference)
        try:
            for m in models:m.start()
            with tempfile.TemporaryDirectory() as temporary:
                for size in (1,2,3,4,5,6,7,8,17,85,452):
                    prompt=([9707,11,358,525,264,17847,13,198]*57)[:size]
                    for m in models:
                        m.reset()
                        for start in range(0,len(prompt),8):m.prefill(prompt[start:start+8])
                        m.token(40)
                    self.assertEqual(read_dump_logits(candidate,Path(temporary)/'a').tobytes(),
                                     read_dump_logits(reference,Path(temporary)/'b').tobytes(),size)
                # Repartition and replay verification across a previously written suffix.
                for workers in (1,3,17):
                    for m in models:
                        m.set_workers(workers);m.reset();m.prefill(prompt[:8])
                    for count in (4,1,3,2):
                        tokens=[40,1097,264,1296][:count]
                        result=candidate.block(tokens,logits=True)
                        expected=[reference.token(t) for t in tokens]
                        self.assertEqual([v[:2] for v in result],[v[:2] for v in expected])
                        self.assertEqual(read_dump_logits(candidate,Path(temporary)/'a').tobytes(),
                                         read_dump_logits(reference,Path(temporary)/'b').tobytes())
                        for m in models:m.command('truncate 8')
        finally:
            for m in models:m.close()

if __name__=='__main__':unittest.main()
