"""Opt-in reassociated attention comparison and deterministic KV replay."""
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.rtl_model import RTLModel
from tools.evaluate_projection_quality import read_dump_logits

CANDIDATE=os.environ.get('SV_DOT_BINARY')
REFERENCE=os.environ.get('SV_DOT_REFERENCE')

@unittest.skipUnless(CANDIDATE and REFERENCE,'Set SV_DOT_BINARY and SV_DOT_REFERENCE')
class AttentionDotTests(unittest.TestCase):
    def test_partitions_sampling_and_cache_replay(self):
        models=[RTLModel(binary=p,workers=1,lanes=8,block_size=4,prefill_size=8) for p in (REFERENCE,CANDIDATE)]
        try:
            for m in models:m.start()
            with tempfile.TemporaryDirectory() as directory:
                def step(token,uniform):
                    outputs=[m.token(token,uniform=uniform) for m in models]
                    self.assertEqual(outputs[0][:2],outputs[1][:2])
                    arrays=[read_dump_logits(m,Path(directory)/str(i)) for i,m in enumerate(models)]
                    for a in arrays:
                        self.assertEqual(len(a),151936)
                        self.assertTrue(np.isfinite(a).all())
                    self.assertLessEqual(float(np.max(np.abs(arrays[0]-arrays[1]))),1e-6)
                    return arrays[1]
                for workers,token in [(1,9707),(3,11),(8,1917),(17,0)]:
                    for m in models:m.set_workers(workers)
                    step(token,.42)
                for m in models:self.assertEqual(m.command('truncate 2'),'truncated')
                expected=step(9707,.7)
                for m in models:m.reset()
                step(9707,.42);step(11,.42)
                np.testing.assert_array_equal(expected.view(np.uint64),step(9707,.7).view(np.uint64))
                for m in models:m.close();m.start()
                step(9707,.42);step(11,.42)
                np.testing.assert_array_equal(expected.view(np.uint64),step(9707,.7).view(np.uint64))
        finally:
            for m in models:m.close()

if __name__=='__main__':unittest.main()
