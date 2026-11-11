"""Opt-in real-model attention checks; build the isolated candidate first."""
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.rtl_model import RTLModel

BINARY = os.environ.get('SV_ATTENTION_BINARY')


@unittest.skipUnless(BINARY and Path(BINARY).is_file(), 'Set SV_ATTENTION_BINARY to built candidate')
class ParallelAttentionTests(unittest.TestCase):
    def test_head_partitions_full_logits_and_cache_replay(self):
        serial = RTLModel(binary=BINARY, workers=0, lanes=1)
        parallel = RTLModel(binary=BINARY, workers=3, lanes=1)
        try:
            serial.start(); parallel.start()
            with tempfile.TemporaryDirectory() as directory:
                paths = [Path(directory)/'serial.bin', Path(directory)/'parallel.bin']
                def step(token, uniform):
                    arrays, results = [], []
                    for model, path in zip((serial, parallel), paths):
                        results.append(model.token(token, uniform=uniform))
                        model.command(f'dump {path}')
                        arrays.append(np.fromfile(path, dtype=np.float64))
                    self.assertEqual(results[0], results[1])
                    self.assertEqual(arrays[0].size,151936)
                    self.assertTrue(np.isfinite(arrays[0]).all())
                    np.testing.assert_array_equal(*arrays)
                    return arrays[1]
                for workers, token in [(3,9707),(8,11),(17,1917),(1,0)]:
                    parallel.set_workers(workers)
                    step(token,.42)
                for model in (serial,parallel): model.command('truncate 2')
                expected=step(9707,.7)
                for model in (serial,parallel): model.reset()
                step(9707,.42);step(11,.42)
                np.testing.assert_array_equal(expected,step(9707,.7))
                for model in (serial,parallel): model.close();model.start()
                step(9707,.42);step(11,.42)
                np.testing.assert_array_equal(expected,step(9707,.7))
        finally:
            serial.close();parallel.close()


if __name__ == '__main__': unittest.main()
