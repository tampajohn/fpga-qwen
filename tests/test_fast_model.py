"""Numerical/sampling gate for the opt-in lane8 build; no speed claims.

Fixed uniforms test the same sampler inputs across reduction orders. Passing this
short fixture does not establish sampled equivalence for every prompt or seed.
"""
import random
import tempfile
from pathlib import Path
import unittest
import os

import numpy as np

from tools.rtl_model import RTLModel, FAST_BINARY as DEFAULT_FAST_BINARY


FAST_BINARY = Path(os.environ.get('SV_TEST_BINARY', DEFAULT_FAST_BINARY))


@unittest.skipUnless(FAST_BINARY.is_file(), 'Build the fast simulator variant first')
class FastModelTests(unittest.TestCase):
    def test_lane8_logits_sampling_truncate_reset_and_restart(self):
        models = [RTLModel(workers=8, binary=FAST_BINARY) for _ in range(2)]
        rng = random.Random(42)
        uniforms = [rng.random() for _ in range(5)]
        # Teacher forcing isolates arithmetic differences from divergent inputs.
        prefix = [9707, 11, 1917, 0]

        def start_models():
            for model, lanes in zip(models, (1, 8)):
                model.start()
                self.assertEqual(model.command(f'lanes {lanes}'), 'configured')

        try:
            start_models()
            with tempfile.TemporaryDirectory() as directory:
                paths = [Path(directory) / f'lanes{lanes}.bin' for lanes in (1, 8)]

                def step(token, uniform):
                    results, logits = [], []
                    for model, path in zip(models, paths):
                        results.append(model.token(token, uniform=uniform))
                        self.assertEqual(model.command(f'dump {path}'), 'dumped')
                        values = np.fromfile(path, dtype=np.float64)
                        self.assertEqual(values.size, 151936)
                        self.assertTrue(np.isfinite(values).all())
                        logits.append(values)
                    self.assertEqual(results[0][:2], results[1][:2],
                                     'Same-uniform sampled token or KV position differs')
                    self.assertEqual(int(np.argmax(logits[0])), int(np.argmax(logits[1])),
                                     'Greedy token differs')
                    np.testing.assert_allclose(logits[0], logits[1], rtol=0, atol=1e-6)
                    return results, logits

                for token, uniform in zip(prefix, uniforms):
                    step(token, uniform)
                for model in models:
                    self.assertEqual(model.command('truncate 2'), 'truncated')
                # Replace the suffix, then feed the shared sampled result back.
                replacement = step(9707, uniforms[2])
                next_token = replacement[0][0][0]
                cached = step(next_token, uniforms[3])
                replay = prefix[:2] + [9707, next_token]

                def check_replay(expected):
                    actual = None
                    for token, uniform in zip(replay, uniforms):
                        actual = step(token, uniform)
                    for index in range(2):
                        self.assertEqual(expected[0][index], actual[0][index])
                        np.testing.assert_array_equal(expected[1][index], actual[1][index])

                for model in models:
                    model.reset()
                check_replay(cached)
                for model in models:
                    model.close()
                start_models()
                check_replay(cached)
        finally:
            for model in models:
                model.close()


if __name__ == '__main__':
    unittest.main()
