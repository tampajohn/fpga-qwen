"""SV top-k/top-p sampling matches an independent NumPy probability oracle."""
import tempfile
import threading
from pathlib import Path
import unittest
import numpy as np
from tools.rtl_model import RTLModel


class SamplingTests(unittest.TestCase):
    def test_sampling_math_and_truncated_history(self):
        model = RTLModel(workers=8)
        history = []
        try:
            model.start()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'logits.bin'
                for token, uniform in [(9707, 0.0), (11, .25), (1917, .5), (9707, .999999)]:
                    history.append(token)
                    chosen, _, _ = model.token(token, uniform=uniform)
                    model.command(f'dump {path}')
                    logits = np.fromfile(path, dtype=np.float64)
                    seen = sorted({i for i in history[-256:] if i<151643})
                    logits[seen] -= 1.5
                    logits /= .7
                    ranked = np.argsort(-logits, kind='stable')[:20]
                    probs = np.exp(logits[ranked]-logits[ranked[0]])
                    kept = np.searchsorted(np.cumsum(probs), probs.sum()*.8, side='left')+1
                    probs = probs[:kept]
                    expected = ranked[np.searchsorted(np.cumsum(probs), uniform*probs.sum(), side='right')]
                    self.assertEqual(chosen, expected)
                model.command('truncate 1')
                result = model.token(11, uniform=.25)
                model.reset()
                model.token(9707, uniform=0)
                self.assertEqual(result, model.token(11, uniform=.25))
        finally:
            model.close()

    def test_concurrent_close(self):
        model = RTLModel(workers=2)
        model.start()
        errors = []
        def close():
            try:
                model.close()
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=close) for _ in range(2)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertEqual(errors, [])
        self.assertIsNone(model.process)


if __name__ == '__main__':
    unittest.main()
