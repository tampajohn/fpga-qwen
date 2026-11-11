"""Full-model KV layout gate across truncation, stale suffixes, and reset.

Opt-in: set SV_TEST_BINARY and SV_REFERENCE_BINARY to independently built models.
Uses 133 original positions, truncates to63, replaces80tokens across127/128, then
replays from reset. This covers every layer/head through inference, but is not a
4096-token capacity-boundary test. No GPU or tensor framework is used.
"""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.benchmark_matched import make_prompt
from tools.chat_protocol import ChatProtocol
from tools.experiment import locate_model
from tools.rtl_model import RTLModel

CANDIDATE = os.environ.get('SV_TEST_BINARY')
REFERENCE = os.environ.get('SV_REFERENCE_BINARY')


@unittest.skipUnless(CANDIDATE and REFERENCE, 'Set both SV_TEST_BINARY and SV_REFERENCE_BINARY')
class KVLayoutTests(unittest.TestCase):
    def test_long_suffix_replacement_and_reset_match_reference(self):
        checkpoint = locate_model(None)
        prompt = make_prompt(ChatProtocol(checkpoint), 'medium')
        original = prompt[:133]
        replacement = prompt[80:160]
        self.assertEqual((len(original),len(replacement)),(133,80))
        models = [RTLModel(checkpoint,workers=8,binary=path,lanes=lanes) for path,lanes in [
            (REFERENCE,int(os.environ.get('SV_REFERENCE_LANES','8'))),
            (CANDIDATE,int(os.environ.get('SV_TEST_LANES','8')))]]
        try:
            for model in models:
                model.start()
            with tempfile.TemporaryDirectory() as directory:
                paths = [Path(directory)/f'logits-{index}.bin' for index in range(2)]

                def feed(tokens, start, checkpoints):
                    snapshots = {}
                    for position,token in enumerate(tokens,start+1):
                        outputs = [model.token(token,logits=position in checkpoints) for model in models]
                        self.assertTrue(all(output[1]==position for output in outputs))
                        if position not in checkpoints:
                            continue
                        logits = []
                        for model,path in zip(models,paths):
                            self.assertEqual(model.command(f'dump {path}'),'dumped')
                            values = np.fromfile(path,dtype=np.float64)
                            self.assertEqual(values.size,151936)
                            self.assertTrue(np.isfinite(values).all())
                            logits.append(values)
                        self.assertEqual(outputs[0][0],outputs[1][0],f'Greedy choice differs at position{position}')
                        np.testing.assert_allclose(logits[0],logits[1],rtol=0,atol=1e-6,
                                                   err_msg=f'KV layout mismatch at position{position}')
                        snapshots[position] = logits
                    return snapshots

                feed(original,0,{63,64,127,128,133})
                for model in models:
                    self.assertEqual(model.command('truncate 63'),'truncated')
                checkpoints = {64,65,127,128,143}
                cached = feed(replacement,63,checkpoints)
                for model in models:
                    model.reset()
                replayed = feed(original[:63]+replacement,0,checkpoints)
                for position in checkpoints:
                    for index in range(2):
                        # Reset must hide stale future KV and suffix replacement must
                        # be identical to fresh causal evaluation within each binary.
                        np.testing.assert_array_equal(cached[position][index].view(np.uint64),
                                                      replayed[position][index].view(np.uint64),
                                                      err_msg=f'Cached/fresh mismatch at position{position}, engine{index}')
        finally:
            for model in models:
                model.close()


if __name__=='__main__':
    unittest.main()
