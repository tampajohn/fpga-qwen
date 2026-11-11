"""The dump acknowledgement must make every logit byte available immediately."""
import os
from pathlib import Path
import tempfile
import unittest

from tools.rtl_model import FAST_BINARY, RTLModel

BINARY = Path(os.environ.get('SV_TEST_BINARY', FAST_BINARY))


@unittest.skipUnless(BINARY.is_file(), 'Build an SV executable first')
class LogitDumpTests(unittest.TestCase):
    def test_acknowledgement_publishes_complete_file(self):
        model = RTLModel(binary=BINARY, workers=1, lanes=8)
        try:
            model.start()
            model.token(9707)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'logits.bin'
                original = None
                for _ in range(128):
                    # No sleep, second command, or retry: the ack is the barrier.
                    self.assertEqual(model.command(f'dump {path}'), 'dumped')
                    contents = path.read_bytes()
                    self.assertEqual(len(contents), 151936 * 8)
                    if original is None:
                        original = contents
                    else:
                        self.assertEqual(contents, original)
        finally:
            model.close()


if __name__ == '__main__':
    unittest.main()
