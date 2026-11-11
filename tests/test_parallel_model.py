"""Verify partition boundaries, runtime resizing, and cache correctness bitwise."""
import tempfile
from pathlib import Path
import unittest
from tools.rtl_model import RTLModel


class ParallelModelTests(unittest.TestCase):
    def test_serial_and_resized_workers_have_identical_logits(self):
        serial, parallel = RTLModel(workers=0), RTLModel(workers=1)
        try:
            serial.start(); parallel.start()
            with tempfile.TemporaryDirectory() as directory:
                left, right = Path(directory)/'serial.bin', Path(directory)/'parallel.bin'
                # Three workers exercises uneven output-row partitions.
                for workers, token in [(1, 9707), (3, 11), (8, 1917), (0, 0), (4, 9707)]:
                    parallel.set_workers(workers)
                    expected, actual = serial.token(token), parallel.token(token)
                    self.assertEqual(expected[:2], actual[:2])
                    serial.command(f'dump {left}')
                    parallel.command(f'dump {right}')
                    self.assertEqual(left.read_bytes(), right.read_bytes(), f'{workers} workers')
                for model in (serial, parallel):
                    self.assertEqual(model.command('truncate 1'), 'truncated')
                self.assertEqual(serial.token(1917), parallel.token(1917))
                for model in (serial, parallel):
                    model.reset()
                self.assertEqual(serial.token(9707), parallel.token(9707))
        finally:
            serial.close(); parallel.close()

    def test_worker_validation(self):
        for workers in [-1, 33, True, '4']:
            with self.assertRaises(ValueError):
                RTLModel(workers=workers)


if __name__ == '__main__':
    unittest.main()
