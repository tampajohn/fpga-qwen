"""Unchecked projection reads still reject an invalid full span before dispatch."""
from pathlib import Path
import subprocess
import unittest
from tools.rtl_model import prepare_layout
from tools.experiment import ROOT


class ProjectionSpanTests(unittest.TestCase):
    def test_invalid_span_is_rejected_before_worker_reads(self):
        binary = ROOT/'build/variants/scalar-O3-lto-unchecked/full/Vqwen_behavioral'
        if not binary.is_file():
            self.skipTest('Build the isolated unchecked projection candidate first')
        model, layout = prepare_layout(None)
        offsets = [int(value) for value in layout.read_text().split()]
        slot = max(range(len(offsets)), key=offsets.__getitem__)
        result = subprocess.run([str(binary), str(model/'model.safetensors'), str(layout), '1'],
                                input=f'bench {slot} 151936 3072 1\n', text=True,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Projection weight span out of bounds', result.stderr)
        self.assertEqual(result.stdout.strip(), 'ready')


if __name__ == '__main__':
    unittest.main()
