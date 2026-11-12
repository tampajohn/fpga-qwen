"""Direct dump diagnostics preserve worker pools after benchmark warmup."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
from compare_batched_attention import read_dump_logits

class DirectDump(unittest.TestCase):
    def test_only_dump_command_and_exact_shape(self):
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'logits'
            calls=[]
            class Model:
                def command(self,command):
                    calls.append(command);np.arange(151936,dtype=np.float64).tofile(path);return 'dumped'
            values=read_dump_logits(Model(),path)
            self.assertEqual(len(values),151936)
            self.assertEqual(calls,[f'dump {path}'])
    def test_truncated_and_nonfinite_dump_rejected(self):
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'logits'
            for data in (np.zeros(3),np.full(151936,np.nan)):
                class Model:
                    def command(self,command):data.tofile(path);return 'dumped'
                with self.assertRaises(ValueError):read_dump_logits(Model(),path)

if __name__=='__main__':unittest.main()
