"""Opt-in exact scalar descriptor grouping gates, including uneven partitions."""
import os
from pathlib import Path
import tempfile
import unittest
from tools.rtl_model import RTLModel
from tools.chat_protocol import ChatProtocol
from tools.benchmark_matched import make_prompt
from tools.evaluate_projection_quality import read_dump_logits
from tools.benchmark_blocks import compare_logits

@unittest.skipUnless(os.environ.get('SV_SCALAR_GROUP_BINARY'),'Set scalar group binary/reference')
class ScalarGroupTests(unittest.TestCase):
    def models(self,workers=8):
        models=[RTLModel(binary=os.environ[k],workers=workers,lanes=8,block_size=4,prefill_size=8)
                for k in ('SV_SCALAR_GROUP_REFERENCE','SV_SCALAR_GROUP_BINARY')]
        for model in models:self.addCleanup(model.close);model.start()
        return models

    def compare(self,models):
        with tempfile.TemporaryDirectory() as directory:
            arrays=[read_dump_logits(model,Path(directory)/str(i))for i,model in enumerate(models)]
        compare_logits(*arrays,0)

    def test_scalar_stats_samples_and_uneven_partitions(self):
        for workers in (1,3,8,17):
            a,b=self.models(workers)
            for token in (9707,11):
                self.assertEqual(a.token(token,uniform=.42),b.token(token,uniform=.42))
                self.assertEqual(list(map(int,b.command('groupstats').split())),[56,140,84,84*workers*1024*8])
                self.compare((a,b))
            for m in (a,b):m.reset()
            self.assertEqual(a.block([9707,11,1917],logits=True),b.block([9707,11,1917],logits=True));self.compare((a,b))
            a.close();b.close()

    def test_short_long_full_logits(self):
        a,b=self.models();protocol=ChatProtocol(a.model)
        for profile in ('short','long'):
            prompt=make_prompt(protocol,profile)
            for model in (a,b):
                model.reset()
                for start in range(0,len(prompt)-1,8):model.prefill(prompt[:-1][start:start+8])
            pending=prompt[-1]
            for _ in range(8):
                x,y=a.token(pending),b.token(pending);self.assertEqual(x,y);self.compare((a,b));pending=x[0]
