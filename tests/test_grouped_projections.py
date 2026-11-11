"""Opt-in exact B4 descriptor-group parity with uneven worker partitions."""
import os
from pathlib import Path
import tempfile
import unittest
from tools.benchmark_blocks import BlockProtocolModel,dump_logits,compare_logits

CANDIDATE=os.environ.get('SV_GROUP_BINARY')
REFERENCE=os.environ.get('SV_GROUP_REFERENCE')


@unittest.skipUnless(CANDIDATE and REFERENCE,'Set SV_GROUP_BINARY and SV_GROUP_REFERENCE')
class GroupedProjectionTests(unittest.TestCase):
    def test_uneven_partitions_shared_inputs_and_tail_blocks(self):
        for workers in (1,3,8,17):
            with self.subTest(workers=workers):
                models=[BlockProtocolModel(binary=path,workers=workers,lanes=8) for path in (REFERENCE,CANDIDATE)]
                try:
                    for model in models:model.start()
                    self.assertEqual(models[1].command('groupstats'),'0 0 0 0')
                    with tempfile.TemporaryDirectory() as directory:
                        for tokens in ([9707,11,1917,0],[358,11,9707],[1917],[0,358]):
                            outputs=[model.block(tokens,logits=True,uniforms=[.42]*len(tokens)) for model in models]
                            self.assertEqual(outputs[0],outputs[1])
                            arrays=[dump_logits(model,Path(directory)/f'{i}.bin') for i,model in enumerate(models)]
                            compare_logits(*arrays,0)
                        self.assertEqual(list(map(int,models[1].command('groupstats').split())),
                                         [4*56,4*140,4*84,4*84*workers*1024*4*8])
                finally:
                    for model in models:model.close()


if __name__=='__main__':unittest.main()
