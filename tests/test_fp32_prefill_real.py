"""Opt-in routing gates for FP32 prefill; B8 and B1 need not be bit identical."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from tools.rtl_model import RTLModel
from tools.evaluate_projection_quality import read_dump_logits

@unittest.skipUnless(os.environ.get('SV_FP32_PREFILL_BINARY'), 'explicit candidate required')
class FP32PrefillRoutes(unittest.TestCase):
    def test_tails_and_repartitioned_route_replay(self):
        binary=os.environ['SV_FP32_PREFILL_BINARY']
        models=[RTLModel(binary=binary,workers=w,lanes=8,block_size=4,prefill_size=8) for w in (8,3)]
        a,b=models
        prompt=[9707,11,358,525,264,17847,13,198]
        cases=[]
        try:
            for m in models:m.start()
            info=json.loads(a.command('precision'))
            self.assertEqual(info['precision_profile'],'fp32_b1_fp32_prefill')
            self.assertEqual(info['canonical_prefix_policy'],'none')
            with tempfile.TemporaryDirectory() as temp:
                def compare(label):
                    x=read_dump_logits(a,Path(temp)/'a');y=read_dump_logits(b,Path(temp)/'b')
                    self.assertEqual(x.tobytes(),y.tobytes(),label)
                    cases.append(label)
                for count in range(1,9):
                    for m in models:m.reset();m.prefill(prompt)
                    a.prefill(prompt[:count])
                    if count<=4:
                        for token in prompt[:count]:b.token(token,logits=False)
                    else:b.prefill(prompt[:count])
                    self.assertEqual(a.token(40)[:2],b.token(40)[:2])
                    compare(f'count{count}')
                    # Replay the same routing after rollback overwrites all suffix KV.
                    for m in models:
                        m.command('truncate 8');m.prefill(prompt[:count]);m.token(40)
                    compare(f'truncate{count}')
                # Fresh restart preserves the canonical routing of a wide tail.
                a.close();a.start();b.reset()
                for m in models:m.prefill(prompt[:7]);m.token(40)
                compare('restart7')
            Path('build/fp32-prefill-route-gates.json').write_text(json.dumps(dict(binary_sha256=__import__('hashlib').sha256(Path(binary).read_bytes()).hexdigest(),cases=cases,precision=info),indent=2)+'\n')
        finally:
            for m in models:m.close()

if __name__=='__main__':unittest.main()
