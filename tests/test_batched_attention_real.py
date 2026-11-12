"""Opt-in exact full-model gates for batching FP64 attention dispatch."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from tools.rtl_model import RTLModel, SimulationCancelled
from tools.evaluate_projection_quality import read_dump_logits

@unittest.skipUnless(os.environ.get('SV_BATCH_ATTENTION_BINARY') and os.environ.get('SV_BATCH_ATTENTION_REFERENCE'), 'explicit artifacts required')
class BatchedAttentionParity(unittest.TestCase):
    def test_causal_tails_contexts_and_rollback(self):
        models=[RTLModel(binary=os.environ[key],workers=8,lanes=8,block_size=4,prefill_size=8)
                for key in ('SV_BATCH_ATTENTION_BINARY','SV_BATCH_ATTENTION_REFERENCE')]
        a,b=models
        seed=[9707,11,358,525,264,17847,13,198]
        try:
            expected=dict(precision_profile='fp32_b1_fp64_prefill',scalar_projection='fp32',
                          prefill_projection='fp64',verification_projection='fp32',attention_kv='fp64',
                          max_verification_block=4,prefix_reuse_supported=False,canonical_prefix_policy='fp64_b8_aligned_v1')
            for m in models:
                m.start();capabilities=json.loads(m.command('precision'))
                for key,value in expected.items():
                    self.assertIs(type(capabilities.get(key)),type(value))
                    self.assertEqual(capabilities[key],value)
            with tempfile.TemporaryDirectory()as temporary:
                def exact(label):
                    x=read_dump_logits(a,Path(temporary)/'a');y=read_dump_logits(b,Path(temporary)/'b')
                    self.assertEqual(x.tobytes(),y.tobytes(),label)
                def token(value):
                    results=[m.token(value)for m in models]
                    self.assertEqual(results[0][:2],results[1][:2])
                for count in range(1,9):
                    for m in models:m.reset();m.prefill(seed);m.prefill(seed[:count])
                    token(40)
                    exact(('tail',count))
                    for m in models:
                        m.command('truncate 8');m.prefill(list(reversed(seed[:count])))
                    token(198)
                    exact(('overwrite',count))
                for length in (85,127,128,129,255,256,452):
                    prompt=(seed*57)[:length]
                    for m in models:
                        m.reset()
                        for i in range(0,length-1,8):m.prefill(prompt[i:min(i+8,length-1)])
                    token(prompt[-1])
                    exact(('context',length))
                    for next_token in (40,1097,264,1296):
                        token(next_token)
                        exact(('continuation',length,next_token))
                for workers in (1,3,17):
                    results=[]
                    for m in models:
                        m.set_workers(workers);m.reset();m.prefill(seed)
                        results.append(m.block([40,1097,264,1296],logits=True,uniforms=[0.031,0.72,0.999,0.43]))
                    self.assertEqual([r[:2]for r in results[0]],[r[:2]for r in results[1]])
                    exact(('sampled-block',workers))
                # Shared prefix must be independent of differently written future
                # tokens in the SAME batched operation, including overwritten KV.
                for mode in ('prefill','verify'):
                    snapshots=[]
                    for poison in (seed,list(reversed(seed))):
                        results=[]
                        for m in models:
                            m.reset();m.prefill(seed)
                            if mode=='prefill':
                                m.prefill([40,1097,264,1296]+poison[:4]);keep=12
                            else:
                                m.block([40,1097]+poison[:2],logits=True);keep=10
                            self.assertEqual(m.command(f'truncate {keep}'),'truncated')
                            results.append(m.token(198))
                        self.assertEqual([r[:2]for r in results],[results[0][:2]]*2)
                        self.assertEqual(results[0][1],keep+1)
                        exact(('future-exclusion',mode,poison[0]))
                        snapshots.append(read_dump_logits(a,Path(temporary)/'future').tobytes())
                    self.assertEqual(snapshots[0],snapshots[1],('causal-prefix',mode))
                # Cancellation must discard partially computed batched attention.
                a.set_workers(8);a.reset()
                cancel=threading.Event();timer=threading.Timer(.001,cancel.set);timer.start()
                try:
                    with self.assertRaises(SimulationCancelled):a.prefill(seed,cancel=cancel)
                finally:timer.cancel();a.close()
                a.start();b.set_workers(8);b.reset()
                self.assertEqual(a.token(40)[:2],b.token(40)[:2])
                exact('cancel-restart')
        finally:
            for m in models:m.close()

if __name__=='__main__':unittest.main()
