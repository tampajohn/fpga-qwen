"""Runtime profile checks for the paired FP32 prefill benchmark."""
import json
import unittest
from unittest.mock import Mock
from tools.benchmark_matched import SVEngine

class PrecisionCapabilities(unittest.TestCase):
    def engine(self, profile, capabilities):
        engine=SVEngine.__new__(SVEngine)
        engine.precision_profile=profile;engine.verification_capacity=4
        engine.lanes=8;engine.metadata={};engine.model=Mock()
        engine.model.command.return_value=json.dumps(capabilities)
        return engine
    def capabilities(self, fp32):
        return dict(precision_profile='fp32_b1_fp32_prefill' if fp32 else 'fp32_b1_fp64_prefill',
                    scalar_projection='fp32',prefill_projection='fp32' if fp32 else 'fp64',
                    attention_kv='fp64',max_verification_block=4,verification_projection='fp32',
                    prefix_reuse_supported=False,canonical_prefix_policy='none' if fp32 else 'fp64_b8_aligned_v1')
    def test_both_profiles_verified_and_lane_state_set(self):
        for fp32 in (False,True):
            c=self.capabilities(fp32);engine=self.engine(c['precision_profile'],c)
            engine.start();engine.model.set_lanes.assert_called_once_with(8)
            self.assertEqual(engine.metadata['verified_runtime_capabilities'],c)
    def test_wrong_precision_capacity_and_cache_policy_rejected(self):
        for key,value in [('prefill_projection','fp64'),('max_verification_block',1),('canonical_prefix_policy','fp64_b8_aligned_v1')]:
            c=self.capabilities(True);c[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                self.engine('fp32_b1_fp32_prefill',c).start()

if __name__=='__main__':unittest.main()
