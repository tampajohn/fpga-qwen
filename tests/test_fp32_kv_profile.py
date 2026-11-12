import json
import unittest
from unittest.mock import Mock,patch
from tools.rtl_chat import RTLChatEngine
from tests.test_suffix_chat import engine,cycle,run

CAPABILITY=dict(precision_profile='fp32_b1_fp64_prefill_fp32_kv',scalar_projection='fp32',prefill_projection='fp64',
    verification_projection='fp32',attention_arithmetic='fp64',kv_storage='fp32',max_verification_block=4,
    prefix_reuse_supported=False,canonical_prefix_policy='none')

class FP32KVProfile(unittest.TestCase):
    def test_rejects_cache_or_wrong_segmentation_before_loading(self):
        valid=dict(workers=8,lanes=8,block_size=4,prefill_size=8,reuse_prefix=False,
                   precision_profile='fp32_b1_fp64_prefill_fp32_kv')
        with patch('tools.rtl_chat.RTLModel') as model:
            for bad in (dict(canonical_prefix_reuse=True),dict(reuse_prefix=True),dict(block_size=1),
                        dict(workers=0),dict(lanes=1),dict(prefill_size=4)):
                with self.subTest(bad=bad),self.assertRaises(ValueError):RTLChatEngine(**dict(valid,**bad))
            model.assert_not_called()

    def test_policy_checked_and_same_session_always_reset(self):
        candidate=engine(cycle,4);candidate.sim.prefill_size=8;candidate.reuse_prefix=False
        candidate.precision_profile='fp32_b1_fp64_prefill_fp32_kv'
        original=candidate.sim.command
        candidate.sim.command=Mock(side_effect=lambda command,cancel=None:
            json.dumps(CAPABILITY) if command=='precision' else original(command,cancel))
        a=run(candidate,ids=[1]*17,max_new=4);b=run(candidate,ids=[1]*17,max_new=4)
        self.assertEqual(a['token_ids'],b['token_ids']);self.assertEqual(candidate.sim.resets,2)
        self.assertEqual(b['metrics']['reused_prefix_tokens'],0)
        self.assertEqual(candidate.status()['kv_storage'],'fp32')
        self.assertEqual(candidate.status()['attention_arithmetic'],'fp64')
        for change in (dict(prefill_projection='fp32'),dict(kv_storage='fp64'),dict(attention_arithmetic='fp32'),dict(canonical_prefix_policy='fp64_b8_aligned_v1'),
                       dict(precision_profile='fp32_b1_fp64_prefill')):
            candidate.sim.command=Mock(return_value=json.dumps(dict(CAPABILITY,**change)))
            with self.assertRaises(RuntimeError):candidate.verify_precision(None)
