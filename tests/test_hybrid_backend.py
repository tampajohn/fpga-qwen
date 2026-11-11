"""Opt-in browser backend precision, routing and lifecycle safeguards."""
import json
import threading
import unittest
from unittest.mock import Mock, patch
from tools.chat_backends import BackendRouter
from tools.chat_protocol import ChatRequest
from tools.rtl_chat import RTLChatEngine
from tests.test_suffix_chat import engine, cycle, run

EXPECTED=dict(precision_profile='fp32_b1_fp64_prefill',scalar_projection='fp32',
              prefill_projection='fp64',attention_kv='fp64',
              max_verification_block=1,prefix_reuse_supported=False)


def hybrid():
    result=engine(cycle,1)
    result.sim.prefill_size=8
    result.reuse_prefix=False
    result.precision_profile='fp32_b1_fp64_prefill'
    original=result.sim.command
    result.sim.command=Mock(side_effect=lambda command,cancel=None:
        json.dumps(EXPECTED) if command=='precision' else original(command,cancel))
    return result


class HybridBackendTests(unittest.TestCase):
    def test_constructor_rejects_unsafe_profile_before_model_load(self):
        valid=dict(workers=8,lanes=8,block_size=1,prefill_size=8,reuse_prefix=False,
                   precision_profile='fp32_b1_fp64_prefill')
        with patch('tools.rtl_chat.RTLModel') as model:
            for bad in [dict(workers=0),dict(lanes=1),dict(block_size=4),dict(prefill_size=4),
                        dict(reuse_prefix=True),dict(precision_profile='arbitrary')]:
                with self.subTest(bad=bad),self.assertRaises(ValueError):
                    RTLChatEngine(**dict(valid,**bad))
            model.assert_not_called()

    def test_runtime_precision_is_checked_each_request_and_restart(self):
        candidate=hybrid()
        a=run(candidate,ids=[1]*10,max_new=3)
        b=run(candidate,ids=[1]*10,max_new=3)
        candidate.close()
        c=run(candidate,ids=[1]*10,max_new=3)
        self.assertEqual(a['token_ids'],b['token_ids'])
        self.assertEqual(b['token_ids'],c['token_ids'])
        self.assertEqual(candidate.sim.command.call_count,3)
        self.assertEqual(candidate.sim.resets,3)
        self.assertTrue(candidate.status()['precision_verified'])
        self.assertIn('FP32 scalar',candidate.status()['precision'])
        self.assertEqual(c['metrics']['reused_prefix_tokens'],0)
        self.assertFalse(candidate.status()['suffix_supported'])

    def test_runtime_mismatch_fails_before_prefill_and_closes_process(self):
        for capabilities in [{},dict(EXPECTED,scalar_projection='fp64'),
                             dict(EXPECTED,prefix_reuse_supported=0),[],None]:
            with self.subTest(capabilities=capabilities):
                candidate=hybrid()
                candidate.sim.command=Mock(return_value=json.dumps(capabilities))
                with self.assertRaises(RuntimeError):run(candidate,ids=[1]*10)
                self.assertEqual(candidate.sim.prefills,[])
                self.assertIsNone(candidate.sim.process)
                self.assertFalse(candidate.precision_verified)

    def test_router_default_switching_validation_and_no_http_binary_override(self):
        reference=engine(cycle,4)
        candidate=hybrid()
        for model in [reference,candidate]:
            model.prepare=lambda payload: ChatRequest('s',[1,2,3,1],2,payload.get('max_draft',0))
        router=BackendRouter(reference,candidate)
        self.addCleanup(router.close)
        self.assertEqual(router.active,'rtl')
        self.assertEqual(router.catalog()[0]['id'],'rtl')
        self.assertEqual(router.catalog()[1]['min_workers'],1)
        for bad in [dict(simulation_workers=0),dict(max_draft=1),dict(backend='/tmp/other')]:
            with self.assertRaises(ValueError):router.prepare(dict(backend='rtl-fp32',**bad) if 'backend' not in bad else bad)
        request=router.prepare(dict(backend='rtl-fp32',simulation_workers=2,binary='/tmp/untrusted',sv_binary='/tmp/untrusted'))
        reference.sim.start()
        result=list(router.stream(request,threading.Event()))[-1]
        self.assertEqual(result['type'],'done')
        self.assertIsNone(reference.sim.process)
        self.assertEqual(router.status()['precision_profile'],'fp32_b1_fp64_prefill')
        self.assertFalse(router.status()['prefix_reuse_enabled'])
        list(router.stream(router.prepare(dict(backend='rtl')),threading.Event()))
        self.assertIsNone(candidate.sim.process)
        self.assertEqual(router.status()['precision_profile'],'fp64')
        self.assertTrue(router.status()['suffix_supported'])

    def test_uninstalled_hybrid_is_unavailable(self):
        router=BackendRouter(engine(cycle))
        self.addCleanup(router.close)
        self.assertFalse(next(item for item in router.catalog() if item['id']=='rtl-fp32')['available'])
        with self.assertRaises(ValueError):router.prepare(dict(backend='rtl-fp32'))


if __name__=='__main__':unittest.main()
