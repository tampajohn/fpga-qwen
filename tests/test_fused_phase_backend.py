"""Opt-in fused phase selection; no neural inference."""
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
from tools.chat_backends import BackendRouter
from tools.chat_protocol import ChatRequest
from tools.canonical_prefix import FMA_POLICY
from tools.phase_chat import FusedPhasePoolChatEngine
from tools.rtl_model import FMA_PHASE_BINARY
from tests.test_phase_backend import phase_engine
from tests.test_fma_prefill_profile import CAPABILITY
from tests.test_suffix_chat import run,engine,cycle


def fused():
    result=phase_engine();result.__class__=FusedPhasePoolChatEngine
    result.precision_profile='fp32_b1_fp32_fma_prefill'
    original=result.sim.command
    result.sim.command=Mock(side_effect=lambda command,cancel=None:
        json.dumps(dict(CAPABILITY,canonical_prefix_policy=FMA_POLICY)) if command=='precision' else original(command,cancel))
    return result


class FusedPhaseBackendTests(unittest.TestCase):
    def test_constructor_selects_fixed_profile_and_trusted_path(self):
        def base(instance,*args,**kwargs):instance.sim=SimpleNamespace()
        with patch('tools.phase_chat.RTLChatEngine.__init__',autospec=True,side_effect=base) as init:
            result=FusedPhasePoolChatEngine()
        self.assertEqual(init.call_args.kwargs['binary'],FMA_PHASE_BINARY)
        self.assertEqual(init.call_args.kwargs['precision_profile'],'fp32_b1_fp32_fma_prefill')
        self.assertTrue(init.call_args.kwargs['canonical_prefix_reuse'])
        self.assertEqual(result.sim.phase_pool_selector,'PREFILL_12')

    def test_reuse_restart_and_wrong_policy_fail_closed(self):
        model=fused();self.addCleanup(model.close)
        for _ in range(2):result=run(model,ids=[1,2,3]*6,max_new=2)
        self.assertEqual(result['metrics']['reused_prefix_tokens'],16)
        self.assertTrue(model.status()['phase_verified'])
        self.assertEqual(model.status()['canonical_prefix_policy'],FMA_POLICY)
        model.sim.close()
        self.assertEqual(run(model,ids=[1,2,3]*6,max_new=2)['metrics']['reused_prefix_tokens'],0)
        self.assertTrue(model.status()['precision_verified']);self.assertTrue(model.phase_verified)
        original=model.sim.command
        model.sim.command=Mock(side_effect=lambda command,cancel=None:
            json.dumps(CAPABILITY) if command=='precision' else original(command,cancel))
        with self.assertRaises(RuntimeError):run(model,ids=[1,2,3]*6,max_new=2)
        self.assertIsNone(model.sim.process);self.assertEqual(model.canonical_prefix.end,0)
        self.assertFalse(model.phase_verified)

    def test_catalog_keeps_existing_choices_and_rejects_worker_override(self):
        ref,generic,phase,new=engine(cycle),engine(cycle),phase_engine(),fused()
        for item in (ref,generic,phase,new):
            item.prepare=lambda payload:ChatRequest('s',[1,2,3]*4,2,payload.get('max_draft',0))
        router=BackendRouter(ref,generic,phase,fused_phase=new);self.addCleanup(router.close)
        self.assertEqual(router.active,'rtl')
        entries={item['id']:item for item in router.catalog()}
        self.assertTrue(all(entries[x]['available'] for x in ('rtl','rtl-fp32','rtl-fp32-phase','rtl-fp32-fused-phase')))
        self.assertEqual(entries['rtl-fp32-fused-phase']['canonical_prefix_policy'],FMA_POLICY)
        self.assertEqual(entries['rtl-fp32-fused-phase']['max_draft'],3)
        for workers in (0,1,2,4,12,True):
            with self.assertRaises(ValueError):router.prepare(dict(backend='rtl-fp32-fused-phase',simulation_workers=workers))
        req=router.prepare(dict(backend='rtl-fp32-fused-phase',simulation_workers=8,
            binary='/tmp/untrusted',precision_profile='fp64',env={'QWEN_EXPERIMENTAL_PHASE_POOLS':'OFF'}))
        list(router.stream(req,threading.Event()))
        self.assertEqual(router.status()['precision_profile'],'fp32_b1_fp32_fma_prefill')
        self.assertTrue(router.status()['phase_verified'])

if __name__=='__main__':unittest.main()
