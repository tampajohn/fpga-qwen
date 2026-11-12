"""Fixed-phase host selection and capability gates without neural inference."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from tools.chat_backends import BackendRouter
from tools.chat_protocol import ChatRequest
from tools.canonical_prefix import CanonicalPrefix, POLICY
from tools.phase_chat import PhasePoolChatEngine
from tools.rtl_model import RTLModel
from tests.test_suffix_chat import engine, cycle, run

PRECISION=dict(precision_profile='fp32_b1_fp64_prefill',scalar_projection='fp32',
               prefill_projection='fp64',verification_projection='fp32',attention_kv='fp64',
               max_verification_block=4,prefix_reuse_supported=False,canonical_prefix_policy=POLICY)
PHASE=dict(selector='PREFILL_12',resident_decode_workers=8,resident_prefill_workers=12,
           prefill_workers=12,decode_workers=8,pool_constructions=2)


def phase_engine(capabilities=PHASE):
    model=engine(cycle)
    model.__class__=PhasePoolChatEngine
    model.sim.prefill_size=8
    model.precision_profile='fp32_b1_fp64_prefill'
    model.phase_verified=False
    model.reuse_prefix=False
    model.canonical_prefix=CanonicalPrefix()
    original=model.sim.command
    model.sim.command=Mock(side_effect=lambda command,cancel=None:
        json.dumps(PRECISION) if command=='precision' else json.dumps(capabilities)
        if command=='phase_pools' else original(command,cancel))
    return model


class PhaseBackendTests(unittest.TestCase):
    def test_each_request_restart_and_canonical_reuse(self):
        model=phase_engine()
        first=run(model,ids=[1,2,3]*6,max_new=2)
        second=run(model,ids=[1,2,3]*6,max_new=2)
        self.assertEqual(first['token_ids'],second['token_ids'])
        self.assertEqual(second['metrics']['reused_prefix_tokens'],16)
        self.assertEqual(second['metrics']['prefill_workers'],12)
        self.assertTrue(model.status()['phase_verified'])
        model.sim.close()  # Process loss without clearing engine metadata.
        recovered=run(model,ids=[1,2,3]*6,max_new=2)
        self.assertEqual(recovered["metrics"]["reused_prefix_tokens"],0)
        model.close()
        self.assertFalse(model.status()['phase_verified'])
        third=run(model,ids=[1,2,3]*6,max_new=2)
        self.assertEqual(third['metrics']['reused_prefix_tokens'],0)
        self.assertEqual(sum(call.args[0]=='phase_pools' for call in model.sim.command.call_args_list),4)

    def test_mismatch_fails_before_prefill_and_closes(self):
        for actual in ({},[],dict(PHASE,selector='OFF'),dict(PHASE,prefill_workers=8),
                       dict(PHASE,pool_constructions=True)):
            with self.subTest(actual=actual):
                model=phase_engine(actual)
                with self.assertRaises(RuntimeError):run(model)
                self.assertEqual(model.sim.prefills,[])
                self.assertIsNone(model.sim.process)
                self.assertFalse(model.phase_verified)
                original=model.sim.command
                model.sim.command=Mock(side_effect=lambda command,cancel=None:
                    json.dumps(PHASE) if command=='phase_pools' else original(command,cancel))
                run(model,max_new=1)
                self.assertTrue(model.phase_verified)

    def test_router_fixed_workers_and_http_fields_ignored(self):
        phase=phase_engine();generic=engine(cycle);reference=engine(cycle)
        for model in (phase,generic,reference):
            model.prepare=lambda payload:ChatRequest('s',[1,2,3]*4,2,payload.get('max_draft',0))
        router=BackendRouter(reference,generic,phase)
        self.addCleanup(router.close)
        self.assertEqual(router.active,'rtl')
        entry=next(x for x in router.catalog() if x['id']=='rtl-fp32-phase')
        self.assertEqual(entry['fixed_workers'],8)
        self.assertTrue(entry['prefix_reuse_supported'])
        for workers in (0,1,2,4,12,True):
            with self.assertRaises(ValueError):router.prepare(dict(backend='rtl-fp32-phase',simulation_workers=workers))
        prepared=router.prepare(dict(backend='rtl-fp32-phase',simulation_workers=8,
                                     binary='/tmp/untrusted',env={'QWEN_EXPERIMENTAL_PHASE_POOLS':'OFF'}))
        done=list(router.stream(prepared,threading.Event()))[-1]
        self.assertEqual(done['metrics']['decode_workers'],8)
        self.assertEqual(router.status()['resident_worker_threads'],20)
        router.prepare(dict(backend='rtl-fp32',simulation_workers=2))
        request=ChatRequest('s',[1],1,0);request.workers=2
        with self.assertRaises(ValueError):list(phase.stream(request,threading.Event()))

    def test_subprocess_selector_is_explicit_not_ambient(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'build/full').mkdir(parents=True)
            with patch('tools.rtl_model.ROOT',root),patch('tools.rtl_model.prepare_layout',return_value=(root,root/'offsets')):
                for selector in (None,'OFF','PREFILL_12'):
                    model=RTLModel(binary=root/'fixed',workers=8,lanes=8)
                    model.phase_pool_selector=selector
                    model.read=Mock(return_value='ready');model.command=Mock(return_value='configured')
                    with patch.dict(os.environ,{'QWEN_EXPERIMENTAL_PHASE_POOLS':'UNTRUSTED'}),patch('tools.rtl_model.subprocess.Popen') as popen:
                        model.start()
                        self.assertEqual(popen.call_args.kwargs['env'].get('QWEN_EXPERIMENTAL_PHASE_POOLS'),selector)
                        self.assertEqual(os.environ['QWEN_EXPERIMENTAL_PHASE_POOLS'],'UNTRUSTED')
                        model.close()

if __name__=='__main__':unittest.main()
