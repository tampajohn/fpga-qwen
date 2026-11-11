"""Keep the matched timing workload causal across prefill partitions."""
import unittest
import json
from unittest.mock import Mock, patch
from types import SimpleNamespace
from tools.benchmark_matched import SVEngine
from tools.rtl_model import RTLModel


class MatchedPrefillTests(unittest.TestCase):
    def test_start_keeps_adapter_lane_state_in_sync_with_runtime(self):
        model = RTLModel.__new__(RTLModel)
        model.lanes = 1
        model.process = SimpleNamespace(poll=lambda: None)
        model.command = Mock(return_value='configured')
        model.start = Mock()
        engine = SVEngine.__new__(SVEngine)
        engine.model = model
        engine.lanes = 8
        engine.precision_profile = 'fp64'
        engine.start()
        self.assertEqual(model.lanes, 8)
        model.command.assert_called_once_with('lanes 8', None)

    def test_hybrid_precision_requires_runtime_capabilities(self):
        expected = dict(precision_profile='fp32_b1_fp64_prefill', scalar_projection='fp32',
                        prefill_projection='fp64', attention_kv='fp64',
                        max_verification_block=1, prefix_reuse_supported=False)
        for capabilities in (expected, dict(expected, scalar_projection='fp64'),
                             dict(expected, prefix_reuse_supported=0), {}):
            with self.subTest(capabilities=capabilities):
                model = Mock(prefill_size=8)
                model.command.side_effect = [json.dumps(capabilities), 'configured']
                with patch('tools.benchmark_matched.RTLModel', return_value=model):
                    engine = SVEngine(None, 8, None, lanes=8, block_size=1,
                                      prefill_size=8, precision_profile='fp32_b1_fp64_prefill')
                if capabilities == expected and type(capabilities['prefix_reuse_supported']) is bool:
                    engine.start()
                    self.assertEqual(engine.metadata['verified_runtime_capabilities'], expected)
                    self.assertIn('FP32', engine.metadata['arithmetic'])
                else:
                    with self.assertRaises(RuntimeError): engine.start()
                model.command.assert_any_call('precision')

    def test_hybrid_rejects_incompatible_routing_before_model_load(self):
        with patch('tools.benchmark_matched.RTLModel') as model:
            with self.assertRaises(ValueError):
                SVEngine(None, 8, None, lanes=8, block_size=4, prefill_size=8,
                         precision_profile='fp32_b1_fp64_prefill')
            model.assert_not_called()

    def test_prompt_tail_and_pending_decode_token(self):
        for size in (1, 4, 8):
            for length in (1, 2, 4, 5, 6, 8, 9, 17):
                with self.subTest(block_size=size, prompt_length=length):
                    calls = []
                    class Transport:
                        block_size = min(size, 4)
                        prefill_size = size
                        def reset(self): calls.append(('reset',))
                        def block(self, tokens, logits):
                            assert 1 <= len(tokens) <= 4 and logits is False
                            calls.append(('prefill', list(tokens)))
                        def prefill(self, tokens):
                            assert 1 <= len(tokens) <= 8
                            calls.append(('prefill', list(tokens)))
                        def token(self, token, logits=True):
                            calls.append(('decode' if logits else 'prefill', [token]))
                            return token+1, 1, 0.0
                    engine = SVEngine.__new__(SVEngine)
                    engine.model = Transport()
                    engine.tokenizer = SimpleNamespace(decode=lambda *a, **kw: 'text')
                    prompt = list(range(10, 10+length))
                    result = engine.generate(prompt, 3, True)
                    self.assertEqual(calls[0], ('reset',))
                    consumed = [t for kind, ids in calls[1:] if kind == 'prefill' for t in ids]
                    self.assertEqual(consumed, prompt[:-1])
                    decode = [ids[0] for kind, ids in calls[1:] if kind == 'decode']
                    self.assertEqual(decode, [prompt[-1], prompt[-1]+1, prompt[-1]+2])
                    self.assertEqual(result['output_token_count'], 3)


if __name__ == '__main__': unittest.main()
