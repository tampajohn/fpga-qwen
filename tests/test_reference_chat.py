import http.client
import json
import threading
import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Historical PyTorch tests require --extra reference")

from tools.chat_engine import ChatEngine, common_prefix
from tools.chat_server import make_server


class ChatEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ChatEngine(threads=4)

    def setUp(self):
        self.engine.clear_cache()

    def request(self, messages=None, **kwargs):
        payload = dict(session='test', messages=messages or [
            {'role': 'user', 'content': 'Answer with the single word blue.'}],
            max_new_tokens=16, max_draft=7)
        payload.update(kwargs)
        return self.engine.prepare(payload)

    def baseline(self, request):
        with torch.inference_mode():
            result = self.engine.model.generate(
                input_ids=torch.tensor([request.ids], device='cpu'),
                attention_mask=torch.ones((1, len(request.ids)), dtype=torch.long, device='cpu'),
                do_sample=False, max_new_tokens=request.max_new,
                temperature=1.0, top_p=1.0, top_k=50,
                pad_token_id=self.engine.tokenizer.eos_token_id)
        return self.engine.tokenizer.decode(result[0, len(request.ids):], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)

    def test_cpu_greedy_equivalence_and_second_turn_prefix_reuse(self):
        first = self.request()
        events = list(self.engine.stream(first))
        self.assertEqual(events[-1]['type'], 'done')
        self.assertEqual(events[-1]['text'], self.baseline(first))
        self.assertEqual(events[-1]['metrics']['reused_prefix_tokens'], 0)
        second = self.request(messages=[
            {'role': 'user', 'content': 'Answer with the single word blue.'},
            {'role': 'assistant', 'content': events[-1]['text']},
            {'role': 'user', 'content': 'What word did you just say?'}])
        expected_reuse = common_prefix(self.engine.cache_ids, second.ids[:-1])
        events = list(self.engine.stream(second))
        self.assertEqual(events[-1]['text'], self.baseline(second))
        self.assertGreater(expected_reuse, 0)
        self.assertEqual(events[-1]['metrics']['reused_prefix_tokens'], expected_reuse)
        self.assertEqual(self.engine.cache.get_seq_length(), len(self.engine.cache_ids))
        self.assertEqual(self.engine.status()['device'], 'cpu')
        self.assertFalse(self.engine.status()['gpu_inference'])
        self.assertTrue(all(p.device.type == 'cpu' for p in self.engine.model.parameters()))

    def test_suffix_off_and_session_isolation(self):
        request = self.request(max_draft=0)
        events = list(self.engine.stream(request))
        final = events[-1]
        self.assertEqual(final['text'], self.baseline(request))
        self.assertEqual(final['metrics']['proposed_tokens'], 0)
        self.assertEqual(final['metrics']['rounds'], final['metrics']['output_tokens'])
        new_session = self.request(session='different-browser')
        events = list(self.engine.stream(new_session))
        self.assertEqual(events[-1]['metrics']['reused_prefix_tokens'], 0)

    def test_stop_during_decode_and_retry(self):
        cancel = threading.Event()
        request = self.request(messages=[{'role': 'user', 'content':
            'Repeat the phrase red blue green exactly 100 times, with no explanation.'}], max_new_tokens=128)
        events = []
        for event in self.engine.stream(request, cancel):
            events.append(event)
            if event['type'] == 'update':
                cancel.set()
        self.assertEqual(events[-1]['finish_reason'], 'stopped')
        self.assertGreater(events[-1]['metrics']['output_tokens'], 0)
        self.assertLess(events[-1]['metrics']['output_tokens'], 128)
        retry = self.request()
        final = list(self.engine.stream(retry))[-1]
        self.assertEqual(final['text'], self.baseline(retry))

    def test_cancel_before_prefill(self):
        cancel = threading.Event()
        cancel.set()
        final = list(self.engine.stream(self.request(), cancel))[-1]
        self.assertEqual(final['finish_reason'], 'stopped')
        self.assertEqual(final['text'], '')
        self.assertEqual(final['metrics']['output_tokens'], 0)

    def test_interrupted_chunked_prefill_can_resume(self):
        request = self.request(messages=[{'role': 'user', 'content':
            ('The sky is blue. ' * 45) + 'What color is the sky? Answer with one word.'}], max_draft=0)
        self.assertGreater(len(request.ids), 128)
        cancel = threading.Event()
        events = []
        for event in self.engine.stream(request, cancel):
            events.append(event)
            if event['type'] == 'prefill':
                cancel.set()
        self.assertEqual(events[-1]['finish_reason'], 'stopped')
        self.assertEqual(events[-1]['metrics']['output_tokens'], 0)
        self.assertEqual(len(self.engine.cache_ids), 128)
        final = list(self.engine.stream(request))[-1]
        self.assertEqual(final['metrics']['reused_prefix_tokens'], 128)
        self.assertEqual(final['text'], self.baseline(request))

    def test_request_validation(self):
        for kwargs in [{'max_new_tokens': 513}, {'max_new_tokens': True},
                       {'max_draft': 8}, {'session': ''},
                       {'messages': [{'role': 'assistant', 'content': 'bad'}]},
                       {'messages': [{'role': 'user', 'content': ' '}]},
                       {'messages': [{'role': 'user', 'content': 'x ' * 5500}]}]:
            with self.subTest(kwargs=list(kwargs)):
                with self.assertRaises(ValueError):
                    self.request(**kwargs)


