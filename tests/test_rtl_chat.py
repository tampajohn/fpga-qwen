"""Full simulator chat protocol, cancellation and framework isolation checks."""
import sys
import threading
import time
import unittest
from tools.chat_backends import BackendRouter
from tools.rtl_chat import RTLChatEngine
from tools.chat_protocol import ChatRequest


class RTLChatTests(unittest.TestCase):
    def setUp(self):
        self.engine = RTLChatEngine()
        self.router = BackendRouter(self.engine)

    def tearDown(self):
        self.router.close()

    def test_no_pytorch_and_validation(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('transformers', sys.modules)
        self.assertFalse(self.engine.status()['pytorch_loaded'])
        payload = dict(session='test', messages=[dict(role='user', content='Hello')], max_draft=0)
        backend, request = self.router.prepare(payload)
        self.assertEqual(backend, 'rtl')
        self.assertEqual(request.decoding, 'chat')
        self.assertEqual(request.ids[-4:], [151667, 271, 151668, 271])
        for fields in [dict(decoding='invalid'), dict(seed=-1), dict(seed=True), dict(simulation_workers=True), dict(simulation_workers=3), dict(simulation_workers=-1), dict(backend='reference'), dict(max_draft=1), dict(max_new_tokens=999),
                       dict(messages=[dict(role='assistant', content='Hello')]),
                       dict(messages=[dict(role='user', content='x '*5500)])]:
            with self.assertRaises(ValueError):
                self.router.prepare(dict(payload, **fields))

    def test_prefix_reuse_and_reset(self):
        cancel = threading.Event()
        # Short raw requests isolate KV behavior without repeated chat-template prefill.
        first = list(self.engine.stream(ChatRequest('one', [9707, 11], 1, 0), cancel))[-1]
        self.assertEqual(first['type'], 'done')
        second = list(self.engine.stream(ChatRequest('one', [9707, 11, 1917], 1, 0, workers=2), cancel))[-1]
        self.assertEqual(second['metrics']['simulation_workers'], 2)
        self.assertEqual(second['metrics']['reused_prefix_tokens'], 2)
        self.engine.close()
        fresh = list(self.engine.stream(ChatRequest('new', [9707, 11, 1917], 1, 0), cancel))[-1]
        self.assertEqual(second['text'], fresh['text'])
        self.assertEqual(fresh['metrics']['reused_prefix_tokens'], 0)

    def test_cancel_interrupts_token_and_can_resume(self):
        self.engine.sim.start()
        cancel = threading.Event()
        timer = threading.Timer(.1, cancel.set)
        timer.start()
        started = time.perf_counter()
        result = list(self.engine.stream(ChatRequest('one', [9707, 11], 10, 0), cancel))[-1]
        timer.join()
        self.assertEqual(result['finish_reason'], 'stopped')
        self.assertLess(time.perf_counter()-started, 2)
        self.assertIsNone(self.engine.sim.process)
        self.assertEqual(self.engine.cache_ids, [])
        result = list(self.engine.stream(ChatRequest('one', [9707], 1, 0), threading.Event()))[-1]
        self.assertEqual(result['type'], 'done')
        self.assertEqual(result['metrics']['output_tokens'], 1)


if __name__ == '__main__':
    unittest.main()
