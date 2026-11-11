"""Regression for the reported cross-turn repetition, using actual model inference."""
import json
import os
import threading
import unittest
from tools.rtl_chat import RTLChatEngine
from tools.experiment import ROOT


class ConversationTests(unittest.TestCase):
    def test_recover_from_repetitive_history(self):
        engine = RTLChatEngine(binary=os.environ.get('SV_TEST_BINARY'),
                               lanes=int(os.environ.get('SV_TEST_LANES', '1')))
        repeated = "Hello! I'm confused too. Could you clarify what you're asking?"
        messages = []
        for prompt, reply in [('hi', 'Hello! How can I assist you today?'),
                              ('what did you say', 'Hello! What did you say?'),
                              ('?', 'Hello! What did you say?'), ("i'm confused", repeated)]:
            messages += [dict(role='user', content=prompt), dict(role='assistant', content=reply)]
        messages.append(dict(role='user', content='what are you'))
        try:
            request = engine.prepare(dict(session='reported-repetition', messages=messages,
                                          max_new_tokens=80, seed=42))
            final = list(engine.stream(request, threading.Event()))[-1]
            self.assertEqual(final['type'], 'done')
            self.assertNotEqual(final['text'], repeated)
            self.assertTrue(any(word in final['text'].lower() for word in ['assistant', 'qwen', 'ai ']), final['text'])
            messages += [dict(role='assistant', content=final['text']),
                         dict(role='user', content='what model are you')]
            second = list(engine.stream(engine.prepare(dict(session='reported-repetition',
                messages=messages, max_new_tokens=80, seed=43)), threading.Event()))[-1]
            self.assertIn('qwen', second['text'].lower())
            self.assertGreater(second['metrics']['reused_prefix_tokens'], 0)
            report = dict(first=final['text'], second=second['text'],
                          reused_prefix_tokens=second['metrics']['reused_prefix_tokens'])
            (ROOT/'build/repetition-regression.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(report), flush=True)
        finally:
            engine.close()


if __name__ == '__main__':
    unittest.main()
