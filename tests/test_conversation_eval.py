"""Grader regression tests; no model execution or neural framework."""
import unittest
from tools.conversation_eval import CASES, REPORTED_RECOVERY, cache_warmup_messages, evaluate_response, suite_fingerprint


class QualityGraderTests(unittest.TestCase):
    def test_declared_targets_pass(self):
        valid = ['7319', 'Lima', '"Welcome aboard, Maya!"', '2, 5, 9',
                 '{"color":"teal","name":"Mira"}', '20', 'Cedar', 'unknown', '12']
        self.assertEqual(len(valid), len(CASES))
        for case, text in zip(CASES, valid):
            with self.subTest(case=case.name):
                self.assertTrue(evaluate_response(case, text)['passed'])

    def test_repetition_and_instruction_failures_are_not_hidden(self):
        for case in CASES:
            for text in ['', 'Hello! Could you clarify what you mean?', 'I am Qwen3-0.6B.']:
                with self.subTest(case=case.name, text=text):
                    self.assertFalse(evaluate_response(case, text)['passed'])
        self.assertFalse(evaluate_response(CASES[0], 'The code is 7319.')['passed'])
        self.assertFalse(evaluate_response(CASES[1], 'Oslo') ['passed'])
        self.assertFalse(evaluate_response(CASES[3], '2, 9, 5')['passed'])
        self.assertFalse(evaluate_response(CASES[4], '```json\n{"name":"Mira","color":"teal"}\n```')['passed'])
        self.assertFalse(evaluate_response(CASES[4], '{"name":"Mira","color":"teal","extra":1}')['passed'])
        self.assertFalse(evaluate_response(CASES[7], '42')['passed'])

    def test_long_history_cache_warmup_preserves_fixture(self):
        case = REPORTED_RECOVERY
        before = tuple(dict(message) for message in case.messages)
        prefix = cache_warmup_messages(case)
        self.assertEqual(len(prefix), len(case.messages)-2)
        self.assertEqual(prefix[-1]['role'], 'user')
        self.assertEqual(prefix[-1]['content'], "i'm confused")
        self.assertEqual(prefix+case.messages[-2:], case.messages)
        self.assertEqual(case.messages, before)
        for i, message in enumerate(prefix):
            self.assertEqual(message['role'], 'user' if i % 2 == 0 else 'assistant')
        with self.assertRaises(ValueError):
            cache_warmup_messages(CASES[4])

    def test_suite_is_stable_and_identifiable(self):
        self.assertEqual(len({case.name for case in CASES}), len(CASES))
        self.assertEqual(suite_fingerprint(), 'e83a1561ae3c1245ad1a2be15fc6521471ce63c2037da887af7ce0fec3136810')
        self.assertNotEqual(suite_fingerprint(CASES[:-1]), suite_fingerprint())
        self.assertEqual(len(suite_fingerprint()), 64)


if __name__ == '__main__':
    unittest.main()
