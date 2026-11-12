"""Host-only tests for lifecycle selector scoping and restart validation."""
import argparse
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.check_fma_canonical_real import QUERY_VALUE_ENV, query_value_scope, query_value_selector
from tools.rtl_model import RTLModel


class LifecycleSelectorTests(unittest.TestCase):
    def fake(self):
        model = object.__new__(RTLModel)
        model.process = None
        model.command = lambda command: json.dumps(dict(
            selector=os.environ[QUERY_VALUE_ENV], min_context=32,
            context_metric='paired_query_common_kv_tokens', requires_second_query=True,
            paired_head_pairs=0, fallback_head_pairs=0))
        return model

    def test_start_restart_and_restore(self):
        model = self.fake()
        observations = []
        def fake_start(instance):
            instance.process = SimpleNamespace(pid=100 if instance.process is None else instance.process.pid + 1)
        with patch.dict(os.environ, {QUERY_VALUE_ENV: 'ON'}), patch.object(RTLModel, 'start', fake_start):
            with query_value_scope('MIN_CONTEXT:32', observations):
                model.start()
                model.start()
                self.assertEqual(os.environ[QUERY_VALUE_ENV], 'MIN_CONTEXT:32')
            self.assertEqual(os.environ[QUERY_VALUE_ENV], 'ON')
            self.assertIs(RTLModel.start, fake_start)
        self.assertEqual([row['restarted'] for row in observations], [False, True])
        self.assertTrue(all(row['validated'] for row in observations))

    def test_bad_policy_fails_and_restores_unset_environment(self):
        model = self.fake()
        observations = []
        def fake_start(instance):
            instance.process = SimpleNamespace(pid=100)
        with patch.dict(os.environ, {}, clear=True), patch.object(RTLModel, 'start', fake_start):
            with self.assertRaises(RuntimeError):
                with query_value_scope('ON', observations):
                    model.start()  # Claimed threshold32 conflicts with ON's threshold0.
            self.assertNotIn(QUERY_VALUE_ENV, os.environ)
            self.assertIs(RTLModel.start, fake_start)
        self.assertFalse(observations[0]['validated'])

    def test_unselected_mode_leaves_start_and_environment_unchanged(self):
        observations = []
        original = RTLModel.start
        with patch.dict(os.environ, {QUERY_VALUE_ENV: 'ON'}):
            with query_value_scope(None, observations):
                self.assertIs(RTLModel.start, original)
                self.assertEqual(os.environ[QUERY_VALUE_ENV], 'ON')
        self.assertEqual(observations, [])

    def test_selector_validation(self):
        for value in ('OFF', 'ON', 'MIN_CONTEXT:1', 'MIN_CONTEXT:4096'):
            self.assertEqual(query_value_selector(value), value)
        for value in ('', 'on', 'MIN_CONTEXT:0', 'MIN_CONTEXT:4097', 'MIN_CONTEXT:+32', 'MIN_CONTEXT:１２'):
            with self.assertRaises(argparse.ArgumentTypeError):
                query_value_selector(value)


if __name__ == '__main__':
    unittest.main()
