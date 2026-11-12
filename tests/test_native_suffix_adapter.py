"""No inference: protocol transaction and token-accounting regressions."""
import sys
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from native_cpu_chat import NativeCPUChat


class FakeNative(NativeCPUChat):
    def __init__(self, verification=None, capable=True):
        super().__init__(SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids, **kw: str(ids))))
        self.verification = verification
        self.capable = capable
        self.calls = []
        self.position = 0
        self.closed = False

    def start(self, cancel):
        self.process = SimpleNamespace(poll=lambda: None)
        self.ready = dict(prefix_reuse=True, prefix_reuse_chunk=512,
                          verify_greedy_b2=self.capable, max_verified_draft=1)

    def close(self):
        self.closed = True
        self.process = self.ready = None
        self._invalidate_cache()

    def command(self, command, cancel, **kwargs):
        self.calls.append(command)
        op, *args = command.split()
        if op == 'reset':
            self._invalidate_cache()
            self.position = 0
            return dict(ok=True, position=0)
        if op == 'mode':
            return dict(ok=True)
        if op == 'prefill':
            self.position += len(args)
            return dict(token=8, position=self.position)
        if op == 'verify-greedy':
            row = dict(self.verification)
            self.position += row['committed_inputs']
            row.setdefault('position', self.position)
            row.setdefault('verified_positions', 2)
            row.setdefault('finish_reason', 'eos' if row['tokens'][-1] in (151643, 151645) else None)
            return row
        if op == 'step':
            self.position += 1
            return dict(token=int(args[0])+1, position=self.position)
        raise AssertionError(command)


def request(ids=None, max_new=3, max_draft=1, decoding='greedy'):
    return SimpleNamespace(ids=ids if ids is not None else [7, 8, 9, 7], max_new=max_new,
                           max_draft=max_draft, decoding=decoding, session='test', seed=42)


class NativeSuffixTests(unittest.TestCase):
    def run_request(self, engine, req=None):
        return list(engine.stream(req or request(), threading.Event()))[-1]

    def test_accept_consumes_only_verified_inputs(self):
        engine = FakeNative(dict(tokens=[9, 10], accepted=1, committed_inputs=2))
        done = self.run_request(engine)
        self.assertEqual(done['token_ids'], [8, 9, 10])
        self.assertEqual(engine.cache_ids, [7, 8, 9, 7, 8, 9])
        self.assertEqual(engine.canonical_reusable_tokens, 0)
        self.assertEqual(done['metrics']['rounds'], 2)
        self.assertEqual(done['metrics']['accepted_tokens'], 1)
        self.assertTrue(done['metrics']['suffix_enabled'])
        self.assertEqual(done['metrics']['verification_calls'], 1)
        self.assertEqual(done['metrics']['verified_positions'], 2)
        self.assertEqual(done['metrics']['rejected_tokens'], 0)
        self.assertEqual(engine.calls[-1], 'verify-greedy 8 9')

    def test_mismatch_continues_from_correction(self):
        engine = FakeNative(dict(tokens=[15], accepted=0, committed_inputs=1))
        done = self.run_request(engine)
        self.assertEqual(done['token_ids'], [8, 15, 16])
        self.assertEqual(engine.cache_ids[-2:], [8, 15])
        self.assertEqual(engine.calls[-1], 'step 15')

    def test_eos_first_or_bonus_is_never_consumed(self):
        for tokens, accepted, committed, expected in [([151645], 0, 1, [8]), ([9, 151645], 1, 2, [8, 9])]:
            engine = FakeNative(dict(tokens=tokens, accepted=accepted, committed_inputs=committed))
            done = self.run_request(engine)
            self.assertEqual(done['token_ids'], expected)
            self.assertEqual(done['finish_reason'], 'eos')
            self.assertNotIn(151645, engine.cache_ids)

    def test_matching_eos_draft_commits_only_pending(self):
        engine = FakeNative(dict(tokens=[151645], accepted=1, committed_inputs=1))
        done = self.run_request(engine, request(ids=[7, 8, 151645, 7]))
        self.assertEqual(done['token_ids'], [8])
        self.assertEqual(engine.cache_ids, [7, 8, 151645, 7, 8])
        self.assertEqual(engine.calls[-1], 'verify-greedy 8 151645')

    def test_old_capability_disabled_draft_and_single_budget_use_steps(self):
        for engine, req in [(FakeNative(capable=False), request()), (FakeNative(), request(max_draft=0)),
                            (FakeNative(), request(max_new=2))]:
            done = self.run_request(engine, req)
            self.assertFalse(any(call.startswith('verify-greedy') for call in engine.calls))
            self.assertEqual(done['token_ids'], [8, 9, 10][:req.max_new])

    def test_no_observed_suffix_uses_steps(self):
        engine = FakeNative()
        self.run_request(engine, request(ids=[1, 2, 3]))
        self.assertFalse(any(call.startswith('verify-greedy') for call in engine.calls))

    def test_reuse_ceiling_not_extended_by_verify(self):
        engine = FakeNative(dict(tokens=[9, 10], accepted=1, committed_inputs=2))
        self.run_request(engine, request(ids=[1]*508+[7, 8, 9, 7]))
        self.assertEqual(engine.canonical_reusable_tokens, 512)
        self.assertEqual(len(engine.cache_ids), 514)

    def test_cancel_between_two_verified_outputs_closes_and_clears(self):
        engine = FakeNative(dict(tokens=[9, 10], accepted=1, committed_inputs=2))
        cancel = threading.Event()
        events = []
        for event in engine.stream(request(), cancel):
            events.append(event)
            if event['type'] == 'update' and event['token_ids'] == [8, 9]:
                cancel.set()
        self.assertEqual(events[-1]['finish_reason'], 'stopped')
        self.assertEqual(events[-1]['token_ids'], [8, 9])
        self.assertTrue(engine.closed)
        self.assertEqual(engine.cache_ids, [])

    def test_invalid_transaction_closes(self):
        engine = FakeNative(dict(tokens=[9, 10], accepted=0, committed_inputs=2))
        with self.assertRaisesRegex(RuntimeError, 'verification response'):
            self.run_request(engine)
        self.assertTrue(engine.closed)
        self.assertEqual(engine.cache_ids, [])

    def test_sampled_optin_rejected(self):
        engine = FakeNative()
        with self.assertRaisesRegex(ValueError, 'greedy'):
            self.run_request(engine, request(decoding='chat'))
        self.assertEqual(engine.calls, [])


if __name__ == '__main__':
    unittest.main()
