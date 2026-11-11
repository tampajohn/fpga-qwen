"""Keep the matched timing workload causal across prefill partitions."""
import unittest
from types import SimpleNamespace
from tools.benchmark_matched import SVEngine


class MatchedPrefillTests(unittest.TestCase):
    def test_prompt_tail_and_pending_decode_token(self):
        for size in (1, 4):
            for length in (1, 2, 4, 5, 6, 8, 9):
                with self.subTest(block_size=size, prompt_length=length):
                    calls = []
                    class Transport:
                        block_size = size
                        def reset(self): calls.append(('reset',))
                        def block(self, tokens, logits):
                            assert 1 <= len(tokens) <= 4 and logits is False
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
