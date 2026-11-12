"""Independent NumPy oracle for every logit of the full SV model; no PyTorch."""
import os
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.experiment import read_header
from tools.rtl_model import RTLModel


class NumpyQwen:
    def __init__(self, model, weight_precision='bf16', kv_precision='f32'):
        # Defaults preserve the existing BF16-checkpoint / FP32-KV oracle.
        # f32 weights are the same exact BF16 values widened from this checkpoint.
        if weight_precision not in ('bf16', 'f16', 'f32') or kv_precision not in ('f16', 'f32'):
            raise ValueError('Unsupported oracle weight/KV precision')
        self.weight_precision, self.kv_precision = weight_precision, kv_precision
        header, start, _ = read_header(model / 'model.safetensors')
        self.weights = {}
        with (model / 'model.safetensors').open('rb') as f:
            for name, spec in header.items():
                if name == '__metadata__' or name == 'lm_head.weight':
                    continue
                f.seek(start + spec['data_offsets'][0])
                data = np.frombuffer(f.read(spec['data_offsets'][1]-spec['data_offsets'][0]), dtype='<u2')
                value = (data.astype(np.uint32)<<16).view(np.float32).reshape(spec['shape'])
                # Native MOSTLY_F16 GGUF retains one-dimensional norm weights
                # as FP32. Round only matrix tensors, then widen for FP32 math.
                if weight_precision == 'f16' and len(spec['shape']) == 2:
                    value = value.astype(np.float16).astype(np.float32)
                self.weights[name] = value
        self.cache = []

    def reset(self):
        self.cache = []

    def token(self, token, logits=True):
        w = self.weights
        position = len(self.cache)
        payload = []
        def norm(x, weight):
            return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + np.float32(1e-6)) * weight
        angle = position / (1000000.0 ** (np.arange(64, dtype=np.float64)*2/128))
        cos, sin = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
        def rope(x):
            left, right = np.split(x, 2, axis=-1)
            return np.concatenate((left*cos-right*sin, right*cos+left*sin), axis=-1)
        x = w['model.embed_tokens.weight'][token].copy()
        for layer in range(28):
            def weight(suffix):
                return w[f'model.layers.{layer}.{suffix}.weight']
            h = norm(x, weight('input_layernorm'))
            q = rope(norm((weight('self_attn.q_proj') @ h).reshape(16, 128), weight('self_attn.q_norm')))
            k = rope(norm((weight('self_attn.k_proj') @ h).reshape(8, 128), weight('self_attn.k_norm')))
            v = (weight('self_attn.v_proj') @ h).reshape(8, 128)
            if self.kv_precision == 'f16':
                # Current-position values must be rounded before attention too,
                # not merely when they become previous-token cache entries.
                k = k.astype(np.float16).astype(np.float32)
                v = v.astype(np.float16).astype(np.float32)
            payload.append((k, v))
            keys = np.stack([entry[layer][0] for entry in self.cache] + [k])
            vals = np.stack([entry[layer][1] for entry in self.cache] + [v])
            keys, vals = np.repeat(keys, 2, axis=1), np.repeat(vals, 2, axis=1)
            scores = np.einsum('hd,thd->ht', q, keys) / np.float32(np.sqrt(128))
            probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            attended = np.einsum('ht,thd->hd', probs, vals).reshape(-1)
            x += weight('self_attn.o_proj') @ attended
            h = norm(x, weight('post_attention_layernorm'))
            gate, up = weight('mlp.gate_proj') @ h, weight('mlp.up_proj') @ h
            activated = gate * (.5 + .5*np.tanh(gate*.5)) * up
            x += weight('mlp.down_proj') @ activated
        self.cache.append(payload)
        if logits:
            return w['model.embed_tokens.weight'] @ norm(x, w['model.norm.weight'])
        return None


class FullModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = RTLModel()
        cls.oracle = NumpyQwen(cls.sim.model)
        cls.sim.start()
        cls.errors = []

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()
        print(json.dumps({'full_vocab_logit_errors': cls.errors}), flush=True)

    def compare(self, token):
        expected = self.oracle.token(token)
        best, position, _ = self.sim.token(token)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'logits.bin'
            self.assertEqual(self.sim.command(f'dump {path}'), 'dumped')
            actual = np.fromfile(path, dtype=np.float64)
        self.assertEqual(position, len(self.oracle.cache))
        self.assertEqual(best, int(expected.argmax()))
        error = float(np.max(np.abs(actual-expected)))
        self.errors.append(error)
        np.testing.assert_allclose(actual, expected, atol=.0001, rtol=.00001)
        return actual

    def test_full_logits_cache_and_reset(self):
        self.sim.reset(); self.oracle.reset()
        initial = self.compare(9707)
        # Multi-token attention catches RoPE, GQA, cache indexing and causal errors.
        self.compare(11)
        self.compare(1917)
        self.assertEqual(self.sim.command('truncate 1'), 'truncated')
        self.oracle.cache = self.oracle.cache[:1]
        self.compare(0)
        self.sim.reset(); self.oracle.reset()
        repeated = self.compare(9707)
        np.testing.assert_array_equal(initial, repeated)


if __name__ == '__main__':
    unittest.main()
