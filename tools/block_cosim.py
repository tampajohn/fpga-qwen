#!/usr/bin/env python3
"""One complete Qwen3 layer: RTL linear ops, CPU nonlinear ops and KV payload."""
import argparse
import json
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

from experiment import locate_model, ROOT
from kv import RTLCursor


def quantize_rows(values, bound):
    """Independent scale per row/token; round-to-nearest-even, signed symmetric."""
    maxima = np.max(np.abs(values), axis=-1, keepdims=True)
    scales = np.where(maxima == 0, np.float32(1), maxima / np.float32(bound))
    return np.clip(np.rint(values / scales), -bound, bound).astype(np.int8), scales


class LinearEngine:
    def __init__(self, weights, mode, directory):
        self.mode = mode
        self.directory = directory
        self.records = []
        self.weights = {name: quantize_rows(w, 7) for name, w in weights.items() if w.ndim == 2}
        self.fp_weights = weights
        for name, (q, _) in self.weights.items():
            rows, k = q.shape
            (directory / f'{name}.bin').write_bytes(struct.pack('<III', 0x34574d51, rows, k) + q.tobytes())

    def __call__(self, name, x):
        if self.mode == 'fp32':
            return x @ self.fp_weights[name].T
        q, w_scale = self.weights[name]
        a, a_scale = quantize_rows(x, 127)
        # Independent integer oracle for every output, using NumPy, not RTL code.
        expected = a.astype(np.int32) @ q.astype(np.int32).T
        if self.mode == 'rtl':
            with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
                tmp = Path(tmp)
                (tmp / 'x.bin').write_bytes(struct.pack('<III', 0x38584d51, *a.shape) + a.tobytes())
                start = time.monotonic()
                result = subprocess.run([str(ROOT / 'build/linear/Vq4_matvec'),
                                         str(self.directory / f'{name}.bin'),
                                         str(tmp / 'x.bin'), str(tmp / 'y.bin')],
                                        check=True, capture_output=True, text=True, timeout=180)
                record = json.loads(result.stdout)
                record.update(projection=name, tokens=len(x), simulated_wall_seconds=time.monotonic()-start)
                self.records.append(record)
                raw = (tmp / 'y.bin').read_bytes()
                if struct.unpack('<III', raw[:12]) != (0x32594d51, len(x), len(q)):
                    raise AssertionError('RTL output shape mismatch')
                actual = np.frombuffer(raw[12:], dtype='<i4').reshape(len(x), len(q))
                np.testing.assert_array_equal(actual, expected, err_msg=f'{name}: RTL integer mismatch')
        else:
            actual = expected
        return actual.astype(np.float32) * a_scale * w_scale.T


class KVPayload:
    """CPU payload array with RTL or software ownership/visibility metadata."""
    def __init__(self, config, dtype=np.float16, cursor=None):
        shape = (4096, config['num_key_value_heads'], config['head_dim'])
        self.k = np.full(shape, np.nan, dtype=dtype)
        self.v = np.full(shape, np.nan, dtype=dtype)
        self.cursor = cursor
        self.committed = 0
        self.pending = 0

    def append(self, k, v):
        count = len(k)
        if self.pending or not 1 <= count <= 8 or self.committed+count > len(self.k):
            raise ValueError('Invalid KV reservation')
        start = self.committed
        if self.cursor:
            self.cursor.command(0, count)
        self.pending = count
        self.k[start:start+count] = k
        self.v[start:start+count] = v
        if self.cursor:
            state = self.cursor.command(1, count)
            if state.visible_end != start+count:
                raise AssertionError('RTL payload visibility mismatch')
        return (self.k[:start+count].astype(np.float32),
                self.v[:start+count].astype(np.float32))

    def commit(self, accepted):
        if not self.pending or not 0 <= accepted <= self.pending:
            raise ValueError('Invalid commit')
        if self.cursor:
            self.cursor.command(2, accepted)
        self.committed += accepted
        self.pending = 0
        if self.cursor and self.cursor.state.committed != self.committed:
            raise AssertionError('RTL commit mismatch')

    def abort(self):
        if not self.pending:
            raise ValueError('No pending transaction')
        if self.cursor:
            self.cursor.command(3)
        self.pending = 0

    def truncate(self, length):
        if self.pending or not 0 <= length <= self.committed:
            raise ValueError('Invalid prefix truncation')
        if self.cursor:
            self.cursor.command(4, length)
        self.committed = length


class Block:
    def __init__(self, config, weights, engine):
        self.config, self.w, self.linear = config, weights, engine

    def norm(self, x, name):
        return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) +
                           np.float32(self.config['rms_norm_eps'])) * self.w[name]

    def rope(self, x, positions):
        dim = x.shape[-1]
        inv = np.float32(1) / (np.float32(self.config['rope_theta']) **
                               (np.arange(0, dim, 2, dtype=np.float32)/dim))
        angle = positions.astype(np.float32)[:, None] * inv[None, :]
        angle = np.concatenate([angle, angle], axis=-1)[:, None, :]
        rotated = np.concatenate([-x[..., dim//2:], x[..., :dim//2]], axis=-1)
        return x*np.cos(angle) + rotated*np.sin(angle)

    def __call__(self, x, cache):
        c = self.config
        n, start = len(x), cache.committed
        h = self.norm(x, 'input_layernorm.weight')
        d, heads, kvheads = c['head_dim'], c['num_attention_heads'], c['num_key_value_heads']
        q = self.linear('self_attn.q_proj.weight', h).reshape(n, heads, d)
        k = self.linear('self_attn.k_proj.weight', h).reshape(n, kvheads, d)
        v = self.linear('self_attn.v_proj.weight', h).reshape(n, kvheads, d)
        positions = np.arange(start, start+n)
        q = self.rope(self.norm(q, 'self_attn.q_norm.weight'), positions)
        k = self.rope(self.norm(k, 'self_attn.k_norm.weight'), positions)
        keys, values = cache.append(k, v)
        # Each pair of query heads shares one KV head. Explicit causal masking
        # is essential when the whole speculative block is in the cache.
        keys = np.repeat(keys, heads//kvheads, axis=1)
        values = np.repeat(values, heads//kvheads, axis=1)
        scores = np.einsum('thd,shd->hts', q, keys) * np.float32(d**-0.5)
        causal = np.arange(len(keys))[None, :] <= positions[:, None]
        scores = np.where(causal[None, :, :], scores, -np.inf)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        attended = np.einsum('hts,shd->thd', probs, values).reshape(n, heads*d)
        h = x + self.linear('self_attn.o_proj.weight', attended)
        normalized = self.norm(h, 'post_attention_layernorm.weight')
        gate = self.linear('mlp.gate_proj.weight', normalized)
        up = self.linear('mlp.up_proj.weight', normalized)
        # Stable sigmoid; tanh identity avoids exp overflow.
        activated = gate * (np.float32(0.5) * (np.tanh(gate*np.float32(0.5)) + np.float32(1)))
        output = h + self.linear('mlp.down_proj.weight', activated * up)
        if not np.all(np.isfinite(output)):
            raise AssertionError('Nonfinite block output; check KV visibility and arithmetic')
        return output


class TorchQuantLinear(torch.nn.Module):
    """Separate PyTorch W4A8 arithmetic used inside the stock HF decoder layer."""
    def __init__(self, weight):
        super().__init__()
        scale = weight.abs().amax(dim=-1, keepdim=True) / 7
        self.register_buffer('scale', torch.where(scale == 0, torch.ones_like(scale), scale))
        self.register_buffer('q', torch.clamp(torch.round(weight / self.scale), -7, 7).to(torch.int32))

    def forward(self, x):
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        scale = x.abs().amax(dim=-1, keepdim=True) / 127
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int32)
        y = (q @ self.q.T).float() * scale * self.scale.T
        return y.reshape(*shape, -1)


def hf_oracle(config, weights, x, quantized=False):
    config = Qwen3Config(**config)
    config._attn_implementation = 'eager'
    layer = Qwen3DecoderLayer(config, layer_idx=0).float().eval()
    layer.load_state_dict({name: torch.from_numpy(w.copy()) for name, w in weights.items()}, strict=True)
    if quantized:
        for name, w in weights.items():
            if w.ndim != 2:
                continue
            parent_name, attr, _ = name.rsplit('.', 2)
            setattr(layer.get_submodule(parent_name), attr, TorchQuantLinear(torch.from_numpy(w.copy())))
    x = torch.from_numpy(x.copy())[None]
    positions = torch.arange(x.shape[1])[None]
    mask = torch.full((1, 1, x.shape[1], x.shape[1]), float('-inf')).triu(1)
    rotary = Qwen3RotaryEmbedding(config)
    with torch.inference_mode():
        y = layer(x, attention_mask=mask, position_ids=positions,
                  position_embeddings=rotary(x, positions))
    return y[0].numpy()


def errors(a, b):
    return dict(max_abs=float(np.max(np.abs(a-b))),
                relative_rmse=float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='def add(a, b):\n    return a + b\n\n')
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = locate_model(args.model)
    config = json.loads((model / 'config.json').read_text())
    if config['hidden_size'] != 1024 or config['model_type'] != 'qwen3' or config.get('rope_scaling'):
        raise ValueError('Unsupported model configuration')
    dest = ROOT / 'build/block'
    dest.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))
    ids = tokenizer.encode(args.prompt).ids[:8]
    if len(ids) < 8:
        raise ValueError('Use a prompt with at least eight tokens')
    correction = tokenizer.encode(' subtraction').ids[0]
    with safe_open(str(model / 'model.safetensors'), framework='pt', device='cpu') as f:
        prefix = 'model.layers.0.'
        weights = {name[len(prefix):]: f.get_tensor(name).float().numpy()
                   for name in f.keys() if name.startswith(prefix)}
        embedded = f.get_tensor('model.embed_tokens.weight')[ids + [correction]].float().numpy()
    report = dict(model='Qwen/Qwen3-0.6B', source=str(model), layer=0, token_ids=ids,
                  correction_token=correction, scope='one full layer; all 7 linear projections in RTL; '
                  'FP32 nonlinearities and FP16 KV payload on CPU; RTL KV metadata',
                  memory_model='ideal 64-bit weight beat/cycle, no DDR/HBM timing', results={})
    print('Checking independent Hugging Face decoder-layer oracle...', flush=True)
    fp = LinearEngine(weights, 'fp32', dest)
    fp_result = Block(config, weights, fp)(embedded[:8], KVPayload(config, dtype=np.float32))
    oracle = hf_oracle(config, weights, embedded[:8])
    np.testing.assert_allclose(fp_result, oracle, atol=3e-5, rtol=3e-5)
    report['results']['fp32_vs_hf'] = errors(fp_result, oracle)

    software = LinearEngine(weights, 'software', dest)
    sw_block = Block(config, weights, software)
    quant_result = sw_block(embedded[:8], KVPayload(config, dtype=np.float32))
    quant_oracle = hf_oracle(config, weights, embedded[:8], quantized=True)
    np.testing.assert_allclose(quant_result, quant_oracle, atol=3e-4, rtol=3e-4)
    report['results']['w4a8_vs_hf_w4a8'] = errors(quant_result, quant_oracle)
    report['results']['w4a8_vs_fp32_quality_proxy_only'] = errors(quant_result, oracle)

    rtl = LinearEngine(weights, 'rtl', dest)
    rtl_block = Block(config, weights, rtl)
    with RTLCursor() as cursor:
        cache = KVPayload(config, cursor=cursor)
        print('Running complete layer: 4-token prefix through RTL...', flush=True)
        prefix_out = rtl_block(embedded[:4], cache)
        cache.commit(4)
        print('Verifying 4 candidate positions through RTL...', flush=True)
        candidate_out = rtl_block(embedded[4:8], cache)
        fresh = sw_block(embedded[:8], KVPayload(config))
        np.testing.assert_allclose(np.concatenate([prefix_out, candidate_out]), fresh, atol=3e-4, rtol=3e-4)
        report['results']['rtl_prefix_plus_candidates_vs_fresh'] = errors(
            np.concatenate([prefix_out, candidate_out]), fresh)
        # Deliberately accept two candidates; this is a controlled rollback test,
        # not model-measured acceptance. Rejected payload remains in RAM but is
        # invisible. Poison it to prove attention does not read stale tail slots.
        cache.commit(2)
        cache.k[6:] = np.nan
        cache.v[6:] = np.nan
        print('Rejecting tail, inserting correction, checking against fresh prefix...', flush=True)
        corrected = rtl_block(embedded[8:9], cache)
        cache.commit(1)
        corrected_input = np.concatenate([embedded[:6], embedded[8:9]])
        corrected_ref = sw_block(corrected_input, KVPayload(config))[-1:]
        np.testing.assert_allclose(corrected, corrected_ref, atol=3e-4, rtol=3e-4)
        report['results']['correction_after_rollback'] = errors(corrected, corrected_ref)

        # Exact prefix reuse, not stale suffix-state reuse.
        cache.truncate(4)
        print('Reusing prefix after truncation; checking batched vs serial execution...', flush=True)
        # Keep expensive RTL passes to four; test the serial path independently
        # in software with the same quantization and FP16 cache storage.
        replay = rtl_block(embedded[4:8], cache)
        cache.commit(4)
        np.testing.assert_array_equal(replay, candidate_out)
        serial_cache = KVPayload(config)
        serial = []
        for row in embedded[:8]:
            serial.append(sw_block(row[None], serial_cache))
            serial_cache.commit(1)
        serial = np.concatenate(serial)
        np.testing.assert_allclose(serial, fresh, atol=3e-4, rtol=3e-4)
        report['results']['batched_vs_serial_w4a8'] = errors(fresh, serial)
        report['results']['final_committed_tokens'] = cursor.state.committed

    report['rtl_linear_calls'] = rtl.records
    report['rtl_integer_outputs_checked'] = sum(r['output_bytes']//4 for r in rtl.records)
    report['rtl_linear_cycles_only'] = sum(r['cycles'] for r in rtl.records)
    report['rtl_weight_bytes'] = sum(r['weight_bytes'] for r in rtl.records)
    report['limitations'] = ['No full 28-layer generation or LM head yet',
                             'No measured suffix acceptance; rejection is scripted',
                             'No FPGA synthesis, place/route, or measured clock',
                             'Nonlinearities and attention remain CPU operations',
                             'KV payload remains CPU RAM; RTL only controls visibility',
                             'Per-row W4A8 quantization is experimental, not accuracy-calibrated',
                             'Simulated cycles exclude all CPU operations and file transfers']
    (dest / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['rtl_linear_calls', 'source']}, indent=2))
    print('PASS: full layer co-simulation, every RTL integer result, causal cache reuse and rollback.', flush=True)


if __name__ == '__main__':
    main()
