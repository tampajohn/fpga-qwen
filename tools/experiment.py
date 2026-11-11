#!/usr/bin/env python3
"""Prepare real checkpoint slices and report RTL measurements. Standard library only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build'
PROJECTIONS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def locate_model(value):
    if value:
        p = Path(value).expanduser().resolve()
    else:
        hub = Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots'
        choices = sorted(hub.glob('*/model.safetensors'))
        if not choices:
            raise ValueError('No cached Qwen3-0.6B; pass --model /path/to/checkpoint-directory')
        p = choices[-1].parent
    if not (p / 'model.safetensors').is_file():
        raise ValueError('Expected an unsharded model.safetensors and config.json')
    return p


def read_header(path):
    with path.open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        if size > 64 * 1024 * 1024:
            raise ValueError('Unreasonable safetensors header length')
        raw = f.read(size)
    return json.loads(raw), size + 8, hashlib.sha256(raw).hexdigest()


def identical_tensors(path, start, a, b):
    if a['shape'] != b['shape'] or a['dtype'] != b['dtype']:
        return False
    length = a['data_offsets'][1] - a['data_offsets'][0]
    with path.open('rb') as fa, path.open('rb') as fb:
        fa.seek(start + a['data_offsets'][0])
        fb.seek(start + b['data_offsets'][0])
        for offset in range(0, length, 1024 * 1024):
            n = min(1024 * 1024, length - offset)
            if fa.read(n) != fb.read(n):
                return False
    return True


def prepare(model):
    model = locate_model(model)
    config = json.loads((model / 'config.json').read_text())
    if config['model_type'] != 'qwen3' or config['hidden_size'] != 1024:
        raise ValueError('This experiment targets Qwen3-0.6B')
    checkpoint = model / 'model.safetensors'
    header, start, digest = read_header(checkpoint)
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    stored = sum(math.prod(v['shape']) for v in tensors.values())
    tied_duplicate = False
    if config.get('tie_word_embeddings') and 'lm_head.weight' in tensors:
        tied_duplicate = identical_tensors(checkpoint, start, tensors['lm_head.weight'],
                                          tensors['model.embed_tokens.weight'])
        if not tied_duplicate:
            raise ValueError('Config claims tied embeddings but checkpoint tensors differ')
    unique = stored - (math.prod(tensors['lm_head.weight']['shape']) if tied_duplicate else 0)
    dest = BUILD / 'fixtures'
    dest.mkdir(parents=True, exist_ok=True)
    entries = []
    with checkpoint.open('rb') as f:
        for short in PROJECTIONS:
            family = 'mlp' if short in ['gate_proj', 'up_proj', 'down_proj'] else 'self_attn'
            name = f'model.layers.0.{family}.{short}.weight'
            spec = tensors[name]
            if spec['dtype'] != 'BF16':
                raise ValueError(f'{name}: expected BF16 checkpoint tensor')
            rows, k = 8, spec['shape'][1]
            if k > 3072 or k % 16:
                raise ValueError(f'{name}: unsupported width {k}')
            f.seek(start + spec['data_offsets'][0])
            raw = f.read(rows * k * 2)
            if len(raw) != rows * k * 2:
                raise ValueError(f'{name}: truncated checkpoint')
            values = [struct.unpack('<f', struct.pack('<I', n << 16))[0]
                      for (n,) in struct.iter_unpack('<H', raw)]
            quantized, scales = [], []
            error, energy = 0.0, 0.0
            for r in range(rows):
                row = values[r*k:(r+1)*k]
                if not all(math.isfinite(x) for x in row):
                    raise ValueError('Nonfinite checkpoint weights')
                scale = max(abs(x) for x in row) / 7 or 1.0
                q = [max(-7, min(7, round(x / scale))) for x in row]
                scales.append(scale)
                quantized.extend(q)
                error += sum((x - n*scale)**2 for x, n in zip(row, q))
                energy += sum(x*x for x in row)
            (dest / f'{short}.bin').write_bytes(
                struct.pack('<III', 0x34574d51, rows, k) +
                bytes(n & 255 for n in quantized))
            entries.append(dict(name=name, shape=spec['shape'], tested_rows=rows, k=k,
                                scales=scales, weight_relative_rmse=math.sqrt(error/energy),
                                source_slice_sha256=hashlib.sha256(raw).hexdigest()))
    kv_per_token = (2 * config['num_hidden_layers'] * config['num_key_value_heads'] *
                    config['head_dim'] * 2)
    manifest = dict(model='Qwen/Qwen3-0.6B', source=str(model), config=config,
                    header_sha256=digest, stored_parameters=stored, unique_parameters=unique,
                    verified_duplicate_lm_head=tied_duplicate,
                    ideal_all_int4_weight_bytes=(unique+1)//2,
                    fp16_kv_bytes_per_token=kv_per_token,
                    fp16_kv_at_4096=kv_per_token*4096,
                    quantization='experimental symmetric per-row int4; not GPTQ/AWQ',
                    activation_source='synthetic signed int8, deterministic seeds 17 and 93',
                    fixtures=entries)
    (BUILD / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Prepared {len(entries)} real projection slices; {unique:,} unique parameters; '
          f'FP16 KV at 4096 tokens: {kv_per_token*4096 / 2**20:.0f} MiB')


def report():
    m = json.loads((BUILD / 'manifest.json').read_text())
    with (BUILD / 'results.csv').open() as f:
        results = list(csv.DictReader(f))
    if len(results) != 128:
        raise ValueError(f'Expected 128 scenarios, found {len(results)}')
    groups = {}
    for row in results:
        key = (row['matrix'], row['seed'], row['gap_cycles'])
        groups.setdefault(key, {})[int(row['token_width'])] = row
    for group in groups.values():
        baseline = group[1]
        for width, row in group.items():
            if int(row['weight_bytes']) * width != int(baseline['weight_bytes']):
                raise AssertionError('Weight reuse count mismatch')
            if row['outputs'] != baseline['outputs']:
                raise AssertionError('Unequal comparison workloads')
    lines = ['Qwen3-0.6B RTL experiment — measured simulated cycles, NOT FPGA tokens/sec',
             f"Unique parameters: {m['unique_parameters']:,}",
             f"Ideal all-int4 weights: {m['ideal_all_int4_weight_bytes']/1e6:.2f} MB (no scales)",
             f"FP16 KV at 4096 tokens: {m['fp16_kv_at_4096']/2**20:.0f} MiB",
             '128 scenarios; real layer-0 slices plus signed-edge fixture; 2 activation seeds.',
             'Each comparison computes the same eight activation vectors.',
             '', 'Aggregate real projection slices, seed 17:']
    for gap in [0, 3]:
        base = None
        for width in [1, 2, 4, 8]:
            subset = [r for r in results if r['matrix'] != 'signed_edges' and
                      int(r['seed']) == 17 and int(r['gap_cycles']) == gap and
                      int(r['token_width']) == width]
            cycles = sum(int(r['cycles']) for r in subset)
            weight_bytes = sum(int(r['weight_bytes']) for r in subset)
            if base is None:
                base = cycles
            lines.append(f'  gap={gap} width={width}: {cycles:>7,} cycles, '
                         f'{weight_bytes:>7,} weight bytes, {base/cycles:.2f}x block throughput')
    lines += ['', 'Interpretation: parallel token lanes trade arithmetic resources for weight reuse.',
              'This is verification arithmetic throughput assuming all positions are evaluated.',
              'No measured acceptance rate, attention, KV controller, nonlinearities, tokenizer,',
              'or end-to-end model generation is implemented yet. No board clock is claimed.',
              'Activation preload and fixed output stalls are included in the cycle totals.',
              'Memory model: 64-bit beats, 12-cycle startup per row, configured inter-beat gap.',
              'Not a DDR/HBM timing model. Eight-token hardware exists in every comparison.']
    text = '\n'.join(lines) + '\n'
    (BUILD / 'summary.txt').write_text(text)
    print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'report'])
    parser.add_argument('--model', help='local Qwen3-0.6B checkpoint directory')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.model)
    else:
        report()
