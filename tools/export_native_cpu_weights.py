#!/usr/bin/env python3
"""Offline BF16 checkpoint -> immutable QWCPU01 FP16/F32 pack; no Torch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

try:
    from .experiment import locate_model
except ImportError:
    from experiment import locate_model

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def shapes():
    result = {'model.embed_tokens.weight': (151936, 1024), 'model.norm.weight': (1024,)}
    layer = {'input_layernorm': (1024,), 'post_attention_layernorm': (1024,),
             'self_attn.q_norm': (128,), 'self_attn.k_norm': (128,),
             'self_attn.q_proj': (2048, 1024), 'self_attn.k_proj': (1024, 1024),
             'self_attn.v_proj': (1024, 1024), 'self_attn.o_proj': (1024, 2048),
             'mlp.gate_proj': (3072, 1024), 'mlp.up_proj': (3072, 1024),
             'mlp.down_proj': (1024, 3072)}
    for index in range(28):
        result.update({f'model.layers.{index}.{name}.weight': shape for name, shape in layer.items()})
    return result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def export(checkpoint, output):
    source = checkpoint / 'model.safetensors'
    targets = [output, Path(str(output) + '.index'), Path(str(output) + '.manifest.json')]
    output.parent.mkdir(parents=True, exist_ok=True)
    if any(path.exists() or path.is_symlink() for path in targets):
        raise FileExistsError('Weight pack outputs are immutable; choose a fresh path')
    before = sha(source)
    size = source.stat().st_size
    with source.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Truncated safetensors header')
        header_bytes = struct.unpack('<Q', prefix)[0]
        if not 2 <= header_bytes <= min(64 * 1024 * 1024, size - 8):
            raise ValueError('Invalid safetensors header length')
        header = json.loads(stream.read(header_bytes), object_pairs_hook=unique_object)
    start = 8 + header_bytes
    expected = shapes()
    expected['lm_head.weight'] = expected['model.embed_tokens.weight']
    actual = {k: v for k, v in header.items() if k != '__metadata__'}
    if set(actual) != set(expected) or len(actual) != 311:
        raise ValueError('Checkpoint must contain exactly the expected 311 tensors')
    intervals = []
    for name, shape in expected.items():
        spec = actual[name]
        offsets = spec.get('data_offsets')
        if spec.get('dtype') != 'BF16' or spec.get('shape') != list(shape):
            raise ValueError(f'Unexpected tensor dtype/shape: {name}')
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
            raise ValueError(f'Invalid offsets: {name}')
        begin, end = offsets
        if begin < 0 or end - begin != int(np.prod(shape)) * 2 or start + end > size:
            raise ValueError(f'Tensor bounds/length mismatch: {name}')
        intervals.append((begin, end))
    intervals.sort()
    if intervals[0][0] != 0 or any(a[1] != b[0] for a, b in zip(intervals, intervals[1:])) or start + intervals[-1][1] != size:
        raise ValueError('Checkpoint tensor regions overlap, contain holes, or leave trailing bytes')
    tied = actual['lm_head.weight']['data_offsets']
    embed = actual['model.embed_tokens.weight']['data_offsets']
    with source.open('rb') as a, source.open('rb') as b:
        a.seek(start + tied[0]); b.seek(start + embed[0])
        left = tied[1] - tied[0]
        while left:
            count = min(left, 1024 * 1024)
            if a.read(count) != b.read(count):
                raise ValueError('lm_head.weight is not bit-identical to the embedding')
            left -= count
    del expected['lm_head.weight']
    published = []
    with tempfile.TemporaryDirectory(prefix='.qwen-pack-', dir=output.parent) as temporary:
        temp = Path(temporary)
        binary, index, manifest = (temp / name for name in ('weights.bin', 'weights.bin.index', 'weights.bin.manifest.json'))
        records = []
        with source.open('rb') as source_stream, binary.open('wb') as packed:
            packed.write(b'QWCPU01' + bytes(57))
            for name, shape in expected.items():
                begin, end = actual[name]['data_offsets']
                offset = (packed.tell() + 63) // 64 * 64
                packed.write(bytes(offset - packed.tell()))
                source_stream.seek(start + begin)
                dtype = 'F16' if len(shape) == 2 else 'F32'
                source_hash, converted_hash = hashlib.sha256(), hashlib.sha256()
                remaining = end - begin
                while remaining:
                    raw = source_stream.read(min(remaining, 1024 * 1024))
                    if not raw or len(raw) % 2:
                        raise ValueError('Truncated tensor during export')
                    source_hash.update(raw)
                    values = (np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
                    if not np.isfinite(values).all():
                        raise ValueError(f'Nonfinite checkpoint tensor: {name}')
                    with np.errstate(over='ignore'):
                        converted = values.astype('<f2' if dtype == 'F16' else '<f4')
                    if not np.isfinite(converted).all():
                        raise ValueError(f'FP16 overflow in tensor: {name}')
                    payload = converted.tobytes()
                    converted_hash.update(payload); packed.write(payload)
                    remaining -= len(raw)
                length = packed.tell() - offset
                rows, columns = (shape if len(shape) == 2 else (shape[0], 1))
                if length != rows * columns * (2 if dtype == 'F16' else 4):
                    raise ValueError('Internal converted tensor length mismatch')
                records.append(dict(name=name, dtype=dtype, rows=rows, columns=columns,
                                    offset_bytes=offset, length_bytes=length,
                                    checkpoint_sha256=source_hash.hexdigest(), sha256=converted_hash.hexdigest()))
            packed.flush(); os.fsync(packed.fileno())
        index.write_text('QWCPU01 310\n' + ''.join(f"{r['name']} {r['dtype']} {r['rows']} {r['columns']} {r['offset_bytes']} {r['length_bytes']}\n" for r in records))
        if sha(source) != before:
            raise RuntimeError('Checkpoint changed while exporting')
        manifest.write_text(json.dumps(dict(format='QWCPU01', completed=True, gpu=False, pytorch=False,
            checkpoint=dict(path=str(source.resolve()), sha256=before),
            exporter_sha256=sha(Path(__file__)), numpy_version=np.__version__,
            binary=dict(path=output.name, bytes=binary.stat().st_size, sha256=sha(binary)),
            index=dict(path=targets[1].name, sha256=sha(index)), tensor_count=310,
            tied_lm_head_verified=True, rounding='BF16 widened to FP32; matrices rounded to FP16 round-to-nearest-even; 1D norms preserved FP32; little endian',
            tensors=records), indent=2) + '\n')
        try:
            # Hard links atomically create each output without replacing an existing file.
            # Manifest is the commit marker: consumers require all three files.
            for staged, target in zip((binary, index, manifest), targets):
                with staged.open('rb') as stream:
                    os.fsync(stream.fileno())
                staged.chmod(0o444)
                os.link(staged, target)
                published.append(target)
            fd = os.open(output.parent, os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
        except BaseException:
            for path in reversed(published):
                path.unlink()
            raise
    return targets[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/native-cpu/weights.bin')
    args = parser.parse_args()
    print(export(locate_model(args.checkpoint), args.output.resolve()))


if __name__ == '__main__':
    main()
