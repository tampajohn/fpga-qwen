"""Exact projection gates for the opt-in B1 two-row tile; no timing claim.

Reuse sim/fp32_projection_probe.cpp unchanged. Both B1 binaries must have
KERNEL_ROWS=151936 (vocabulary capacity); B4 binaries must use capacity3072.
Candidate B1 uses FP32_ROW_PAIRS; candidate B4 uses the same edited SV source
without that flag. Supply frozen legacy/candidate binaries and their manifests.
No builds, full-model inference, host neural implementation, or scoring here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from experiment import read_header
from fp32_rounding_oracle import projection_fixture
from rtl_model import prepare_layout


MATRICES = [('q', 4, 2048, 1024), ('gate', 11, 3072, 1024),
            ('down', 13, 1024, 3072), ('vocabulary', 2, 151936, 1024)]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def command_text(value):
    if isinstance(value, list):
        return ' '.join(command_text(x) for x in value)
    return str(value)


def inspect_artifact(path, batch, kind):
    path = path.resolve()
    manifest_path = path.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    digest = sha(path)
    if manifest['binary_sha256'] != digest:
        raise RuntimeError(f'Manifest/executable mismatch: {path}')
    command = command_text(manifest['command'])
    tokens = command.split()
    for define in (f'+define+KERNEL_BATCH={batch}', f'+define+KERNEL_ROWS={151936 if batch == 1 else 3072}'):
        if define not in tokens:
            raise RuntimeError(f'Missing required compiled probe dimension: {define}')
    if '-fno-fast-math' not in command or '-ffp-contract=off' not in command:
        raise RuntimeError('Separate-round strict FP32 archive flags required')
    if '+define+FP32_EXPLICIT_FMA' in command:
        raise RuntimeError('This gate requires separate multiply/add, not explicit FMA')
    paired = '+define+FP32_ROW_PAIRS' in command
    if paired != (kind == 'candidate' and batch == 1):
        raise RuntimeError(f'Wrong row-pair flag for {kind} B{batch}')
    return dict(path=str(path), batch=batch, kind=kind, sha256=digest,
                manifest_sha256=sha(manifest_path), manifest=manifest)


def verify_matrix_layout(model, layout):
    offsets = [int(line) for line in layout.read_text().splitlines()]
    header, start, _ = read_header(model / 'model.safetensors')
    names = {'q': 'model.layers.0.self_attn.q_proj.weight',
             'gate': 'model.layers.0.mlp.gate_proj.weight',
             'down': 'model.layers.0.mlp.down_proj.weight',
             'vocabulary': 'lm_head.weight' if 'lm_head.weight' in header else 'model.embed_tokens.weight'}
    for name, slot, rows, width in MATRICES:
        tensor = header[names[name]]
        if (tensor['dtype'] != 'BF16' or tensor['shape'] != [rows, width]
                or offsets[slot] != (start + tensor['data_offsets'][0]) // 2):
            raise RuntimeError(f'Wrong full-matrix weight mapping: {name}')
    return offsets


def cases(batch):
    # rows1 exercises scalar-only fallback; rows2 is the minimum tile; rows3/7
    # exercise odd endings. Workers8 with tiny rows includes empty partitions.
    for rows, width in ((1, 8), (2, 16), (3, 16), (7, 24)):
        for workers in (1, 3, 8):
            yield f'tiny-{rows}-{width}', 4, rows, width, workers, True
    for name, slot, rows, width in MATRICES:
        if batch == 4 and name == 'vocabulary':
            continue
        for workers in (1, 8, 17):
            yield name, slot, rows, width, workers, False
        # Real row stride, with an odd final matrix row and uneven boundaries.
        for workers in (3, 8, 17):
            yield name + '-odd-tail', slot, rows - 1, width, workers, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for batch in (1, 4):
        for kind in ('legacy', 'candidate'):
            parser.add_argument(f'--{kind}-b{batch}', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    artifacts = {(batch, kind): inspect_artifact(getattr(args, f'{kind}_b{batch}'), batch, kind)
                 for batch in (1, 4) for kind in ('legacy', 'candidate')}
    source_key = 'rtl/fp32_projection_kernel.sv'
    candidate_sources = [artifacts[batch, 'candidate']['manifest']['source_hashes'][source_key] for batch in (1, 4)]
    compilers = [artifacts[batch, 'candidate']['manifest'].get('compiler_binary_sha256') for batch in (1, 4)]
    if not compilers[0] or compilers[0] != compilers[1]:
        raise RuntimeError('B1 and unchanged B4 candidates must use the same compiler binary')
    if candidate_sources[0] != candidate_sources[1]:
        raise RuntimeError('B1 and unchanged B4 candidates must use the same edited SV source')
    model, layout = prepare_layout(None)
    offsets = verify_matrix_layout(model, layout)
    report = dict(completed=False, timing_claim=False, gpu=False, pytorch=False,
                  artifacts=list(artifacts.values()), checks=[], invalid_inputs=[],
                  checkpoint_sha256=sha(model / 'model.safetensors'),
                  source_hashes={p.name: sha(p) for p in (Path(__file__),
                      Path(__file__).with_name('fp32_rounding_oracle.py'))},
                  scope='B1 full Q/gate/down/vocabulary and odd tails; B4 unchanged full Q/gate/down. Tiny bounded exact-rational oracle only; full matrices use frozen separate-round legacy outputs. Synthetic deterministic activations from the existing probe. No broad model-quality or performance conclusion.',
                  partition_policy='Probe partitions [rows*id/workers, rows*(id+1)/workers); tiny rows include empty workers; 3/17 workers and odd matrices expose odd starts/ends.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gold = {}
    try:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for batch in (1, 4):
                for name, slot, rows, width, workers, tiny in cases(batch):
                    outputs = {}
                    partitions = [[rows * i // workers, rows * (i + 1) // workers] for i in range(workers)]
                    for kind in ('legacy', 'candidate'):
                        path = artifacts[batch, kind]['path']
                        dump = directory / f'{kind}.bin'
                        subprocess.run([path, str(model / 'model.safetensors'), str(layout),
                                        str(slot), str(rows), str(width), str(workers), '1', str(dump)],
                                       check=True, capture_output=True, text=True)
                        if dump.stat().st_size != rows * batch * 8:
                            raise AssertionError('Wrong full output byte count')
                        array = np.fromfile(dump, dtype=np.float64)
                        if not np.isfinite(array).all():
                            raise AssertionError('Nonfinite projection output')
                        outputs[kind] = array.tobytes()
                    if outputs['legacy'] != outputs['candidate']:
                        raise AssertionError(f'Legacy bit mismatch B{batch} {name} workers{workers}')
                    key = (batch, slot, rows, width)
                    if key in gold and outputs['candidate'] != gold[key]:
                        raise AssertionError('Worker partitioning changed projection bits')
                    gold[key] = outputs['candidate']
                    if tiny:
                        oracle = np.asarray(projection_fixture(model / 'model.safetensors', offsets[slot],
                                                               rows, width, batch), dtype=np.float64).tobytes()
                        if outputs['candidate'] != oracle:
                            raise AssertionError('Independent exact-rational rounding mismatch')
                    record = dict(batch=batch, name=name, slot=slot, rows=rows, width=width,
                                  workers=workers, partitions=partitions, legacy_bit_exact=True,
                                  partition_invariant=True, rational_oracle_exact=True if tiny else None,
                                  output_sha256=hashlib.sha256(outputs['candidate']).hexdigest())
                    report['checks'].append(record)
                    args.output.write_text(json.dumps(report, indent=2) + '\n')
                    print(json.dumps({k: record[k] for k in ('batch', 'name', 'workers', 'legacy_bit_exact')}), flush=True)
                maximum = 151936 if batch == 1 else 3072
                for rows, width, workers in ((0, 8, 1), (maximum + 1, 8, 1), (1, 7, 1),
                                             (1, 9, 1), (1, 3080, 1), (1, 8, 0), (1, 8, 33)):
                    result = subprocess.run([artifacts[batch, 'candidate']['path'],
                        str(model / 'model.safetensors'), str(layout), '4', str(rows), str(width),
                        str(workers), '1', str(directory / 'invalid.bin')], capture_output=True, text=True)
                    if result.returncode != 1 or 'Invalid probe shape' not in result.stderr:
                        raise AssertionError('Invalid shape was not rejected by unchanged transport')
                    report['invalid_inputs'].append(dict(batch=batch, rows=rows, width=width, workers=workers, rejected=True))
        for item in report['artifacts']:
            if sha(item['path']) != item['sha256'] or sha(Path(item['path']).parent / 'manifest.json') != item['manifest_sha256']:
                raise RuntimeError('Frozen executable/manifest changed during checks')
        report['completed'] = True
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
