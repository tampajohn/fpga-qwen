#!/usr/bin/env python3
"""Guarded tiny-trajectory worker/idle/pack checks, never a performance benchmark."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import select
import struct
import subprocess
import time

from benchmark_guard import BenchmarkGate

ROOT = Path(__file__).resolve().parents[1]
VOCABULARY = 151936


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def guard(directory):
    status = BenchmarkGate(directory).status()
    metadata = json.loads((directory / 'run.json').read_text())
    if not status['running'] or metadata.get('pid') != os.getppid():
        raise RuntimeError('Run directly under benchmark_guard.py')
    return {'directory': str(directory), 'parent_pid': metadata['pid']}


def environment():
    env = dict(os.environ)
    removed = []
    for key in list(env):
        if (key.startswith(('DYLD_', 'GGML_KLEIDIAI_', 'LLAMA_ARG_'))
                or key in ('GGML_BACKEND_PATH', 'GGML_TOTAL_THREADS', 'VECLIB_MAXIMUM_THREADS')):
            removed.append(key)
            del env[key]
    # Hold vendor BLAS settings constant while testing custom pool extremes.
    env['VECLIB_MAXIMUM_THREADS'] = '1'
    return env, sorted(removed)


class Engine:
    def __init__(self, binary, weights, workers, directory, env):
        self.process = None
        self.log = (directory / 'stderr.log').open('xb')
        self.buffer = b''
        try:
            self.process = subprocess.Popen(
                [str(binary), '--weights', str(weights), '--threads', str(workers),
                 '--prefill-threads', str(workers), '--context', '8'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                bufsize=0, env=env)
            self.ready = self.read()
            expected = dict(ready=True, protocol=1, gpu=False, pytorch=False,
                            weight_precision='f16', activation_precision='f32',
                            kv_precision='f16', decode_threads=workers,
                            prefill_threads=workers, context=8, position=0)
            if any(self.ready.get(k) != v for k, v in expected.items()):
                raise RuntimeError('Unexpected native handshake')
        except BaseException:
            self.close()
            raise

    def read(self):
        deadline = time.monotonic() + 180
        while b'\n' not in self.buffer:
            if time.monotonic() >= deadline:
                raise TimeoutError('Native robustness response timed out')
            if select.select([self.process.stdout], [], [], .1)[0]:
                block = os.read(self.process.stdout.fileno(), 65536)
                if not block:
                    raise RuntimeError('Native stdout closed')
                self.buffer += block
                if len(self.buffer) > 1024 * 1024:
                    raise RuntimeError('Oversized native response')
        line, self.buffer = self.buffer.split(b'\n', 1)
        result = json.loads(line)
        if not isinstance(result, dict) or 'error' in result:
            raise RuntimeError(f'Native error: {result}')
        return result

    def command(self, text):
        if '\n' in text or '\r' in text:
            raise ValueError('Command must occupy one line')
        payload = memoryview((text + '\n').encode())
        while payload:
            count = self.process.stdin.write(payload)
            if not count:
                raise RuntimeError('Native stdin closed')
            payload = payload[count:]
        return self.read()

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            for pipe in (self.process.stdin, self.process.stdout):
                if pipe:
                    pipe.close()
        self.log.close()


def malformed_checks(binary, weights, directory, env):
    original = Path(str(weights) + '.index').read_text().splitlines()
    fixtures = []
    fixtures.append(('bad_header', original, 'Invalid weight pack header', True))
    changed = original.copy(); changed[0] = 'QWCPU01 309'
    fixtures.append(('bad_count', changed, 'Invalid weight index header/count', False))
    changed = original.copy(); fields = changed[1].split(); fields[2] = '1'; changed[1] = ' '.join(fields)
    fixtures.append(('wrong_shape', changed, 'Wrong tensor shape', False))
    changed = original.copy(); fields = changed[1].split(); fields[4] = str((weights.stat().st_size + 63) // 64 * 64); changed[1] = ' '.join(fields)
    fixtures.append(('out_of_bounds', changed, 'Invalid tensor range', False))
    changed = original.copy(); fields = changed[1].split(); fields[4] = '65'; changed[1] = ' '.join(fields)
    fixtures.append(('unaligned', changed, 'Invalid tensor range', False))
    changed = original.copy(); changed[2] = changed[1]
    fixtures.append(('duplicate', changed, 'Unexpected or duplicate tensor', False))
    # Alias two FP32 norm regions, so finite-value validation cannot mask overlap.
    changed = original.copy()
    source = next(line.split() for line in original[1:] if line.split()[0] == 'model.layers.0.input_layernorm.weight')
    target = next(i for i, line in enumerate(original) if line.split()[0] == 'model.norm.weight')
    fields = changed[target].split(); fields[4] = source[4]; changed[target] = ' '.join(fields)
    fixtures.append(('overlap', changed, 'Overlapping tensors', False))
    results = []
    for name, lines, expected_error, bad_header in fixtures:
        fixture_dir = directory / name; fixture_dir.mkdir()
        fixture = fixture_dir / 'weights.bin'
        if bad_header:
            # Rejection occurs at the header before index or tensor data access.
            fixture.write_bytes(b'BADCPU1' + bytes(57))
        else:
            # Only index bytes change: immutable canonical binary is read via a
            # symlink. No 1.2-GB copy/clone or writable canonical handle needed.
            fixture.symlink_to(weights)
        index = Path(str(fixture) + '.index')
        index.write_text('\n'.join(lines) + '\n')
        command = [str(binary), '--weights', str(fixture), '--threads', '1',
                   '--prefill-threads', '1', '--context', '8']
        result = subprocess.run(command, input='quit\n', text=True,
                                capture_output=True, timeout=180, env=env)
        (fixture_dir / 'stdout.log').write_text(result.stdout)
        (fixture_dir / 'stderr.log').write_text(result.stderr)
        if result.returncode == 0 or expected_error not in result.stderr or result.stdout.strip():
            raise RuntimeError(f'{name}: malformed fixture was not rejected as expected')
        results.append(dict(case=name, returncode=result.returncode,
                            expected_error=expected_error, index_sha256=sha(index),
                            binary_kind='64-byte invalid-header fixture' if bad_header else 'read-only canonical symlink'))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT / 'build/benchmark-status')
    args = parser.parse_args()
    guarded = guard(args.guard_state_dir.resolve())
    binary, weights, directory = args.binary.resolve(), args.weights.resolve(), args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, passed=False, diagnostic_only=True,
                  performance_claim_eligible=False, gpu=False, pytorch=False,
                  guard=guarded, runs=[], checks={},
                  scope='Tiny four-token trajectory; exact worker-count/idle/reset equivalence and malformed-pack rejection. Not a full-context or throughput gate.')
    def save():
        temporary = directory / 'report.pending.json'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(directory / 'report.json')
    try:
        save()
        paths = [binary, weights, Path(str(weights) + '.index'), Path(__file__).resolve()]
        manifest = Path(str(weights) + '.manifest.json')
        if manifest.is_file():
            paths.append(manifest)
        report['inputs'] = {str(path): sha(path) for path in paths}
        env, removed = environment()
        report['environment'] = dict(removed_names=removed, VECLIB_MAXIMUM_THREADS='1')
        reference = None
        for workers in (1, 64):
            run_dir = directory / f'workers-{workers}'; run_dir.mkdir()
            engine = Engine(binary, weights, workers, run_dir, env)
            run = dict(workers=workers, ready=engine.ready, rounds=[])
            report['runs'].append(run)
            try:
                for repetition, idle in enumerate((0.0, .002, .05)):
                    reset = engine.command('reset')
                    if reset.get('ok') is not True or reset.get('position') != 0:
                        raise RuntimeError('Reset failed')
                    if engine.command('mode greedy').get('ok') is not True:
                        raise RuntimeError('Mode failed')
                    records, payloads = [], []
                    for step, command in enumerate(('prefill 9707 11', 'step 1917', 'step 0')):
                        time.sleep(idle)
                        response = engine.command(command)
                        if response.get('position') != step + 2 or type(response.get('token')) is not int or not 0 <= response['token'] < VOCABULARY:
                            raise RuntimeError('Bad token/position')
                        path = run_dir / f'round-{repetition}-step-{step}.f32'
                        dump = engine.command('dump ' + str(path))
                        if dump.get('ok') is not True or dump.get('position') != step + 2 or dump.get('count') != VOCABULARY:
                            raise RuntimeError('Bad dump metadata')
                        payload = path.read_bytes()
                        if len(payload) != VOCABULARY * 4 or not all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', payload)):
                            raise RuntimeError('Bad/nonfinite logits')
                        payloads.append(payload)
                        records.append(dict(position=step + 2, token=response['token'], sha256=sha(path)))
                    evidence = ([r['token'] for r in records], payloads)
                    if reference is None:
                        reference = evidence
                    elif evidence != reference:
                        raise RuntimeError('Worker/idle/reset changed exact full logits or greedy tokens')
                    run['rounds'].append(dict(idle_seconds=idle, records=records))
                    save()
                time.sleep(.05)
                if engine.command('quit').get('ok') is not True:
                    raise RuntimeError('Quit not acknowledged')
                if engine.process.wait(timeout=10) != 0:
                    raise RuntimeError('Clean worker shutdown failed')
                run['clean_shutdown'] = True
            finally:
                engine.close()
        report['checks']['worker_idle_reset_exact'] = True
        report['checks']['malformed_packs'] = malformed_checks(binary, weights, directory, env)
        if any(sha(Path(path)) != digest for path, digest in report['inputs'].items()):
            raise RuntimeError('Immutable input changed')
        report.update(completed=True, passed=True)
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        save()
    print(json.dumps({'passed': report['passed'], 'directory': str(directory)}))


if __name__ == '__main__':
    main()
