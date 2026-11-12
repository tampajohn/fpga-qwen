#!/usr/bin/env python3
"""Guarded macOS sample capture of the original warmed CPU engine.

Raw text/logs stay under ignored build/. Only sanitized summary.json may be
published. sample includes waiting threads and aggregates away timestamps:
its stack counts are NOT CPU-time or wall-time shares, nor phase attribution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

from benchmark_guard import BenchmarkGate
from native_cpu_chat import NativeCPUChat, EOS

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def anchor():
    before = time.perf_counter_ns()
    epoch = time.time_ns()
    after = time.perf_counter_ns()
    return dict(unix_ns=epoch, perf_before_ns=before, perf_after_ns=after)


def ids(path):
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not value or any(type(x) is not int or not 0 <= x < 151936 for x in value):
        raise ValueError('Expected nonempty JSON token-ID array')
    return value


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 3), (signal.SIGTERM, 3), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
    raise RuntimeError('Owned sampler did not exit')


def parse_sample(text):
    if 'Call graph:' not in text or 'Total number in stack' not in text:
        raise RuntimeError('sample output lacks recognized call graph')
    graph = text.split('Call graph:', 1)[1].split('Total number in stack', 1)[0]
    counts = [int(n) for n in re.findall(r'^\s+(\d+)\s+Thread_\S+.*$', graph, re.M)]
    if not counts or sum(counts) == 0:
        raise RuntimeError('sample output has no recognized thread observations')
    tops = []
    marker = 'Sort by top of stack, same collapsed (when >= 5):'
    if marker in text:
        for line in text.split(marker, 1)[1].splitlines():
            match = re.match(r'^\s*(.+?)\s+\(in [^)]+\)\s+(\d+)\s*$', line)
            if match:
                tops.append(dict(symbol=match[1], thread_stack_observations=int(match[2])))
    return dict(thread_root_count=len(counts), thread_stack_observations=sum(counts),
                per_thread_observations_min=min(counts), per_thread_observations_max=max(counts),
                top_stack_symbols=tops[:30],
                interpretation='Aggregated sampled thread stacks INCLUDING WAITING threads. Counts across threads are not elapsed samples, CPU time, CPU percentages, or wall-time fractions. Symbols do not identify vocabulary versus layer projections when kernels are shared.')


def request(client, cancel, prompt, count, expected):
    if client.command('reset', cancel).get('position') != 0:
        raise RuntimeError('Reset failed')
    if client.command('mode greedy', cancel).get('ok') is not True:
        raise RuntimeError('Mode failed')
    start_anchor = anchor()
    begin = time.perf_counter_ns()
    response = client.command('prefill ' + ' '.join(map(str, prompt)), cancel)
    arrivals, outputs = [], []
    for index in range(count):
        now = time.perf_counter_ns()
        token = response.get('token')
        if type(token) is not int or not 0 <= token < 151936 or response.get('position') != len(prompt) + index:
            raise RuntimeError('Unexpected token/position')
        if token in EOS:
            raise RuntimeError('Natural EOS reached before fixed diagnostic workload completed')
        outputs.append(token); arrivals.append(now)
        if index + 1 < count:
            response = client.command(f'step {token}', cancel)
    if expected is not None and outputs != expected:
        raise RuntimeError('Diagnostic output trajectory changed')
    return dict(output_ids=outputs, cached_tokens=0, prompt_tokens=len(prompt),
                output_tokens=count, clock_before=start_anchor, clock_after=anchor(),
                perf_windows_ns=dict(request_start=begin, first_token=arrivals[0],
                                     last_token=arrivals[-1], returned=time.perf_counter_ns()),
                token_arrivals_perf_ns=arrivals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('binary', 'weights', 'prompt-ids', 'directory'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--expected-output-ids', type=Path)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--prefill-threads', type=int, default=6)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--duration', type=int, default=20)
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT / 'build/benchmark-status')
    args = parser.parse_args()
    if not 5 <= args.duration <= 30 or not 2 <= args.tokens <= 128:
        parser.error('Use duration5..30 seconds and2..128 output tokens')
    if not 1 <= args.threads <= 64 or not 1 <= args.prefill_threads <= 64:
        parser.error('Thread counts must be1..64')
    state = args.guard_state_dir.resolve()
    if not BenchmarkGate(state).status()['running'] or json.loads((state / 'run.json').read_text()).get('pid') != os.getppid():
        raise RuntimeError('Run directly under benchmark_guard.py')
    directory = args.directory.resolve()
    if not directory.is_relative_to(ROOT / 'build'):
        raise ValueError('Raw profiler artifacts must stay inside ignored repository build/')
    directory.mkdir(parents=True, exist_ok=False)
    binary, weights, prompt_path = args.binary.resolve(), args.weights.resolve(), args.prompt_ids.resolve()
    report = dict(completed=False, diagnostic_only=True, performance_claim_eligible=False,
                  gpu=False, pytorch=False, status='preparing', requests=[],
                  sampler='macOS sample,1ms interval, all thread states',
                  phase_limit='Aggregate sample has no per-sample timestamps; request windows cannot filter its stacks into prefill/decode. No CPU-share claim.',
                  attach_limit='sample has no readiness notification here. Popen/exit windows include attach/finalization; recording start and overlap duration are not independently confirmed.')
    client = None; recorder = None; sampler_logs = []
    original_env = dict(os.environ)
    def save():
        (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    try:
        prompt = ids(prompt_path)
        if len(prompt) + args.tokens > 4096:
            raise ValueError('Diagnostic workload exceeds context capacity')
        expected = ids(args.expected_output_ids) if args.expected_output_ids else None
        if expected is not None and len(expected) != args.tokens:
            raise ValueError('Expected output count mismatch')
        paths = [binary, weights, Path(str(weights)+'.index'), Path(str(weights)+'.manifest.json'), prompt_path,
                 Path(__file__).resolve(), ROOT/'tools/native_cpu_chat.py']
        manifest = binary.parent / 'manifest.json'
        if not manifest.is_file():
            raise ValueError('Explicit immutable binary build manifest required')
        paths.append(manifest)
        build = json.loads(manifest.read_text())
        if build.get('completed') is not True or build.get('binary_sha256') != sha(binary) or build.get('gpu') is not False:
            raise ValueError('Build manifest does not match CPU binary')
        if args.expected_output_ids:
            paths.append(args.expected_output_ids.resolve())
        report['inputs'] = {str(p): sha(p) for p in paths}
        removed = []
        for key in list(os.environ):
            if key.startswith(('DYLD_', 'GGML_KLEIDIAI_', 'LLAMA_ARG_')) or key in ('GGML_BACKEND_PATH', 'GGML_TOTAL_THREADS', 'VECLIB_MAXIMUM_THREADS'):
                removed.append(key); del os.environ[key]
        report['removed_environment_names'] = sorted(removed)
        client = NativeCPUChat(None, binary, weights, args.threads, args.prefill_threads, reuse_prefix=False)
        cancel = threading.Event(); client.start(cancel)
        report['ready'] = client.ready; report['target_pid'] = client.process.pid
        report['warmups'] = []
        for _ in range(2):
            warm = request(client, cancel, prompt, args.tokens, expected)
            if expected is None:
                expected = warm['output_ids']
            report['warmups'].append(warm)
        report['trajectory_reference'] = 'explicit expected IDs' if args.expected_output_ids else 'first warmup IDs; repeatability only, not an independent quality gate'
        sample_path = directory / 'sample.raw.txt'
        command = ['/usr/bin/sample', str(client.process.pid), str(args.duration), '1', '-file', str(sample_path)]
        report['sampler_command'] = command; report['recorder_launch'] = anchor()
        sampler_logs = [(directory/'sample.stdout.log').open('xb'), (directory/'sample.stderr.log').open('xb')]
        recorder = subprocess.Popen(command, stdout=sampler_logs[0], stderr=sampler_logs[1], start_new_session=True)
        deadline = time.monotonic() + args.duration + 60
        report['status'] = 'sampling'; save()
        while recorder.poll() is None:
            if time.monotonic() > deadline:
                raise TimeoutError('Owned sample recorder exceeded recording/finalization deadline')
            report['requests'].append(request(client, cancel, prompt, args.tokens, expected))
        report['recorder_exit_code'] = recorder.wait(); report['recorder_exit'] = anchor()
        if recorder.returncode != 0 or not sample_path.is_file() or not report['requests']:
            raise RuntimeError('sample failed or produced no repeated workload; inspect local logs')
        parsed = parse_sample(sample_path.read_text(errors='replace'))
        report['sample_counts'] = parsed; report['raw_sample_sha256'] = sha(sample_path)
        if client.process.pid != report['target_pid'] or client.process.poll() is not None:
            raise RuntimeError('Profiled engine identity changed')
        if any(sha(Path(p)) != value for p, value in report['inputs'].items()):
            raise RuntimeError('Immutable profile input changed')
        report.update(completed=True, status='recorded-aggregate-thread-stacks')
        summary = dict(completed=True, diagnostic_only=True, performance_claim_eligible=False,
                       binary_sha256=report['inputs'][str(binary)], raw_sample_sha256=report['raw_sample_sha256'],
                       prompt_sha256=report['inputs'][str(prompt_path)], prompt_tokens=len(prompt),
                       tokens=args.tokens, fresh_requests=len(report['requests']), decode_threads=args.threads,
                       prefill_threads=args.prefill_threads, sample_counts=parsed,
                       phase_limit=report['phase_limit'], attach_limit=report['attach_limit'])
        (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        pending_error = sys.exc_info()[0] is not None
        cleanup_errors = []
        cleanups = [lambda: stop_owned(recorder)]
        if client is not None:
            cleanups.append(client.close)
        cleanups.extend(stream.close for stream in sampler_logs)
        for cleanup in cleanups:
            try:
                cleanup()
            except BaseException as error:
                cleanup_errors.append(dict(type=type(error).__name__, message=str(error)))
        os.environ.clear(); os.environ.update(original_env)
        if cleanup_errors:
            report.update(completed=False, cleanup_errors=cleanup_errors)
            (directory / 'summary.json').unlink(missing_ok=True)
        save()
        if cleanup_errors and not pending_error:
            raise RuntimeError('Owned profiler cleanup failed; inspect local report')
    print(json.dumps({'completed': True, 'summary': str(directory/'summary.json')}))


if __name__ == '__main__':
    main()
