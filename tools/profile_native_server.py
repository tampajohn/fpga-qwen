"""Attach Time Profiler to the exact warmed Phase0 server (diagnostics only).

Run directly under the serving checkout's benchmark_guard.py. This helper starts
one owned Phase0NativeEngine, warms it, waits for xctrace's Darwin start
notification, then repeats the unchanged 85/452-token, 32-output protocol until
the recorder actually exits. No fallback engine, guessed CPU percentages, or
scoreboard claim. Trace inspection/CPU-time attribution is a separate step.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET

import benchmark_matched as matched
from benchmark_native_phase0 import (EXPECTED_PROMPT_LENGTHS, Phase0NativeEngine,
    freeze_build, guard, host_snapshot, sha, validate_result, verify_build)
from chat_protocol import ChatProtocol
from experiment import ROOT, locate_model


def clock_anchor():
    before = time.perf_counter_ns()
    wall = time.time_ns()
    after = time.perf_counter_ns()
    return dict(unix_time_ns=wall, utc=datetime.fromtimestamp(wall / 1e9, timezone.utc).isoformat(),
                perf_counter_before_ns=before, perf_counter_after_ns=after)


def estimated_unix_ns(perf_seconds, anchor):
    midpoint = (anchor['perf_counter_before_ns'] + anchor['perf_counter_after_ns']) // 2
    return anchor['unix_time_ns'] + round(perf_seconds * 1e9) - midpoint


class StartNotification:
    """Own one Darwin notification FD; notify_cancel closes its resources."""
    def __init__(self):
        self.library = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
        self.library.notify_register_file_descriptor.argtypes = [ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.library.notify_register_file_descriptor.restype = ctypes.c_uint32
        self.library.notify_cancel.argtypes = [ctypes.c_int]
        self.library.notify_cancel.restype = ctypes.c_uint32
        self.name = f'com.tampajohn.fpgaqwen.nativeprofile.{os.getpid()}.{uuid.uuid4().hex}'
        descriptor, token = ctypes.c_int(-1), ctypes.c_int(-1)
        status = self.library.notify_register_file_descriptor(self.name.encode(), ctypes.byref(descriptor),
                                                               0, ctypes.byref(token))
        if status:
            raise RuntimeError(f'Darwin tracing-start notification registration failed: {status}')
        self.descriptor, self.token = descriptor.value, token.value

    def wait(self, recorder, timeout=40):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            code = recorder.poll()
            if code is not None:
                raise RuntimeError(f'xctrace exited {code} before tracing-start notification; inspect recorder logs')
            ready, _, _ = select.select([self.descriptor], [], [], .1)
            if ready:
                payload = os.read(self.descriptor, 4)
                if len(payload) != 4:
                    raise RuntimeError('Incomplete Darwin tracing-start notification')
                return dict(observed=clock_anchor(), notification=self.name, payload_hex=payload.hex())
        raise TimeoutError('xctrace did not confirm tracing start within40seconds; no recording success assumed')

    def close(self):
        if self.token is not None:
            status = self.library.notify_cancel(self.token)
            self.token = None
            if status:
                raise RuntimeError(f'Darwin notification cancellation failed: {status}')


ORIGINAL_TIMING_RESULT = matched.timing_result


def timed_request(engine, prompt, expected):
    """Retain the existing HTTP implementation and capture its exact timestamps.

    The isolated driver is single-threaded. Its temporary timing-result hook
    runs after the SSE stream, leaves HTTP/EOS/cache behavior untouched, and is
    restored even on failure. It adds diagnostic bookkeeping, not neural work.
    """
    anchor = clock_anchor()
    captured = {}
    if matched.timing_result is not ORIGINAL_TIMING_RESULT:
        raise RuntimeError('Unexpected timing hook; refusing ambiguous instrumentation')

    def capture(start, arrivals, ids, text, stopped, backend_timings=None):
        captured.update(start=start, arrivals=list(arrivals))
        return ORIGINAL_TIMING_RESULT(start, arrivals, ids, text, stopped, backend_timings)

    matched.timing_result = capture
    try:
        result = engine.generate(prompt, 32, True)
    finally:
        matched.timing_result = ORIGINAL_TIMING_RESULT
    returned = time.perf_counter()
    after = clock_anchor()
    validate_result(result, prompt, 32, expected)
    if len(captured.get('arrivals', [])) != 32:
        raise RuntimeError('Missing token arrival timestamps')
    start, arrivals = captured['start'], captured['arrivals']
    if not start <= arrivals[0] <= arrivals[-1] <= returned:
        raise RuntimeError('Invalid monotonic request window')
    windows = dict(request_start=start, first_token=arrivals[0], last_token=arrivals[-1], request_returned=returned)
    result['profiling_windows'] = dict(
        perf_counter_seconds=windows,
        estimated_unix_ns={key: estimated_unix_ns(value, anchor) for key, value in windows.items()},
        relative_seconds={key: value - start for key, value in windows.items()},
        token_arrival_seconds_from_request=[value - start for value in arrivals],
        clock_before=anchor, clock_after=after,
        interpretation='Request-start to first-token includes prefill, first sampling and HTTP delivery; first-to-last-token is client-observed decode. UTC estimates use the bracketed clock correlation, not server-internal phase markers.')
    return result


def stop_recorder(process):
    """Signal only the process group created by this helper, then reap it."""
    if process is None:
        return None
    if process.poll() is not None:
        return process.returncode
    for sig, timeout in ((signal.SIGINT, 12), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
    raise RuntimeError('Owned xctrace process did not exit after cleanup signals')


def save(directory, report):
    pending = directory / 'report.pending.json'
    pending.write_text(json.dumps(report, indent=2) + '\n')
    pending.replace(directory / 'report.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--precision', choices=('f16', 'f32', 'q4_k_m'), required=True)
    parser.add_argument('--decode-threads', type=int, required=True)
    parser.add_argument('--prefill-threads', type=int, required=True)
    parser.add_argument('--profile', choices=('short', 'long'), required=True)
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'build/bench')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--expected-output-ids', type=Path)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--duration', type=int, default=25)
    parser.add_argument('--guard-state-dir', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    if (not 1 <= args.decode_threads <= 32 or not 1 <= args.prefill_threads <= 32
            or not 1 <= args.warmups <= 3 or not 20 <= args.duration <= 30):
        parser.error('Use1..32threads,1..3warmups and20..30seconds of tracing')
    guarded = guard(args.guard_state_dir)
    directory = args.directory.resolve()
    if directory.exists():
        parser.error('Choose a new directory; existing traces/reports are immutable')
    directory.mkdir(parents=True)
    report = dict(completed=False, diagnostic_only=True, performance_claim_eligible=False,
                  status='preparing', guard=guarded, gpu=False, pytorch=False, requests=[], warmups=[],
                  limitations=[
                      'Time Profiler perturbs execution. These runs are not part of the throughput scoreboard.',
                      'No CPU percentages are computed here. Sample availability and per-thread CPU attribution require trace inspection.',
                      'CPU samples are not critical-path wall-time fractions; sleeping/spinning workers and shared kernel symbols need separate interpretation.',
                      'Vocabulary and layer projections can share kernel symbols; this trace alone may not distinguish them.',
                      'Client HTTP arrival windows approximate server phases and may straddle trace boundaries; filter using stored clock correlations.'])
    engine = recorder = notification = None
    recorder_stdout = recorder_stderr = None
    try:
        build = freeze_build(args.build_dir)
        checkpoint = locate_model(args.checkpoint)
        protocol = ChatProtocol(checkpoint)
        prompt = matched.make_prompt(protocol, args.profile)
        if len(prompt) != EXPECTED_PROMPT_LENGTHS[args.profile]:
            raise RuntimeError('Historical prompt length changed')
        model = (args.models_dir / f'qwen3-0.6b-{args.precision}.gguf').resolve()
        expected = json.loads(args.expected_output_ids.read_text()) if args.expected_output_ids else None
        if expected is not None and (not isinstance(expected, list) or len(expected) != 32
                or any(type(token) is not int or not 0 <= token < 151936 for token in expected)):
            raise RuntimeError('Expected output IDs must contain32Qwen integer tokens')
        inputs = [model, *(checkpoint / name for name in ('tokenizer.json', 'tokenizer_config.json', 'config.json'))]
        if args.expected_output_ids:
            inputs.append(args.expected_output_ids.resolve())
        input_hashes = {str(path): sha(path) for path in inputs}
        sources = [Path(__file__).resolve(), ROOT / 'tools/benchmark_native_phase0.py',
                   ROOT / 'tools/benchmark_matched.py', ROOT / 'tools/benchmark_guard.py', ROOT / 'tools/chat_protocol.py']
        source_hashes = {str(path): sha(path) for path in sources}
        config = dict(name='profile-exact-server', build_dir=str(args.build_dir.resolve()),
                      decode_threads=args.decode_threads, prefill_threads=args.prefill_threads)
        report.update(build=build, config=config, precision=args.precision, profile=args.profile,
                      prompt_ids=prompt, output_tokens=32, input_sha256=input_hashes,
                      source_sha256=source_hashes, host_before=host_snapshot())
        engine = Phase0NativeEngine(args.precision, config, build, model, directory / 'server')
        report['server_startup_seconds'] = engine.start()
        report['server'] = dict(pid=engine.process.pid, command=engine.command, metadata=engine.metadata,
                                removed_environment_keys=engine.removed_environment_keys, log=str(engine.log_path))
        for _ in range(args.warmups):
            result = timed_request(engine, prompt, expected)
            if expected is None:
                expected = result['output_token_ids']
            report['warmups'].append(result)
        report['expected_output_ids'] = expected
        report['status'] = 'waiting-for-tracing-start'
        notification = StartNotification()
        trace = directory / 'time-profiler.trace'
        command = ['xcrun', 'xctrace', 'record', '--template', 'Time Profiler', '--attach', str(engine.process.pid),
                   '--time-limit', f'{args.duration}s', '--output', str(trace),
                   '--notify-tracing-started', notification.name, '--no-prompt']
        recorder_stdout = (directory / 'recorder.stdout.log').open('w')
        recorder_stderr = (directory / 'recorder.stderr.log').open('w')
        report['recorder'] = dict(command=command, requested_duration_seconds=args.duration,
                                  launched=clock_anchor(), trace=str(trace))
        recorder = subprocess.Popen(command, stdout=recorder_stdout, stderr=recorder_stderr, start_new_session=True)
        report['recorder']['pid'] = recorder.pid
        save(directory, report)
        report['recorder']['start_notification'] = notification.wait(recorder)
        started = time.perf_counter()
        report['status'] = 'recording'
        save(directory, report)
        while recorder.poll() is None and time.perf_counter() - started < args.duration:
            if engine.process.poll() is not None:
                raise RuntimeError('Owned native server exited during profiling')
            result = timed_request(engine, prompt, expected)
            result['recorder_alive_after_request'] = recorder.poll() is None
            report['requests'].append(result)
            save(directory, report)
        workload_finished = time.perf_counter()
        request_windows = [row['profiling_windows']['perf_counter_seconds'] for row in report['requests']]
        workload_span = (request_windows[-1]['request_returned'] - request_windows[0]['request_start']) if request_windows else 0
        report['workload_window'] = dict(
            notification_observed_perf_counter=started,
            first_request_start_perf_counter=request_windows[0]['request_start'] if request_windows else None,
            last_request_returned_perf_counter=request_windows[-1]['request_returned'] if request_windows else None,
            submission_loop_finished_perf_counter=workload_finished,
            request_span_seconds=workload_span,
            elapsed_before_recorder_wait_seconds=workload_finished - started,
            recording_sample_coverage_validated=False,
            interpretation='Observed client request span only. Recorder finalization is excluded; actual sampled recording coverage still requires trace analysis.')
        # Stop submitting work after the requested window. Finalizing a trace
        # may take longer; wait on the actual recorder rather than infer exit.
        try:
            code = recorder.wait(timeout=40)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError('xctrace did not finish/finalize within40seconds after the workload window') from error
        recorder_finished = time.perf_counter()
        report['recorder'].update(exit_code=code, exited=clock_anchor(),
                                  lifetime_after_start_notification_seconds=recorder_finished - started,
                                  finalization_wait_seconds=recorder_finished - workload_finished,
                                  recording_duration_validated=False)
        if code != 0:
            raise RuntimeError(f'xctrace failed with exit{code}; inspect recorder logs')
        if not report['requests'] or workload_span < args.duration - 2:
            raise RuntimeError('Observed client request span is shorter than the requested workload window; finalization time does not count')
        if not trace.exists():
            raise RuntimeError('xctrace exited successfully but produced no trace artifact')
        # The recorder is reaped before parsing/hashing its trace. No arbitrary
        # sleep is substituted for completion, and the owned server is now idle.
        report['status'] = 'validating-trace'
        save(directory, report)
        toc = directory / 'trace-toc.xml'
        toc_command = ['xcrun', 'xctrace', 'export', '--input', str(trace), '--toc', '--output', str(toc)]
        with (directory / 'toc.stdout.log').open('w') as stdout, (directory / 'toc.stderr.log').open('w') as stderr:
            exported = subprocess.run(toc_command, stdout=stdout, stderr=stderr, timeout=30)
        report['toc_export'] = dict(command=toc_command, exit_code=exported.returncode)
        if exported.returncode != 0 or not toc.is_file():
            raise RuntimeError('Trace TOC export failed; recording is not validated')
        tree = ET.parse(toc)
        runs = list(tree.iter('run'))
        if not runs:
            raise RuntimeError('Trace TOC contains no recording runs')
        report['toc_export'].update(run_count=len(runs), sha256=sha(toc),
            table_schemas=sorted({node.get('schema') for node in tree.iter('table') if node.get('schema')}),
            processes=[dict(node.attrib) for node in tree.iter('process')],
            sample_counts_validated=False)
        files = sorted(path for path in trace.rglob('*') if path.is_file()) if trace.is_dir() else [trace]
        if not files or sum(path.stat().st_size for path in files) == 0:
            raise RuntimeError('Trace artifact is empty')
        report['trace_files'] = {str(path.relative_to(directory)): dict(bytes=path.stat().st_size, sha256=sha(path)) for path in files}
        verify_build(build)
        if input_hashes != {str(path): sha(path) for path in inputs} or source_hashes != {str(path): sha(path) for path in sources}:
            raise RuntimeError('Inputs or diagnostic source changed during profiling')
        report['guard_after'] = guard(args.guard_state_dir)
        report['host_after'] = host_snapshot()
        report['status'] = 'recorded-needs-sample-analysis'
        report['completed'] = True
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        cleanup_errors = []
        try:
            code = stop_recorder(recorder)
            if recorder is not None:
                report.setdefault('recorder', {})['final_exit_code'] = code
        except Exception as error:
            cleanup_errors.append('recorder: ' + str(error))
        if engine is not None:
            try:
                engine.close()
            except Exception as error:
                cleanup_errors.append('server: ' + str(error))
        if notification is not None:
            try:
                notification.close()
            except Exception as error:
                cleanup_errors.append('notification: ' + str(error))
        for log in (recorder_stdout, recorder_stderr):
            if log is not None:
                try:
                    log.close()
                except Exception as error:
                    cleanup_errors.append('log: ' + str(error))
        report['cleanup_errors'] = cleanup_errors
        if cleanup_errors:
            report['completed'] = False
            report['status'] = 'cleanup-failed'
        save(directory, report)
    if not report['completed']:
        raise RuntimeError('Profiling did not complete cleanly; inspect report.json')
    print(directory / 'report.json')


if __name__ == '__main__':
    main()
