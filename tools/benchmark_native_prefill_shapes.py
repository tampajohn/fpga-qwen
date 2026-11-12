#!/usr/bin/env python3
"""Paired retained-prefix tail-prefill diagnostic; not fresh TTFT or llama timing."""
import argparse
import array
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import threading
import time

from analyze_native_phase0 import interval
from benchmark_native_phase0 import guard, host_snapshot, sha
from experiment import ROOT
from native_cpu_chat import NativeCPUChat

SUITES = {
    'original': {0: (2, 16, 85, 452), 3584: (2, 16, 64, 384)},
    'dispatch-validation': {0: (32, 63, 64, 65, 128, 256, 512),
                            448: (64,), 449: (63, 64), 3584: (2, 16, 64, 384)},
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def save(path, report):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def snapshot(client, directory, label):
    result = {}
    for kind, command, count in (('logits', 'dump', 151936), ('hidden', 'dump-hidden', 1024)):
        path = directory/f'{label}-{kind}.f32'
        require(not path.exists(), 'Snapshot destination exists')
        ack = client.command(f'{command} {path}', threading.Event())
        require(ack.get('ok') is True and ack.get('count') == count, 'Dump acknowledgement mismatch')
        data = path.read_bytes()
        values = array.array('f');values.frombytes(data)
        require(sys.byteorder == 'little' and len(values) == count and len(data) == count*4
                and all(math.isfinite(value) for value in values), 'Invalid full-array snapshot')
        result[kind] = dict(path=str(path), sha256=sha(path), bytes=len(data), position=ack['position'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-binary', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--prompt', type=Path, default=ROOT/'build/native-cpu/extended-tutorial-fixtures/context3968.json')
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--suite', choices=tuple(SUITES), default='original',
                        help='Predeclared shape suite; original preserves the initial eight-shape diagnostic')
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--prefill-threads', type=int, default=6)
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT/'build/benchmark-status')
    args = parser.parse_args()
    require(10 <= args.rounds <= 100 and args.rounds % 2 == 0, 'Use10..100 even paired rounds')
    require(2 <= args.warmups <= 10, 'Use2..10 symmetric warmups')
    require(1 <= args.threads <= 64 and 1 <= args.prefill_threads <= 64, 'Use1..64 workers')
    shapes = SUITES[args.suite]
    guarded = guard(args.guard_state_dir)
    directory = args.directory.resolve();directory.mkdir(parents=True, exist_ok=False)
    output = directory/'report.json'
    binaries = {'reference': args.reference_binary.resolve(), 'candidate': args.binary.resolve()}
    require(binaries['reference'] != binaries['candidate'], 'Use distinct immutable build paths')
    weights = args.weights.resolve()
    report = dict(completed=False, paired_sampling_complete=False, performance_claim_eligible=False,
        gpu=False, pytorch=False, diagnostic_only=True, rounds=args.rounds, warmups=args.warmups,
        suite=args.suite, declared_shapes={str(prefix): list(tails) for prefix, tails in shapes.items()},
        threads=args.threads, prefill_threads=args.prefill_threads, preflight=[], priming=[], shapes=[],
        guard_before=guarded, host_before=host_snapshot(),
        timing_scope='Only prefill tail command: CLI internal seconds and pipe round-trip wall seconds. Prefix priming, rewind, startup, dump IPC and validation are excluded.',
        limitations=['These token slices are numerical prefill-shape fixtures, not complete user messages or conversation quality tests.',
                     'Two processes remain resident; all priming and measured inference executes serially.',
                     'Each nonzero prefix is primed once per process per phase, then retained through rewind; this is not fresh-request TTFT.',
                     'Prefixes448/449 are numerical API fixtures, not canonical512-block browser prefix reuse. Both engines consume identical complete prefix IDs before tail execution.',
                     'Reference and candidate may differ in resident memory; OS activity/placement remain uncontrolled.',
                     'Separate shape intervals are unadjusted; no threshold is chosen from the fastest observed shape and no pooled/llama speed claim.'])
    clients = {}
    logs = {}
    removed_environment = {}
    try:
        for name in list(os.environ):
            if name.startswith(('DYLD_', 'GGML_', 'LLAMA_ARG_')) or name == 'VECLIB_MAXIMUM_THREADS':
                removed_environment[name] = os.environ.pop(name)
        report['removed_environment_names'] = sorted(removed_environment)
        report['requested_veclib_maximum_threads'] = args.prefill_threads
        ids = json.loads(args.prompt.read_text())
        require(isinstance(ids, list) and len(ids) == 3968
                and all(type(token) is int and 0 <= token < 151936 for token in ids), 'Expected complete3968-token source fixture')
        report['source_fixture'] = dict(path=str(args.prompt.resolve()), ids=ids,
            selection='Tail IDs are source_ids[P:P+B], prefix IDs source_ids[:P], without token insertion.')
        paths = [weights, Path(str(weights)+'.index'), Path(str(weights)+'.manifest.json'), args.prompt.resolve()]
        for binary in binaries.values():
            paths.extend((binary, binary.parent/'manifest.json'))
        report['inputs'] = {str(path): sha(path) for path in paths}
        sources = [Path(__file__).resolve()]+[ROOT/'tools'/name for name in (
            'native_cpu_chat.py', 'suffix.py', 'benchmark_native_phase0.py', 'benchmark_guard.py',
            'analyze_native_phase0.py', 'experiment.py', 'benchmark_matched.py', 'chat_protocol.py')]
        report['source_hashes'] = {str(path): sha(path) for path in sources}
        pack = json.loads(Path(str(weights)+'.manifest.json').read_text())
        require(pack.get('completed') is True and pack.get('format') == 'QWCPU01'
                and pack['binary']['sha256'] == sha(weights)
                and pack['index']['sha256'] == sha(Path(str(weights)+'.index')), 'Pack provenance mismatch')
        report['builds'] = {}
        for label, binary in binaries.items():
            build = json.loads((binary.parent/'manifest.json').read_text())
            require(build.get('completed') is True and build.get('gpu') is False and build.get('pytorch') is False
                    and build.get('binary_sha256') == sha(binary), 'Build provenance mismatch')
            report['builds'][label] = build
            client = NativeCPUChat(None, binary=binary, weights=weights, threads=args.threads,
                                   prefill_threads=args.prefill_threads, reuse_prefix=False)
            clients[label] = client
            path = binary.parent/'chat.log';logs[label] = (path, path.stat().st_size if path.exists() else 0)
            started = time.perf_counter();client.start(threading.Event())
            require(client.prefix_reuse_supported(), 'Canonical rewind capability required')
            report.setdefault('startup', {})[label] = dict(seconds=time.perf_counter()-started, pid=client.process.pid, ready=client.ready)
        save(output, report)
        def prime(prefix, phase):
            for label, client in clients.items():
                reset = client.command('reset', threading.Event())
                require(reset.get('position') == 0 and reset.get('ok') is True, 'Reset failed')
                require(client.command('mode greedy', threading.Event()).get('ok') is True, 'Greedy mode failed')
                if prefix:
                    response = client.command('prefill '+' '.join(map(str, ids[:prefix])), threading.Event())
                    require(response.get('position') == prefix, 'Priming position mismatch')
                report['priming'].append(dict(phase=phase, configuration=label, prefix_tokens=prefix,
                    full_prefill_executed=bool(prefix), outside_timing=True))
        def run_tail(client, prefix, batch, command):
            ack = client.command(f'rewind {prefix}', threading.Event())
            require(ack.get('ok') is True and ack.get('position') == prefix, 'Rewind failed')
            begin = time.perf_counter();response = client.command(command, threading.Event())
            elapsed = time.perf_counter()-begin
            require(response.get('position') == prefix+batch and type(response.get('token')) is int
                    and 0 <= response['token'] < 151936, 'Prefill result position/token mismatch')
            seconds = response.get('seconds')
            require(type(seconds) in (float, int) and math.isfinite(seconds) and seconds > 0
                    and math.isfinite(elapsed) and elapsed > 0, 'Invalid command duration')
            return dict(token=response['token'], position=response['position'], cli_seconds=seconds, wall_seconds=elapsed)
        expected = {}
        # Entire shape corpus must pass before any warmup/scored timing phase.
        for prefix, batches in shapes.items():
            prime(prefix, 'preflight')
            for batch in batches:
                guard(args.guard_state_dir)
                name = f'P{prefix}-B{batch}'
                command = 'prefill '+' '.join(map(str, ids[prefix:prefix+batch]))
                row = dict(shape=name, prefix=prefix, tail=batch, configurations={})
                report['preflight'].append(row)
                for label, client in clients.items():
                    result = run_tail(client, prefix, batch, command)
                    arrays = snapshot(client, directory, f'{name}-{label}')
                    for artifact in arrays.values():
                        report['inputs'][artifact['path']] = artifact['sha256']
                    require(all(a['position'] == prefix+batch for a in arrays.values()), 'Dump position changed')
                    row['configurations'][label] = dict(token=result['token'], position=result['position'], arrays=arrays)
                reference, candidate = (row['configurations'][label] for label in ('reference','candidate'))
                require(reference['token'] == candidate['token'], 'Preflight predicted ID mismatch')
                for kind in ('logits', 'hidden'):
                    require(reference['arrays'][kind]['sha256'] == candidate['arrays'][kind]['sha256'],
                            'Preflight full-array bit mismatch: '+name+'/'+kind)
                row['passed'] = True;expected[name] = reference['token'];save(output, report)
        report['all_shapes_exact_before_timing'] = True
        save(output, report)
        for prefix, batches in shapes.items():
            prime(prefix, 'timing')
            for batch in batches:
                name = f'P{prefix}-B{batch}'
                command = 'prefill '+' '.join(map(str, ids[prefix:prefix+batch]))
                shape = dict(shape=name, prefix=prefix, tail=batch, warmups=[], pairs=[])
                report['shapes'].append(shape)
                for index in range(args.warmups+args.rounds):
                    guard(args.guard_state_dir)
                    order = ['reference','candidate'] if index % 2 == 0 else ['candidate','reference']
                    pair = dict(order=order, results={})
                    for label in order:
                        result = run_tail(clients[label], prefix, batch, command)
                        require(result['token'] == expected[name], 'Visit predicted ID changed: '+name)
                        pair['results'][label] = result
                    if index < args.warmups:
                        pair['warmup'] = index;shape['warmups'].append(pair)
                    else:
                        pair['round'] = index-args.warmups
                        pair['ratios'] = {metric: pair['results']['reference'][metric]/pair['results']['candidate'][metric]
                                          for metric in ('cli_seconds','wall_seconds')}
                        shape['pairs'].append(pair)
                    save(output, report)
        rng = random.Random(20261112)
        samples = [[rng.randrange(args.rounds) for _ in range(args.rounds)] for _ in range(10000)]
        for shape in report['shapes']:
            shape['analysis'] = {metric: dict(paired_speedup=interval([p['ratios'][metric] for p in shape['pairs']], samples),
                medians={label:statistics.median(p['results'][label][metric] for p in shape['pairs'])
                         for label in ('reference','candidate')}) for metric in ('cli_seconds','wall_seconds')}
        report['bootstrap'] = dict(seed=20261112,resamples=10000,confidence_level=.95,
            method='Paired-round percentile bootstrap of median reference/candidate command latency ratio')
        require(all(sha(Path(path)) == digest for path, digest in report['inputs'].items())
                and all(sha(Path(path)) == digest for path, digest in report['source_hashes'].items()), 'Frozen inputs/helpers changed')
        report.update(completed=True, paired_sampling_complete=True, host_after=host_snapshot(), guard_after=guard(args.guard_state_dir))
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__,message=str(error));raise
    finally:
        for label, client in clients.items():
            try:
                client.close()
            except BaseException as error:
                report.update(completed=False, paired_sampling_complete=False, cleanup_error=str(error))
            if label in logs:
                path, offset = logs[label]
                if path.exists():
                    with path.open('rb') as stream:
                        stream.seek(offset);(directory/f'{label}.log').write_bytes(stream.read())
                    report.setdefault('runtime_logs', {})[label] = dict(path=str(directory/f'{label}.log'), sha256=sha(directory/f'{label}.log'))
        os.environ.update(removed_environment)
        save(output, report)
    require(report['completed'], 'Cleanup failed')
    print(output)


if __name__ == '__main__':
    main()
