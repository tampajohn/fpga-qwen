#!/usr/bin/env python3
"""Same-binary real-adapter greedy suffix off/on confirmation; no llama claim."""
import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import statistics
import threading
import time

from analyze_native_phase0 import interval
from benchmark_matched import make_prompt
from benchmark_native_phase0 import guard, host_snapshot, sha
from chat_protocol import ChatProtocol, ChatRequest
from experiment import ROOT, locate_model
from native_cpu_chat import NativeCPUChat, EOS


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def save(path, report):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


class ObservedChat(NativeCPUChat):
    def command(self, command, cancel, *args, **kwargs):
        result = super().command(command, cancel, *args, **kwargs)
        operation = command.split(' ', 1)[0]
        if operation in ('prefill', 'step', 'verify-greedy'):
            self.calls[operation] += 1
            if operation == 'verify-greedy':
                self.predicted.extend(result['tokens'])
                self.verified += result['verified_positions']
                self.committed += result['committed_inputs']
            else:
                self.predicted.append(result['token'])
        return result

    def generate(self, request):
        self.predicted = []
        self.calls = dict(prefill=0, step=0, **{'verify-greedy': 0})
        self.verified = self.committed = 0
        done = None
        for event in self.stream(request, threading.Event()):
            if event['type'] == 'done':
                require(done is None, 'Duplicate done event')
                done = event
        require(done is not None and done['finish_reason'] in ('eos', 'length'), 'Incomplete generation')
        ids = done['token_ids']
        eos = done['finish_reason'] == 'eos'
        require(len(ids) >= 2, 'Fewer than two visible tokens; decode rate is not interpretable')
        require(self.predicted == ids+([self.predicted[-1]] if eos else [])
                and not any(token in EOS for token in ids), 'Predicted/visible token stream mismatch')
        require((eos and self.predicted[-1] in EOS and len(ids) < request.max_new)
                or (not eos and len(ids) == request.max_new), 'Invalid natural EOS/cap behavior')
        metrics = done['metrics']
        require(metrics['suffix_enabled'] is bool(request.max_draft)
                and metrics['reused_prefix_tokens'] == 0
                and metrics['evaluated_prompt_tokens'] == len(request.ids)
                and metrics['output_tokens'] == len(ids), 'Actual suffix/prompt/output configuration mismatch')
        require(metrics['verification_calls'] == self.calls['verify-greedy']
                and metrics['verified_positions'] == self.verified
                and metrics['proposed_tokens'] == self.calls['verify-greedy']
                and metrics['accepted_tokens']+metrics['rejected_tokens'] == metrics['proposed_tokens'],
                'Verification work accounting mismatch')
        result = dict(token_ids=ids, predicted_ids_including_eos=list(self.predicted),
                      text=done['text'], finish_reason=done['finish_reason'], eos_observed=eos,
                      output_tokens=len(ids), prompt_tokens=len(request.ids),
                      first_token_seconds=metrics['first_token_ms']/1000,
                      steady_decode_tokens_per_second=metrics['decode_tokens_per_second'],
                      request_seconds=metrics['elapsed_seconds'], adapter_metrics=metrics,
                      neural_commands=dict(self.calls), neural_call_count=sum(self.calls.values()),
                      proposed_tokens=metrics['proposed_tokens'], accepted_tokens=metrics['accepted_tokens'],
                      rejected_tokens=metrics['rejected_tokens'], verified_positions=self.verified,
                      committed_verified_inputs=self.committed,
                      wasted_verified_positions=self.verified-self.committed)
        for key in ('first_token_seconds', 'steady_decode_tokens_per_second', 'request_seconds'):
            require(math.isfinite(result[key]) and result[key] > 0, 'Invalid timing: '+key)
        return result


def identity(result):
    return {key: result[key] for key in ('token_ids', 'predicted_ids_including_eos', 'text', 'finish_reason', 'eos_observed')}


def analyze(report):
    rng = random.Random(20261112)
    rounds = report['rounds']
    samples = [[rng.randrange(rounds) for _ in range(rounds)] for _ in range(10000)]
    visits = {(row['round'], row['configuration']): row for row in report['visits']}
    require(len(visits) == rounds*2, 'Missing or duplicate scored visits')
    comparisons = []
    for profile in report['workloads']:
        pairs = []
        for index in range(rounds):
            off, on = visits[index, 'suffix-off'], visits[index, 'suffix-on']
            require(off['order_index'] == index % 2 and on['order_index'] == 1-index % 2,
                    'Configuration order did not alternate')
            baseline, candidate = off['profiles'][profile]['scored'], on['profiles'][profile]['scored']
            pairs.append(dict(round=index, off_first=off['order_index'] == 0,
                baseline=baseline, candidate=candidate,
                ratios=dict(decode_speedup=candidate['steady_decode_tokens_per_second']/baseline['steady_decode_tokens_per_second'],
                            ttft_speedup=baseline['first_token_seconds']/candidate['first_token_seconds'],
                            request_speedup=baseline['request_seconds']/candidate['request_seconds'])))
        comparisons.append(dict(profile=profile, raw_pairs=pairs,
            medians={label: {metric: statistics.median(row[side][metric] for row in pairs)
                for metric in ('first_token_seconds', 'steady_decode_tokens_per_second', 'request_seconds')}
                for label, side in (('suffix-off', 'baseline'), ('suffix-on', 'candidate'))},
            metrics={metric: interval([row['ratios'][metric] for row in pairs], samples)
                     for metric in ('decode_speedup', 'ttft_speedup', 'request_speedup')}))
    return dict(comparisons=comparisons,
                bootstrap=dict(seed=20261112, resamples=10000, confidence_level=.95,
                               method='Paired percentile bootstrap of median speedup; same round indices for every metric/profile'),
                limitation='Unadjusted separate workload intervals; no pooled or cross-workload speedup.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--correctness-report', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--short-prompt', type=Path, default=ROOT/'build/native-phase0/prompt-short.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--prefill-threads', type=int, default=6)
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT/'build/benchmark-status')
    args = parser.parse_args()
    require(10 <= args.rounds <= 100 and args.rounds % 2 == 0, 'Use10..100 even paired rounds')
    require(2 <= args.warmups <= 10, 'Use2..10 symmetric warmups')
    require(1 <= args.threads <= 64 and 1 <= args.prefill_threads <= 64, 'Use1..64 workers')
    guarded = guard(args.guard_state_dir)
    output, binary, weights = args.output.resolve(), args.binary.resolve(), args.weights.resolve()
    logs = output.parent/(output.stem+'-logs')
    require(not output.exists() and not logs.exists(), 'Preserve prior output/logs')
    logs.mkdir(parents=True)
    report = dict(completed=False, paired_sampling_complete=False, performance_claim_eligible=False,
        mode='confirmation', gpu=False, pytorch=False, rounds=args.rounds, output_token_cap=64,
        warmups_per_visit_profile=args.warmups, threads=args.threads, prefill_threads=args.prefill_threads,
        configurations=[dict(name='suffix-off', max_draft=0), dict(name='suffix-on', max_draft=1)],
        preflight=[], visits=[], guard_before=guarded, host_before=host_snapshot(),
        comparison_scope='Same candidate binary, weights, threads, greedy sampler and real browser adapter; explicit suffix off/on.',
        timing_scope='Fresh-prefix request wall time including reset, mode, prefill, suffix proposal/IPC and text decoding. Startup and symmetric warmups excluded.',
        limitations=['64 is a natural-EOS output cap, not a forced length. Exact IDs/text/EOS must match on every warmup and scored request.',
                     'One process per visit, with reuse_prefix=False. No prefix priming or retained-prefix benefit.',
                     'Wasted verified positions counts computed verification rows whose input was not committed; not a wall-time attribution.',
                     'OS scheduling/background activity remain uncontrolled. No inferred llama, full-model quality, or combined-workload win.'])
    removed_environment = {}
    try:
        for name in list(os.environ):
            if name.startswith(('DYLD_', 'GGML_', 'LLAMA_ARG_')) or name == 'VECLIB_MAXIMUM_THREADS':
                removed_environment[name] = os.environ.pop(name)
        report['removed_environment_names'] = sorted(removed_environment)
        report['requested_veclib_maximum_threads'] = args.prefill_threads
        checkpoint = locate_model(args.checkpoint)
        protocol = ChatProtocol(checkpoint)
        inputs = [binary, binary.parent/'manifest.json', weights, Path(str(weights)+'.index'),
                  Path(str(weights)+'.manifest.json'), args.short_prompt.resolve(), args.correctness_report.resolve()]
        inputs += [checkpoint/name for name in ('tokenizer.json', 'tokenizer_config.json', 'config.json')]
        sources = [Path(__file__).resolve()] + [ROOT/'tools'/name for name in (
            'native_cpu_chat.py', 'suffix.py', 'chat_protocol.py', 'benchmark_matched.py',
            'benchmark_native_phase0.py', 'benchmark_guard.py', 'analyze_native_phase0.py', 'experiment.py')]
        report['inputs'] = {str(path): sha(path) for path in inputs}
        report['source_hashes'] = {str(path): sha(path) for path in sources}
        build = json.loads((binary.parent/'manifest.json').read_text())
        pack = json.loads(Path(str(weights)+'.manifest.json').read_text())
        require(build.get('completed') is True and build.get('gpu') is False and build.get('pytorch') is False
                and build.get('binary_sha256') == sha(binary), 'Build provenance mismatch')
        require(pack.get('completed') is True and pack.get('format') == 'QWCPU01'
                and pack['binary']['sha256'] == sha(weights)
                and pack['index']['sha256'] == sha(Path(str(weights)+'.index')), 'Pack provenance mismatch')
        quality = json.loads(args.correctness_report.read_text())
        require(quality.get('completed') is True and quality.get('passed') is True, 'Suffix correctness gate not passed')
        for path in (binary, weights, ROOT/'tools/native_cpu_chat.py', ROOT/'tools/suffix.py'):
            require(quality.get('hashes', {}).get(str(path)) == sha(path), 'Correctness scope mismatch: '+str(path))
        report.update(native_build=build, correctness_scope=quality.get('scope'))
        short_ids = json.loads(args.short_prompt.read_text())
        require(isinstance(short_ids, list) and len(short_ids) == 85 and short_ids == make_prompt(protocol, 'short'),
                'Short fixture must match complete existing85-token standard prompt')
        repeat_messages = [dict(role='user', content='Write the exact sequence red blue green twenty times, without commentary.')]
        repeat = protocol.prepare(dict(session='suffix-repeat', messages=repeat_messages,
                                       max_new_tokens=64, decoding='greedy', max_draft=0))
        short = ChatRequest(session='suffix-low-repetition', ids=list(short_ids), max_new=64, max_draft=0, decoding='greedy')
        profiles = {'repeat': repeat, 'low-repetition': short}
        report['workloads'] = {name: dict(ids=list(req.ids), prompt_tokens=len(req.ids), output_cap=64,
            messages=repeat_messages if name == 'repeat' else [dict(role='user', content='Explain memory bandwidth and caching in four concise sentences.')])
            for name, req in profiles.items()}
        expected = {}
        save(output, report)
        def visit(config, phase, index=None, order=None):
            guard(args.guard_state_dir)
            directory = logs/(phase if index is None else f'round-{index:03d}')/config['name']
            directory.mkdir(parents=True)
            client = ObservedChat(protocol, binary=binary, weights=weights, threads=args.threads,
                                  prefill_threads=args.prefill_threads, reuse_prefix=False)
            record = dict(configuration=config['name'], phase=phase, round=index, order_index=order, profiles={})
            report['preflight' if phase == 'preflight' else 'visits'].append(record)
            log_path = binary.parent/'chat.log'
            offset = log_path.stat().st_size if log_path.exists() else 0
            try:
                started = time.perf_counter();client.start(threading.Event())
                record.update(startup_seconds=time.perf_counter()-started, pid=client.process.pid, ready=client.ready)
                require(client.suffix_supported(), 'B2 suffix handshake unavailable')
                for name, req in profiles.items():
                    request = replace(req, max_draft=config['max_draft'])
                    row = dict(warmups=[]);record['profiles'][name] = row
                    for warmup in range(args.warmups+1):
                        result = client.generate(request)
                        observed = identity(result)
                        if name not in expected:
                            require(phase == 'preflight' and config['name'] == 'suffix-off', 'Missing baseline preflight')
                            expected[name] = observed
                        require(observed == expected[name], 'Exact IDs/text/EOS changed: '+name)
                        if warmup < args.warmups:
                            row['warmups'].append(result)
                        else:
                            row['scored'] = result
                        save(output, report)
            finally:
                client.close()
                if log_path.exists():
                    with log_path.open('rb') as stream:
                        stream.seek(offset);(directory/'native.log').write_bytes(stream.read())
                    record['runtime_log'] = dict(path=str(directory/'native.log'), sha256=sha(directory/'native.log'))
                save(output, report)
        for config in report['configurations']:
            visit(config, 'preflight')
        report.update(preflight_exact_ids_text_and_eos_passed=True, expected_outputs=expected)
        require(report['preflight'][1]['profiles']['repeat']['scored']['proposed_tokens'] > 0,
                'Repeat fixture did not exercise verification; do not silently replace it')
        for index in range(args.rounds):
            order = report['configurations'] if index % 2 == 0 else list(reversed(report['configurations']))
            for position, config in enumerate(order):
                visit(config, 'scored', index, position)
        report['analysis'] = analyze(report)
        require(all(sha(Path(path)) == digest for path, digest in report['inputs'].items())
                and all(sha(Path(path)) == digest for path, digest in report['source_hashes'].items()), 'Frozen inputs/helpers changed')
        report.update(completed=True, paired_sampling_complete=True, host_after=host_snapshot(), guard_after=guard(args.guard_state_dir))
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error));raise
    finally:
        os.environ.update(removed_environment)
        save(output, report)
    print(output)


if __name__ == '__main__':
    main()
