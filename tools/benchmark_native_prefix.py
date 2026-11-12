#!/usr/bin/env python3
"""Paired same-binary browser-adapter prefix reuse benchmark; not a llama claim."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics
import threading
import time

from analyze_native_phase0 import interval
from benchmark_native_phase0 import guard, host_snapshot, sha
from chat_protocol import ChatProtocol
from experiment import ROOT, locate_model
from native_cpu_chat import NativeCPUChat, EOS


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def common_prefix(left, right):
    for index, pair in enumerate(zip(left, right)):
        if pair[0] != pair[1]:
            return index
    return min(len(left), len(right))


def workloads(protocol, count):
    result = {}
    for length in (1024, 2048, 3968):
        padding = 0
        for _ in range(10):
            prefix = 'Background notes:' + ' x' * padding
            instruction = ('\nIgnore the background notes. Write 200 consecutive integers '
                           'in increasing order, one integer per line. Do not skip numbers or abbreviate the list. ')
            target_messages = [dict(role='user', content=prefix + instruction + 'Start at 2.')]
            session = f'prefix-context-{length}'
            target = protocol.prepare(dict(session=session, messages=target_messages,
                                           max_new_tokens=count, decoding='greedy'))
            if len(target.ids) == length:
                break
            padding += length - len(target.ids)
            require(padding >= 0, 'Prompt template exceeds target length')
        else:
            raise RuntimeError('Unable to tokenize exact prefix benchmark prompt')
        prime_messages = [dict(role='user', content=prefix + instruction + 'Start at 1.')]
        prime = protocol.prepare(dict(session=session, messages=prime_messages,
                                      max_new_tokens=count, decoding='greedy'))
        shared = common_prefix(prime.ids, target.ids)
        reuse = min(shared, len(prime.ids) // 512 * 512, len(target.ids) - 1) // 512 * 512
        require(reuse >= 512 and prime.ids != target.ids, 'Fixture must edit a tail after a canonical prefix')
        result[f'context{length}'] = dict(prime=prime, target=target,
            evidence=dict(prime_messages=prime_messages, target_messages=target_messages,
                          prime_ids=prime.ids, target_ids=target.ids,
                          prime_tokens=len(prime.ids), target_tokens=len(target.ids),
                          common_prefix_tokens=shared, expected_reused_tokens=reuse,
                          expected_evaluated_tokens=len(target.ids)-reuse,
                          modification='Same-session user-message tail changed from Start at 1. to Start at 2.'))
    return result


class ObservedChat(NativeCPUChat):
    """Observe actual predicted IDs, including EOS, without extra IPC or logits."""
    def command(self, command, cancel, *args, **kwargs):
        result = super().command(command, cancel, *args, **kwargs)
        if command.startswith(('prefill ', 'step ')):
            self.predictions.append(result['token'])
        return result

    def generate(self, request):
        self.predictions = []
        done = None
        for event in self.stream(request, threading.Event()):
            if event['type'] == 'done':
                require(done is None, 'Duplicate done event')
                done = event
        require(done is not None and done['finish_reason'] in ('length', 'eos'), 'Incomplete generation')
        generated = done.get('token_ids')
        eos = done['finish_reason'] == 'eos'
        require(isinstance(generated, list) and len(generated) >= 2,
                'Workload produced fewer than two visible tokens; no decode-rate comparison is valid')
        require(self.predictions == generated + ([self.predictions[-1]] if eos else []),
                'Prediction stream and generated IDs differ')
        require((eos and self.predictions[-1] in EOS) or
                (not eos and len(generated) == request.max_new and not any(t in EOS for t in generated)),
                'Invalid EOS or output-length behavior')
        metrics = done['metrics']
        result = dict(output_token_ids=list(generated), predicted_ids_including_eos=list(self.predictions),
                      output_token_count=len(generated), finish_reason=done['finish_reason'],
                      eos_observed=eos, text=done['text'], prompt_tokens=len(request.ids),
                      reused_prefix_tokens=metrics['reused_prefix_tokens'],
                      evaluated_prompt_tokens=metrics['evaluated_prompt_tokens'],
                      first_token_seconds=metrics['first_token_ms']/1000,
                      steady_decode_tokens_per_second=metrics['decode_tokens_per_second'],
                      request_seconds=metrics['elapsed_seconds'], adapter_metrics=metrics)
        require(metrics['output_tokens'] == len(generated), 'Adapter output count changed')
        for key in ('first_token_seconds', 'steady_decode_tokens_per_second', 'request_seconds'):
            require(math.isfinite(result[key]) and result[key] > 0, 'Invalid timing: ' + key)
        return result


def validate(result, expected, prompt_length, reuse):
    require(result['prompt_tokens'] == prompt_length and result['reused_prefix_tokens'] == reuse
            and result['evaluated_prompt_tokens'] == prompt_length-reuse,
            'Scored prompt or actual reuse/evaluation count changed')
    keys = ('output_token_ids', 'predicted_ids_including_eos', 'finish_reason', 'eos_observed', 'text')
    identity = {key: result[key] for key in keys}
    if expected is not None:
        require(identity == expected, 'Exact output IDs, EOS, or text changed')
    return identity


def analyze(report):
    rounds = report['rounds']
    rng = random.Random(20261112)
    samples = [[rng.randrange(rounds) for _ in range(rounds)] for _ in range(10000)]
    visits = {(v['round'], v['config']): v for v in report['visits']}
    require(len(visits) == 2*rounds, 'Missing or duplicate paired visits')
    result = []
    for profile in report['workloads']:
        pairs = []
        for index in range(rounds):
            baseline_visit, candidate_visit = visits[index, 'cache-off'], visits[index, 'cache-on']
            baseline = baseline_visit['profiles'][profile]['scored']
            candidate = candidate_visit['profiles'][profile]['scored']
            require(baseline_visit['order_index'] == index % 2
                    and candidate_visit['order_index'] == 1-index % 2, 'Configuration order did not alternate')
            pairs.append(dict(round=index, cache_off_before_cache_on=baseline_visit['order_index'] == 0,
                baseline=baseline, candidate=candidate, ratios=dict(
                    ttft_speedup=baseline['first_token_seconds']/candidate['first_token_seconds'],
                    decode_speedup=candidate['steady_decode_tokens_per_second']/baseline['steady_decode_tokens_per_second'],
                    request_speedup=baseline['request_seconds']/candidate['request_seconds'])))
        result.append(dict(profile=profile, raw_pairs=pairs,
            medians={name: {metric: statistics.median(pair[side][metric] for pair in pairs)
                for metric in ('first_token_seconds', 'steady_decode_tokens_per_second', 'request_seconds')}
                for name, side in (('cache-off', 'baseline'), ('cache-on', 'candidate'))},
            metrics={metric: interval([pair['ratios'][metric] for pair in pairs], samples)
                     for metric in ('ttft_speedup', 'decode_speedup', 'request_speedup')}))
    return dict(baseline='cache-off', candidate='cache-on', comparisons=result,
        bootstrap=dict(method='Paired-round percentile bootstrap of median ratios', confidence_level=.95,
                       resamples=10000, seed=20261112, common_round_resamples_across_contexts_and_metrics=True),
        limitation='Intervals are unadjusted; exchangeable-round assumption does not remove OS load or thermal drift.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=ROOT/'build/native-cpu/weights.bin')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--correctness-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--prefill-threads', type=int, default=6)
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT/'build/benchmark-status')
    args = parser.parse_args()
    if not 10 <= args.rounds <= 100 or not 1 <= args.warmups <= 3 or not 2 <= args.tokens <= 128:
        parser.error('Use10..100 preselected paired rounds,1..3 warmups,2..128 output tokens')
    if not 1 <= args.threads <= 64 or not 1 <= args.prefill_threads <= 64:
        parser.error('Use1..64 workers')
    binary, weights, output = args.binary.resolve(), args.weights.resolve(), args.output.resolve()
    logs = output.parent/(output.stem+'-logs')
    if output.exists() or logs.exists():
        parser.error('Preserve earlier evidence; choose a fresh output/log path')
    guarded = guard(args.guard_state_dir)
    logs.mkdir(parents=True)
    checkpoint = locate_model(args.checkpoint)
    protocol = ChatProtocol(checkpoint)
    report = dict(completed=False, paired_sampling_complete=False, performance_claim_eligible=False,
        mode='confirmation', precision='f16', gpu=False, pytorch=False, rounds=args.rounds,
        tokens=args.tokens, warmups_per_profile=args.warmups, preflight=[], visits=[],
        comparison_scope='Same native binary, weights, threads and adapter; prefix reuse on/off. Not a llama comparison.',
        timing_scope='Adapter stream wall time including rewind/reset, mode, IPC and text decoding; startup and priming excluded.',
        configurations=[dict(name='cache-off', reuse_prefix=False), dict(name='cache-on', reuse_prefix=True)],
        decode_threads=args.threads, prefill_threads=args.prefill_threads,
        guard_before=guarded, host_before=host_snapshot(), limitations=[
            'Measures a prepared same-session tail edit, not a first request or arbitrary conversation.',
            'Both configurations run identical full priming prompts outside the scored clock; priming cost is not amortized into TTFT.',
            'EOS is honored. At least two visible tokens are required; exact IDs and stop behavior must match every visit.',
            'One process is live per visit. OS scheduling, memory address placement and background work remain uncontrolled.',
            'Intervals require manual review; this report does not establish a full-logit or answer-quality gate for these long prompts.'])
    def save():
        pending = output.with_suffix('.pending.json')
        pending.write_text(json.dumps(report, indent=2)+'\n'); pending.replace(output)
    try:
        inputs = [binary, binary.parent/'manifest.json', weights, Path(str(weights)+'.index'),
                  Path(str(weights)+'.manifest.json'), args.correctness_report.resolve()]
        inputs += [checkpoint/name for name in ('tokenizer.json', 'tokenizer_config.json', 'config.json')]
        sources = [Path(__file__).resolve()] + [ROOT/'tools'/name for name in (
            'native_cpu_chat.py', 'chat_protocol.py', 'benchmark_guard.py', 'benchmark_native_phase0.py',
            'benchmark_matched.py', 'analyze_native_phase0.py', 'experiment.py')]
        report['inputs'] = {str(path): sha(path) for path in inputs}
        report['source_hashes'] = {str(path): sha(path) for path in sources}
        build = json.loads((binary.parent/'manifest.json').read_text())
        pack = json.loads(Path(str(weights)+'.manifest.json').read_text())
        require(build.get('completed') is True and build.get('binary_sha256') == sha(binary), 'Build provenance mismatch')
        require(pack.get('completed') is True and pack.get('format') == 'QWCPU01'
                and pack['binary']['sha256'] == sha(weights)
                and pack['index']['sha256'] == sha(Path(str(weights)+'.index')), 'Weight provenance mismatch')
        report['native_build'] = build
        quality = json.loads(args.correctness_report.read_text())
        require(quality.get('completed') is True and quality.get('passed') is True,
                'Prefix correctness report has not passed')
        for path in (binary, weights, Path(str(weights)+'.index'), ROOT/'tools/native_cpu_chat.py'):
            require(quality.get('inputs', {}).get(str(path)) == sha(path), 'Correctness scope mismatch: '+str(path))
        report['correctness_report'] = dict(path=str(args.correctness_report.resolve()),
            sha256=sha(args.correctness_report), scope=quality.get('scope'),
            limitation='Matches executable, weights and adapter hashes; long timing fixtures still require their own output-identity preflight.')
        profiles = workloads(protocol, args.tokens)
        report['workloads'] = {name: row['evidence'] for name, row in profiles.items()}
        expected = {}
        save()
        def visit(config, phase, index=None, order=None):
            guard(args.guard_state_dir)
            directory = logs/(phase if index is None else f'round-{index:03d}')/config['name']
            directory.mkdir(parents=True)
            client = ObservedChat(protocol, binary=binary, weights=weights, threads=args.threads,
                                  prefill_threads=args.prefill_threads, reuse_prefix=config['reuse_prefix'])
            record = dict(config=config['name'], phase=phase, round=index, order_index=order, profiles={})
            report['preflight' if phase == 'preflight' else 'visits'].append(record)
            log_path = binary.parent/'chat.log'
            offset = log_path.stat().st_size if log_path.exists() else 0
            try:
                start = time.perf_counter(); client.start(threading.Event())
                record.update(startup_seconds=time.perf_counter()-start, pid=client.process.pid, ready=client.ready)
                require(client.ready.get('prefix_reuse') is True and client.ready.get('prefix_reuse_chunk') == 512,
                        'Binary does not support canonical prefix reuse')
                for name, profile in profiles.items():
                    row = dict(warmups=[]); record['profiles'][name] = row
                    for _ in range(args.warmups):
                        reset = client.command('reset', threading.Event())
                        require(reset.get('ok') is True and reset.get('position') == 0, 'Priming reset failed')
                        warm = client.generate(profile['prime'])
                        identity = validate(warm, expected.get((name, 'prime')), len(profile['prime'].ids), 0)
                        expected.setdefault((name, 'prime'), identity); row['warmups'].append(warm)
                    scored = client.generate(profile['target'])
                    reuse = profile['evidence']['expected_reused_tokens'] if config['reuse_prefix'] else 0
                    identity = validate(scored, expected.get((name, 'target')), len(profile['target'].ids), reuse)
                    expected.setdefault((name, 'target'), identity); row['scored'] = scored
                    save()
            finally:
                client.close()
                if log_path.exists():
                    with log_path.open('rb') as stream:
                        stream.seek(offset); (directory/'native.log').write_bytes(stream.read())
                    record['runtime_log'] = dict(path=str(directory/'native.log'), sha256=sha(directory/'native.log'))
                save()
        for config in report['configurations']:
            visit(config, 'preflight')
        report['preflight_exact_ids_and_stops_passed'] = True
        report['expected_outputs'] = {name: {kind: expected[name, kind] for kind in ('prime', 'target')} for name in profiles}
        for index in range(args.rounds):
            order = report['configurations'] if index % 2 == 0 else list(reversed(report['configurations']))
            for position, config in enumerate(order):
                visit(config, 'scored', index, position)
        report['analysis'] = analyze(report)
        require(report['inputs'] == {str(path): sha(path) for path in inputs}
                and report['source_hashes'] == {str(path): sha(path) for path in sources}, 'Inputs or dependencies changed')
        report.update(completed=True, paired_sampling_complete=True, guard_after=guard(args.guard_state_dir),
                      host_after=host_snapshot())
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error)); raise
    finally:
        save()
    print(output)


if __name__ == '__main__':
    main()
