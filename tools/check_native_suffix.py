#!/usr/bin/env python3
"""Guarded real-adapter greedy suffix equivalence and lifecycle; no timing claims."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import threading

from benchmark_native_phase0 import guard, sha
from chat_protocol import ChatProtocol
from check_native_conversation_extended import CASES as EXTENDED
from conversation_eval import CASES
from experiment import ROOT, locate_model
from native_cpu_chat import NativeCPUChat, EOS


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class ObservedNative(NativeCPUChat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.begin_request()

    def begin_request(self, cancel_verify=False):
        self.predictions, self.transactions, self.decoded_inputs = [], [], []
        self.cancel_next_verify = cancel_verify
        self.cancel_after_write = False
        self.cancelled_process = None
        self.cancelled_dispatch = None

    def _read(self, cancel, timeout=180):
        if self.cancel_after_write:
            # NativeCPUChat.command has already written and flushed the actual
            # verify command. Interrupt before receiving its result. This proves
            # transport dispatch, not which arithmetic instruction was interrupted.
            self.cancel_after_write = False
            self.cancelled_process = self.process
            require(self.process is not None and self.process.poll() is None,
                    'Verification process was not alive at cancellation dispatch')
            cancel.set()
        return super()._read(cancel, timeout)

    def command(self, command, cancel, **kwargs):
        operation = command.split(' ', 1)[0]
        if operation == 'verify-greedy' and self.cancel_next_verify:
            self.cancel_next_verify = False
            pending, draft = map(int, command.split()[1:])
            self.cancelled_dispatch = dict(command=command, pending=pending, draft=draft,
                                           base_position=len(self.cache_ids))
            self.cancel_after_write = True
        result = super().command(command, cancel, **kwargs)
        if operation in ('prefill', 'step'):
            self.predictions.append(result['token'])
            if operation == 'step':
                self.decoded_inputs.append(int(command.split()[1]))
        elif operation == 'verify-greedy':
            pending, draft = map(int, command.split()[1:])
            self.predictions.extend(result['tokens'])
            self.decoded_inputs.extend([pending, draft][:result['committed_inputs']])
            self.transactions.append(dict(pending=pending, draft=draft, response={key: result[key]
                for key in ('tokens', 'accepted', 'committed_inputs', 'position', 'verified_positions', 'finish_reason')}))
        return result


def run_request(engine, request, cancel_verify=False):
    engine.begin_request(cancel_verify)
    events = list(engine.stream(request, threading.Event()))
    require(events and events[-1].get('type') == 'done', 'Missing done event')
    done = events[-1]
    reason, ids = done.get('finish_reason'), done.get('token_ids')
    require(isinstance(ids, list) and len(ids) <= request.max_new
            and all(type(token) is int and 0 <= token < 151936 and token not in EOS for token in ids),
            'Invalid visible IDs or reply cap')
    require(reason in (('stopped',) if cancel_verify else ('eos', 'length')), 'Unexpected natural stop/cancel result')
    if reason == 'eos':
        require(engine.predictions and engine.predictions[-1] in EOS
                and engine.predictions[:-1] == ids, 'EOS-inclusive predictions differ from visible result')
    else:
        require(engine.predictions == ids, 'Predictions differ from visible result')
        if reason == 'length':
            require(len(ids) == request.max_new, 'Length stop did not reach requested cap')
    metrics = done['metrics']
    require(metrics['output_tokens'] == len(ids), 'Output metrics differ from visible IDs')
    completed = len(engine.transactions)
    accepted = sum(row['response']['accepted'] for row in engine.transactions)
    outstanding = int(cancel_verify)
    require(metrics.get('verification_calls') == completed+outstanding
            and metrics.get('proposed_tokens') == completed+outstanding
            and metrics.get('accepted_tokens') == accepted
            and metrics.get('rejected_tokens') == completed-accepted
            and metrics.get('verified_positions') == 2*completed, 'Verification metrics differ from observed commands')
    if not cancel_verify:
        require(engine.cache_ids == request.ids+engine.decoded_inputs, 'Cache contains unconsumed or missing input IDs')
        stats = engine.command('stats', threading.Event())
        require(stats.get('position') == len(engine.cache_ids) == metrics['kv_tokens'], 'Actual KV position differs from cache/metrics')
        ceiling = len(request.ids)//512*512 if engine.prefix_reuse_supported() else 0
        require(engine.canonical_reusable_tokens == ceiling, 'Verification extended the canonical prefill reuse ceiling')
    return dict(result={key: done[key] for key in ('token_ids', 'text', 'finish_reason')},
                predicted_ids_including_eos=list(engine.predictions), metrics=metrics,
                decoded_inputs=list(engine.decoded_inputs), transactions=list(engine.transactions))


def same_output(rows):
    reference = rows[0]
    return all(row['result'] == reference['result']
               and row['predicted_ids_including_eos'] == reference['predicted_ids_including_eos'] for row in rows)


def expected_reuse(previous, request):
    if previous is None or previous.session != request.session:
        return 0
    common = 0
    for left, right in zip(previous.ids, request.ids):
        if left != right:
            break
        common += 1
    return min(common, len(previous.ids)//512*512, len(request.ids)-1)//512*512


def cached_fixtures(protocol):
    padding = 0
    instruction = '\nIgnore the background notes. Count from 1 to 40, separated by commas, without commentary.'
    for _ in range(10):
        content = 'Background notes:'+' x'*padding+instruction
        messages = [dict(role='user', content=content)]
        request = protocol.prepare(dict(session='suffix-cache-a', messages=messages,
            decoding='greedy', max_new_tokens=64, max_draft=1))
        if len(request.ids) == 1025:
            break
        padding += 1025-len(request.ids)
        require(padding >= 0, 'Cache fixture template exceeds target')
    else:
        raise RuntimeError('Could not tokenize exact1025-token cache fixture')
    edited = content.replace('Count from 1 to 40', 'Count from 2 to 41')
    early = edited.replace('Background notes:', 'Revised notes:', 1)
    definitions = [('initial', content, 'a'), ('identical', content, 'a'),
                   ('tail-edit', edited, 'a'), ('identical-tail-edit', edited, 'a'),
                   ('early-edit', early, 'a'), ('session-switch', early, 'b'),
                   ('identical-new-session', early, 'b')]
    result = []
    for name, text, session in definitions:
        messages = [dict(role='user', content=text)]
        req = protocol.prepare(dict(session='suffix-cache-'+session, messages=messages,
            decoding='greedy', max_new_tokens=64, max_draft=1))
        result.append((name, messages, req))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--reference-binary', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT/'build/benchmark-status')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Output already exists')
    report = dict(completed=False, passed=False, gpu=False, pytorch=False, performance_claim_eligible=False,
        guard_before=guard(args.guard_state_dir), cases=[], extended_cases=[],
        scope='Original35 fixed-suite/budget cases remain unchanged. Additional explicit cached-prefix, edited/session followups and verify-dispatch cancellation/restart cases compare real-adapter exact IDs, EOS and text. No independent answer-quality or performance claim.')
    engines = []
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.pending.json')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.output)
    try:
        save()
        binary, reference, weights = args.binary.resolve(), args.reference_binary.resolve(), args.weights.resolve()
        checkpoint = locate_model(None)
        frozen = [binary, reference, binary.parent/'manifest.json', reference.parent/'manifest.json',
                  weights, Path(str(weights)+'.index'), Path(str(weights)+'.manifest.json')]
        frozen += [checkpoint/name for name in ('tokenizer.json', 'tokenizer_config.json', 'config.json')]
        frozen += [ROOT/'tools'/name for name in ('check_native_suffix.py', 'native_cpu_chat.py', 'suffix.py',
            'chat_protocol.py', 'conversation_eval.py', 'check_native_conversation_extended.py',
            'benchmark_native_phase0.py', 'benchmark_matched.py', 'benchmark_guard.py', 'experiment.py')]
        report['hashes'] = {str(path): sha(path) for path in frozen}
        pack = json.loads(Path(str(weights)+'.manifest.json').read_text())
        require(pack.get('completed') is True and pack.get('format') == 'QWCPU01'
                and pack['binary']['sha256'] == report['hashes'][str(weights)]
                and pack['index']['sha256'] == report['hashes'][str(weights)+'.index'], 'Pack manifest/hash mismatch')
        report['builds'] = {}
        for executable in (binary, reference):
            manifest = json.loads((executable.parent/'manifest.json').read_text())
            require(manifest.get('completed') is True and manifest.get('gpu') is False
                    and manifest.get('pytorch') is False
                    and manifest.get('binary_sha256') == report['hashes'][str(executable)], 'Build manifest/hash mismatch')
            report['builds'][str(executable)] = manifest
        protocol = ChatProtocol(checkpoint)
        fixtures = [(c.name, list(c.messages), c.max_new_tokens) for c in (*CASES, *EXTENDED)]
        for name, prompt in [('repeat', 'Write the exact sequence red blue green twenty times, without commentary.'),
                             ('count', 'Count from 1 to 40, separated by commas, without commentary.')]:
            for cap in [1, 2, 3, 64]:
                fixtures.append((f'{name}-budget-{cap}', [dict(role='user', content=prompt)], cap))
        require(len(fixtures) == 35, 'Original fixed35-case gate changed')
        engines = [ObservedNative(protocol, binary=b, weights=weights, reuse_prefix=False) for b in (reference, binary)]
        for name, messages, cap in fixtures:
            guard(args.guard_state_dir)
            request = protocol.prepare(dict(session=name, messages=messages, decoding='greedy',
                                            max_new_tokens=cap, max_draft=0))
            row = dict(name=name, messages=messages, prompt_ids=request.ids,
                       prompt_tokens=len(request.ids), cap=cap, results=[])
            report['cases'].append(row)
            for label, engine, draft in [('reference', engines[0], 0),
                                         ('candidate_dense', engines[1], 0),
                                         ('candidate_suffix', engines[1], 1)]:
                result = run_request(engine, replace(request, max_draft=draft))
                row['results'].append(dict(configuration=label, **result))
                require(result['metrics']['reused_prefix_tokens'] == 0, 'Fresh fixed-suite case reused prefix')
                if label == 'candidate_suffix':
                    require(engine.status().get('suffix_supported') is True, 'Suffix capability not active')
            row['passed'] = same_output(row['results']); save()
            require(row['passed'], 'Trajectory mismatch: '+name)
        suffix_metrics = [row['results'][2]['metrics'] for row in report['cases']]
        report['drafted_tokens'] = sum(m['proposed_tokens'] for m in suffix_metrics)
        report['accepted_tokens'] = sum(m['accepted_tokens'] for m in suffix_metrics)
        report['rejected_tokens'] = sum(m['rejected_tokens'] for m in suffix_metrics)
        require(report['drafted_tokens'] and report['accepted_tokens'] and report['rejected_tokens'],
                'Original fixtures did not exercise drafting, acceptance and rejection')
        report['original35_passed'] = True; save()
        cached = ObservedNative(protocol, binary=binary, weights=weights, reuse_prefix=True)
        engines.append(cached)
        previous = None
        cached_verified = 0
        cache_fixtures = cached_fixtures(protocol)
        for name, messages, request in cache_fixtures:
            guard(args.guard_state_dir)
            expected = expected_reuse(previous, request)
            row = dict(name=name, messages=messages, prompt_ids=request.ids,
                       expected_reused_tokens=expected, results=[])
            report['extended_cases'].append(row)
            for label, engine, draft in [('reference_fresh', engines[0], 0),
                                         ('candidate_suffix_fresh', engines[1], 1),
                                         ('candidate_suffix_cached', cached, 1)]:
                result = run_request(engine, replace(request, max_draft=draft))
                require(result['metrics']['reused_prefix_tokens'] == (expected if engine is cached else 0),
                        'Unexpected prefix reuse: '+name+'/'+label)
                row['results'].append(dict(configuration=label, **result))
            row['passed'] = same_output(row['results']); save()
            require(row['passed'], 'Cached suffix trajectory mismatch: '+name)
            if expected:
                cached_verified += row['results'][2]['metrics']['verification_calls']
            previous = request
        require(cached_verified > 0, 'Cached fixtures never actually verified a suffix under prefix reuse')
        report['cached_prefix_verification_calls'] = cached_verified
        # Repeat the last fixed cached fixture; force cancellation only after an
        # actual verify command was flushed to a live process, before its reply.
        request = cache_fixtures[-1][2]
        expected = expected_reuse(previous, request)
        require(expected >= 512, 'Cancellation fixture lacks a reusable canonical prefix')
        stopped = run_request(cached, request, cancel_verify=True)
        process, dispatch = cached.cancelled_process, cached.cancelled_dispatch
        require(process is not None and process.poll() is not None and dispatch is not None
                and cached.process is None and cached.ready is None and not cached.cache_ids
                and cached.canonical_reusable_tokens == 0 and cached.cache_session is None,
                'Verify cancellation was not observed/reaped or left reusable state')
        require(stopped['metrics']['reused_prefix_tokens'] == expected
                and stopped['metrics']['verification_calls'] >= 1 and stopped['result']['token_ids'],
                'Cancellation did not exercise cached verification after visible output')
        report['cancellation'] = dict(passed=True, dispatch=dispatch, returncode=process.returncode,
            result=stopped, scope='Cancel after verify bytes are flushed, before receiving its response; internal arithmetic progress is not inferred.')
        baseline = run_request(engines[0], replace(request, max_draft=0))
        restarted = run_request(cached, request)
        require(same_output([baseline, restarted]) and restarted['metrics']['reused_prefix_tokens'] == 0,
                'Post-cancel restart differed or reused stale cache')
        prefix = stopped['result']['token_ids']
        require(prefix == baseline['result']['token_ids'][:len(prefix)], 'Cancelled visible output is not an exact baseline prefix')
        repeated = run_request(cached, request)
        require(same_output([baseline, repeated])
                and repeated['metrics']['reused_prefix_tokens'] == expected_reuse(request, request),
                'Post-restart repeated prefix/suffix response differed')
        report['restart'] = dict(passed=True, baseline=baseline, restarted=restarted, repeated=repeated)
        require(all(sha(Path(path)) == digest for path, digest in report['hashes'].items()), 'Inputs/helpers changed during run')
        report.update(completed=True, passed=True, guard_after=guard(args.guard_state_dir))
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error)); raise
    finally:
        cleanup_errors = []
        for engine in engines:
            try:
                engine.close()
            except BaseException as error:
                cleanup_errors.append(dict(type=type(error).__name__, message=str(error)))
        if cleanup_errors:
            report.update(completed=False, passed=False, cleanup_errors=cleanup_errors)
        save()
    require(report['passed'], 'Suffix gate cleanup failed')
    print(args.output)


if __name__ == '__main__':
    main()
