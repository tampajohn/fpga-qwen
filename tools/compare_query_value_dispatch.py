"""Guarded same-executable OFF/ON/threshold context and prefill-batch sweep.

Invoke directly under the SERVING checkout's benchmark_guard.py, then pass its
build/benchmark-status directory with --guard-state-dir. This helper refuses an
unwrapped run. All three modes use the same SHA-pinned executable and arithmetic.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
import time

from benchmark_guard import BenchmarkGate
from benchmark_matched import make_prompt
from chat_protocol import ChatProtocol
from check_phase_pools_real import telemetry
from query_value_dispatch_common import artifact, make_models, policy, read_dump_logits, sha, start


METRICS = ('prefill_seconds', 'ttft_seconds', 'decode_seconds', 'total_seconds')


class InProcessControl:
    """Switch scheduling metadata while proving process/pool identity survives."""
    def __init__(self, model, selector='OFF'):
        self.model = model
        self.selector = selector
        self.pid = model.process.pid
        self.resident = dict(pool_constructions=2, resident_decode_workers=8,
                             resident_prefill_workers=12, prefill_workers=12, decode_workers=8)
        self.check_resident()
        actual = policy(model, selector)
        if type(actual.get('epoch')) is not int or actual['epoch'] != 0:
            raise RuntimeError('In-process comparison requires a fresh policy-setter executable with epoch0')

    def check_resident(self):
        if self.model.process is None or self.model.process.poll() is not None or self.model.process.pid != self.pid:
            raise RuntimeError('In-process comparison lost its original simulator process')
        phase = telemetry(self.model, True)
        if any(type(phase.get(k)) is not type(v) or phase[k] != v for k, v in self.resident.items()):
            raise RuntimeError('Resident phase pools were recreated or reconfigured')
        return phase

    def select(self, selector):
        phase_before = self.check_resident()
        before = policy(self.model, self.selector)
        acknowledgement = json.loads(self.model.command('query_value_set ' + selector))
        actual = policy(self.model, selector)
        phase_after = self.check_resident()
        if acknowledgement != actual or type(actual.get('epoch')) is not int or actual['epoch'] != before['epoch'] + 1:
            raise RuntimeError('Policy setter acknowledgement or epoch differs from expected transition')
        if phase_before != phase_after or any(actual[k] != before[k] for k in ('paired_head_pairs', 'fallback_head_pairs')):
            raise RuntimeError('Policy setter changed routing counters or worker pools')
        self.selector = selector
        return dict(pid=self.pid, before=before, acknowledgement=acknowledgement,
                    phase_pools_unchanged=True, resident=self.resident)


def guard(directory):
    status = BenchmarkGate(directory).status()
    metadata = json.loads((directory / 'run.json').read_text())
    if not status['running'] or metadata.get('pid') != os.getppid():
        raise RuntimeError('Run this helper as a direct child of the serving checkout benchmark_guard.py')
    return metadata


def padded_prompt(protocol, count):
    base = make_prompt(protocol, 'short')
    instruction = 'Explain memory bandwidth and caching in four concise sentences.'
    instruction_ids = protocol.tokenizer.encode(instruction, add_special_tokens=False).ids
    matches = [i for i in range(len(base)) if base[i:i + len(instruction_ids)] == instruction_ids]
    if len(matches) != 1 or count < len(base):
        raise ValueError(f'Need an unambiguous instruction and context at least {len(base)}')
    filler = protocol.tokenizer.encode('Memory bandwidth measures data transfer. Caches retain recently accessed information. ',
                                       add_special_tokens=False).ids
    extra = count - len(base)
    padding = (filler * ((extra + len(filler) - 1) // len(filler)))[:extra]
    split = matches[0]
    # Deliberate token-level padding inside the user message, before the intact
    # instruction. The report saves every ID; this is not a quality benchmark.
    return base[:split] + padding + base[split:]


def request(model, selector, prompt, count, batch):
    model.reset()
    before = telemetry(model, True)
    route_before = policy(model, selector)
    pid_before = model.process.pid
    started = time.perf_counter()
    for offset in range(0, len(prompt) - 1, batch):
        model.prefill(prompt[offset:min(offset + batch, len(prompt) - 1)])
    prefilled = time.perf_counter()
    token, position, _ = model.token(prompt[-1])
    ids = [token]
    first = time.perf_counter()
    for _ in range(count - 1):
        token, position, _ = model.token(token)
        ids.append(token)
    ended = time.perf_counter()
    if any(token in (151643, 151645) for token in ids):
        raise RuntimeError('EOS invalidates fixed-workload comparison; do not mask or ignore it')
    if position != len(prompt) + count - 1:
        raise AssertionError('Wrong consumed-token position')
    after = telemetry(model, True)
    route_after = policy(model, selector)
    if model.process.pid != pid_before or route_after.get('epoch') != route_before.get('epoch'):
        raise AssertionError('Process or query-value policy changed during a timed request')
    minimum = route_before['min_context']
    paired = fallback = 0
    for offset in range(0, len(prompt) - 1, batch):
        size = min(batch, len(prompt) - 1 - offset)
        enabled = sum(query + 1 < size and minimum >= 0 and offset + query + 1 >= minimum
                      for query in range(0, size, 2))
        paired += enabled
        fallback += (size + 1) // 2 - enabled
    route_expected = dict(paired_head_pairs=448 * paired, fallback_head_pairs=448 * fallback)
    route_actual = {key: route_after[key] - route_before[key] for key in route_expected}
    if route_actual != route_expected:
        raise AssertionError(f'Wrong SV query-value routes: {route_actual} != {route_expected}')
    delta = [[after['dispatch_counts'][p][k] - before['dispatch_counts'][p][k]
              for k in range(3)] for p in range(2)]
    if delta[0][0] <= 0 or delta[1][1] <= 0 or any(delta[p][k] for p, k in ((0, 1), (0, 2), (1, 0), (1, 2))):
        raise AssertionError('Wrong worker-pool phase dispatch')
    return dict(prefill_seconds=prefilled - started, ttft_seconds=first - started,
                decode_seconds=ended - first, total_seconds=ended - started,
                decode_tokens_per_second=(count - 1) / (ended - first),
                output_token_ids=ids, final_position=position, phase_dispatch_delta=delta,
                query_value_route_delta=route_actual, query_value_route_expected=route_expected,
                pid=pid_before, query_value_epoch=route_before.get('epoch'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--threshold', type=int, default=256)
    parser.add_argument('--contexts', default='85,128,256,452,768')
    parser.add_argument('--batch-sizes', default='8')
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--guard-state-dir', type=Path, required=True)
    parser.add_argument('--in-process', action='store_true',
                        help='Switch policies in one persistent simulator; never recreate pools')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    contexts = [int(x) for x in args.contexts.split(',')]
    batches = [int(x) for x in args.batch_sizes.split(',')]
    if (not 1 <= args.threshold <= 4096 or not 1 <= args.pairs <= 10 or not 2 <= args.tokens <= 128
            or any(not 2 <= c <= 4096 - args.tokens + 1 for c in contexts)
            or any(not 1 <= b <= 8 for b in batches)):
        parser.error('Invalid context/batch/threshold/sample dimensions')
    guard_metadata = guard(args.guard_state_dir)
    frozen = artifact(args.binary, args.sha256)
    models = make_models(args.binary, count=1 if args.in_process else 3)
    selectors = ['OFF', 'ON', f'MIN_CONTEXT:{args.threshold}']
    protocol = ChatProtocol(models[0].model)
    report = dict(completed=False, artifact=frozen, selectors=selectors, guard=guard_metadata,
                  source_sha256=sha(__file__), tokens=args.tokens, profiles=[],
                  gpu=False, pytorch=False, prefill_workers=12, decode_workers=8,
                  in_process=args.in_process, resident_processes=len(models),
                  limitations=('One persistent process with policy switches outside clocks' if args.in_process else 'Three resident processes from one executable') + '; sequential requests, warm OS caches, no placement/eviction guarantees. Browser guard excludes chat; external OS processes and background services are not controlled. Token-level user-message padding gives exact contexts. No suffix or canonical reuse during timing. Numerical parity only, not broad quality evidence.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report['startup'] = [start(m, s) for m, s in zip(models, selectors)]
        control = InProcessControl(models[0]) if args.in_process else None
        with tempfile.TemporaryDirectory() as directory:
            def model_for(index):
                return models[0] if args.in_process else models[index]
            def bits(index):
                return read_dump_logits(model_for(index), Path(directory) / f'{index}.bin').tobytes()
            def execute(index, prompt, batch):
                transition = control.select(selectors[index]) if control else None
                result = request(model_for(index), selectors[index], prompt, args.tokens, batch)
                if control:
                    control.check_resident()
                    result['policy_transition'] = transition
                return result
            for context in contexts:
                prompt = padded_prompt(protocol, context)
                assert len(prompt) == context
                for batch in batches:
                    warm = []
                    warm_bits = []
                    for index in range(3):
                        warm.append(execute(index, prompt, batch))
                        warm_bits.append(bits(index))  # Capture before another policy overwrites this process's logits.
                    expected = warm[0]['output_token_ids']
                    if any(r['output_token_ids'] != expected for r in warm) or len(set(warm_bits)) != 1:
                        raise AssertionError('Warm policy logits/IDs differ')
                    record = dict(context_tokens=context, prefill_batch=batch, prompt_ids=prompt,
                                  warmup=warm, warm_logits_sha256=hashlib.sha256(warm_bits[0]).hexdigest(), pairs=[])
                    report['profiles'].append(record)
                    for pair in range(args.pairs):
                        order = [(pair + i) % 3 for i in range(3)]
                        if (pair // 3) % 2:
                            order.reverse()
                        results = {}
                        for index in order:
                            results[index] = execute(index, prompt, batch)
                            if results[index]['output_token_ids'] != expected or bits(index) != warm_bits[index]:
                                raise AssertionError('Scored full logits/IDs differ from exact warm baseline')
                        ratios = {selectors[i]: {k: results[0][k] / results[i][k] for k in METRICS} for i in (1, 2)}
                        record['pairs'].append(dict(index=pair, order=order, results={selectors[i]: results[i] for i in range(3)},
                                                    speedup_over_off=ratios, all_full_logits_exact=True))
                        args.output.write_text(json.dumps(report, indent=2) + '\n')
                        print(json.dumps(dict(context=context, batch=batch, pair=pair, ratios=ratios)), flush=True)
                    record['median_speedup_over_off'] = {s: {k: statistics.median(p['speedup_over_off'][s][k] for p in record['pairs'])
                                                             for k in METRICS} for s in selectors[1:]}
        report['final_policy'] = ([policy(models[0], control.selector)] if control
                                  else [policy(m, s) for m, s in zip(models, selectors)])
        if control:
            report['final_resident_phase'] = control.check_resident()
            report['same_process_pid'] = control.pid
        report['guard_after'] = guard(args.guard_state_dir)
        if sha(args.binary) != args.sha256:
            raise RuntimeError('Executable changed during scoring')
        report['completed'] = True
    finally:
        for model in models:
            model.close()
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
