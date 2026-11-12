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


def guard(directory):
    status = BenchmarkGate(directory).status()
    metadata = json.loads((directory / 'run.json').read_text())
    if not status['running'] or metadata.get('pid') != os.getppid():
        raise RuntimeError('Run this helper as a direct child of the serving checkout benchmark_guard.py')
    return metadata


def padded_prompt(protocol, count):
    base = make_prompt(protocol, 'short')
    instruction = 'Explain memory bandwidth and caching in four concise sentences.'
    instruction_ids = protocol.tokenizer.encode(instruction, add_special_tokens=False)
    matches = [i for i in range(len(base)) if base[i:i + len(instruction_ids)] == instruction_ids]
    if len(matches) != 1 or count < len(base):
        raise ValueError(f'Need an unambiguous instruction and context at least {len(base)}')
    filler = protocol.tokenizer.encode('Memory bandwidth measures data transfer. Caches retain recently accessed information. ',
                                       add_special_tokens=False)
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
                query_value_route_delta=route_actual, query_value_route_expected=route_expected)


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
    models = make_models(args.binary)
    selectors = ['OFF', 'ON', f'MIN_CONTEXT:{args.threshold}']
    protocol = ChatProtocol(models[0].model)
    report = dict(completed=False, artifact=frozen, selectors=selectors, guard=guard_metadata,
                  source_sha256=sha(__file__), tokens=args.tokens, profiles=[],
                  gpu=False, pytorch=False, prefill_workers=12, decode_workers=8,
                  limitations='Three resident processes from one executable; sequential requests, warm OS caches, no placement/eviction guarantees. Browser guard excludes chat; external OS processes and background services are not controlled. Token-level user-message padding gives exact contexts. No suffix or canonical reuse during timing. Numerical parity only, not broad quality evidence.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report['startup'] = [start(m, s) for m, s in zip(models, selectors)]
        with tempfile.TemporaryDirectory() as directory:
            def bits(index):
                return read_dump_logits(models[index], Path(directory) / f'{index}.bin').tobytes()
            for context in contexts:
                prompt = padded_prompt(protocol, context)
                assert len(prompt) == context
                for batch in batches:
                    warm = [request(m, s, prompt, args.tokens, batch) for m, s in zip(models, selectors)]
                    expected = warm[0]['output_token_ids']
                    warm_bits = [bits(i) for i in range(3)]
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
                            results[index] = request(models[index], selectors[index], prompt, args.tokens, batch)
                            if results[index]['output_token_ids'] != expected or bits(index) != warm_bits[index]:
                                raise AssertionError('Scored full logits/IDs differ from exact warm baseline')
                        ratios = {selectors[i]: {k: results[0][k] / results[i][k] for k in METRICS} for i in (1, 2)}
                        record['pairs'].append(dict(index=pair, order=order, results={selectors[i]: results[i] for i in range(3)},
                                                    speedup_over_off=ratios, all_full_logits_exact=True))
                        args.output.write_text(json.dumps(report, indent=2) + '\n')
                        print(json.dumps(dict(context=context, batch=batch, pair=pair, ratios=ratios)), flush=True)
                    record['median_speedup_over_off'] = {s: {k: statistics.median(p['speedup_over_off'][s][k] for p in record['pairs'])
                                                             for k in METRICS} for s in selectors[1:]}
        report['final_policy'] = [policy(m, s) for m, s in zip(models, selectors)]
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
