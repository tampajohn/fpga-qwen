#!/usr/bin/env python3
"""Guarded real-adapter greedy suffix equivalence; no performance claims."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import threading

from benchmark_native_phase0 import guard, sha
from chat_protocol import ChatProtocol
from check_native_conversation_extended import CASES as EXTENDED
from conversation_eval import CASES
from experiment import locate_model
from native_cpu_chat import NativeCPUChat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--reference-binary', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--guard-state-dir', type=Path, default=Path('build/benchmark-status'))
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Output already exists')
    report = dict(completed=False, passed=False, gpu=False, performance_claim_eligible=False,
                  guard_before=guard(args.guard_state_dir), cases=[],
                  scope='Exact visible IDs, text and natural stop/cap through the real adapter; fixed nine and eighteen case suites plus repetitive/budget fixtures. Not an independent model-quality claim.')
    frozen = [args.binary.resolve(), args.reference_binary.resolve(), args.weights.resolve(),
              Path(__file__).resolve(), Path(__file__).with_name('native_cpu_chat.py'),
              Path(__file__).with_name('suffix.py')]
    report['hashes'] = {str(p): sha(p) for p in frozen}
    protocol = ChatProtocol(locate_model(None))
    fixtures = [(c.name, list(c.messages), c.max_new_tokens) for c in (*CASES, *EXTENDED)]
    for name, prompt in [('repeat', 'Write the exact sequence red blue green twenty times, without commentary.'),
                         ('count', 'Count from 1 to 40, separated by commas, without commentary.')]:
        for cap in [1, 2, 3, 64]:
            fixtures.append((f'{name}-budget-{cap}', [dict(role='user', content=prompt)], cap))
    engines = [NativeCPUChat(protocol, binary=b, weights=args.weights, reuse_prefix=False)
               for b in [args.reference_binary, args.binary]]
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.pending.json')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.output)
    try:
        for name, messages, cap in fixtures:
            guard(args.guard_state_dir)
            request = protocol.prepare(dict(session=name, messages=messages, decoding='greedy',
                                            max_new_tokens=cap, max_draft=0))
            row = dict(name=name, prompt_tokens=len(request.ids), cap=cap, results=[])
            report['cases'].append(row)
            for label, engine, draft in [('reference', engines[0], 0),
                                         ('candidate_dense', engines[1], 0),
                                         ('candidate_suffix', engines[1], 1)]:
                events = list(engine.stream(replace(request, max_draft=draft), threading.Event()))
                done = events[-1]
                if done.get('type') != 'done' or done.get('finish_reason') not in ('eos', 'length'):
                    raise RuntimeError(f'Incomplete response: {name}/{label}')
                result = {key: done[key] for key in ('token_ids', 'text', 'finish_reason')}
                row['results'].append(dict(configuration=label, result=result, metrics=done['metrics']))
                if label == 'candidate_suffix' and engine.status().get('suffix_supported') is not True:
                    raise RuntimeError('Suffix capability not active')
            row['passed'] = all(r['result'] == row['results'][0]['result'] for r in row['results'])
            save()
            if not row['passed']:
                raise RuntimeError('Trajectory mismatch: '+name)
        if not all(sha(Path(p)) == digest for p, digest in report['hashes'].items()):
            raise RuntimeError('Inputs changed during run')
        suffix_metrics = [r['results'][2]['metrics'] for r in report['cases']]
        report['drafted_tokens'] = sum(m.get('proposed_tokens', 0) for m in suffix_metrics)
        report['accepted_tokens'] = sum(m.get('accepted_tokens', 0) for m in suffix_metrics)
        if not report['drafted_tokens'] or not report['accepted_tokens']:
            raise RuntimeError('Fixtures did not exercise both drafting and acceptance')
        report.update(completed=True, passed=True, guard_after=guard(args.guard_state_dir))
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        for engine in engines:
            engine.close()
        save()
    print(args.output)


if __name__ == '__main__':
    main()
