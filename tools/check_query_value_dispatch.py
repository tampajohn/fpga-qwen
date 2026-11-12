"""Same-executable OFF/ON/threshold exact gates; no timing or broad quality claim."""
import argparse
import json
from pathlib import Path
import tempfile

from query_value_dispatch_common import (PATTERN, artifact, exact, fill, make_models,
                                        policy, repeat_tokens, route_call, sha, start)
from check_phase_pools_real import telemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--threshold', type=int, default=256)
    parser.add_argument('--capacity', action='store_true', help='Also exercise actual 4096-token boundary')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 9 <= args.threshold <= 4080:
        parser.error('Gate threshold must be between 9 and 4080 to test both sides and tails')
    frozen = artifact(args.binary, args.sha256)
    selectors = ['OFF', 'ON', f'MIN_CONTEXT:{args.threshold}']
    models = make_models(args.binary)
    report = dict(completed=False, timing_claim=False, artifact=frozen, selectors=selectors,
                  source_sha256=sha(__file__), cases=[], route_checks=[], actual_capacity_checked=False,
                  scope='Raw KV truncate/replay checks; canonical browser cache and suffix lifecycle suites remain separate requirements.')
    try:
        report['startup'] = [start(m, s) for m, s in zip(models, selectors)]
        with tempfile.TemporaryDirectory() as directory:
            def equal(label, outputs):
                report['cases'].append(exact(models, outputs, directory, label))
            def routed(index, prefix, count, call, scalar=False):
                output, observation = route_call(models[index], selectors[index], prefix, count, call, scalar)
                report['route_checks'].append(observation)
                return output
            def continuation(prefix):
                return [routed(i, prefix, 1, lambda m=m: m.token(198), scalar=True)[:2]
                        for i, m in enumerate(models)]
            prefixes = sorted({0, 63, 451, args.threshold - 8, args.threshold - 3,
                               args.threshold - 1, args.threshold, args.threshold + 1})
            for prefix in prefixes:
                for model in models:
                    model.reset()
                    fill(model, repeat_tokens(prefix))
                # Truncation retains the identical prefix and overwrites a former
                # longer continuation. Near-threshold blocks cross within a pair group.
                for count in range(1, 9):
                    for i, model in enumerate(models):
                        assert model.command(f'truncate {prefix}') == 'truncated'
                        routed(i, prefix, count, lambda m=model: m.prefill(PATTERN[:count]))
                    equal(f'prefix{prefix}-prefill{count}', continuation(prefix + count))
                for count in range(1, 5):
                    for logits in (False, True):
                        for model in models:
                            assert model.command(f'truncate {prefix}') == 'truncated'
                        outputs = [[(x[0], x[1]) for x in routed(i, prefix, count,
                                   lambda m=m: m.block(PATTERN[:count], logits=logits,
                                   uniforms=[.031, .72, .999, .43][:count] if logits else None))]
                                   for i, m in enumerate(models)]
                        if logits:
                            equal(f'prefix{prefix}-sampled-verification{count}', outputs)
                        elif any(value != outputs[0] for value in outputs[1:]):
                            raise AssertionError('No-logits verification IDs/positions differ')
                        equal(f'prefix{prefix}-verification{count}-logits{logits}-continuation',
                              continuation(prefix + count))
                print(f'PASS threshold neighborhood prefix {prefix}', flush=True)
            if args.capacity:
                for model in models:
                    model.reset()
                    fill(model, repeat_tokens(4088))
                for i, model in enumerate(models):
                    result = routed(i, 4088, 8, lambda m=model: m.prefill(PATTERN))
                    assert result[-1][1] == 4096
                report['full_b8_prefill_reached'] = 4096
                for count in range(1, 8):
                    for model in models:
                        assert model.command('truncate 4088') == 'truncated'
                        model.prefill(PATTERN[:count])
                    equal(f'capacity-prefill-tail{count}', [m.token(198)[:2] for m in models])
                for model in models:
                    assert model.command('truncate 4088') == 'truncated'
                    model.prefill(PATTERN[:4])
                equal('capacity-verification4093-to4096',
                      [[(x[0], x[1]) for x in m.block(PATTERN[:4], logits=True)] for m in models])
                report['before_overflow_policy'] = [policy(m, s) for m, s in zip(models, selectors)]
                for model in models:
                    assert model.command('truncate 4092') == 'truncated'
                    try:
                        model.command('prefill 5 9707 9707 9707 9707 9707')
                    except RuntimeError:
                        assert model.process.wait(timeout=2) == 1
                    else:
                        raise AssertionError('Overflow accepted')
                report['overflow_rejected'] = True
                report['actual_capacity_checked'] = True
            if not args.capacity:
                report['before_restart_policy'] = [policy(m, s) for m, s in zip(models, selectors)]
                report['before_restart_phase'] = [telemetry(m, True) for m in models]
            # Restart confirms the experiment's selector survives explicit host
            # reconfiguration, rather than inheriting another model's environment.
            for model in models:
                model.close()
            report['restart'] = [start(m, s) for m, s in zip(models, selectors)]
            for model in models:
                model.reset()
                fill(model, repeat_tokens(args.threshold + 8))
            equal('restart-and-threshold-crossing', [m.token(198)[:2] for m in models])
        report['final_policy'] = [policy(m, s) for m, s in zip(models, selectors)]
        report['phase_telemetry'] = [telemetry(m, True) for m in models]
        if sha(args.binary) != args.sha256:
            raise RuntimeError('Executable changed during gates')
        report['completed'] = True
    finally:
        for model in models:
            model.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS', len(report['cases']), 'three-policy full-vocabulary comparisons')


if __name__ == '__main__':
    main()
