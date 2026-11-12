"""Paired uncertainty analysis for completed native Phase0 confirmations only.

No inference, builds, discovery-to-confirmation conversion, automatic winner, or
quality pass. Bootstrap samples preserve each baseline/candidate measured pair;
all configurations, metrics and contexts reuse the same sampled round indices.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

PROFILES = {'short': 85, 'long': 452}
METRICS = {'decode_speedup': 'steady_decode_tokens_per_second',
           'ttft_speedup': 'first_token_seconds', 'request_speedup': 'request_seconds'}


def profiles_for(report):
    suite = report.get('context_suite', 'standard')
    require(suite in ('standard', 'extended'), 'Unknown context suite')
    return ({f'context{n}': n for n in (1024, 2048, 3968)}
            if suite == 'extended' else PROFILES)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(report, baseline):
    profiles = profiles_for(report)
    require(report.get('completed') is True and report.get('mode') == 'confirmation'
            and report.get('paired_sampling_complete') is True
            and report.get('calibration_only') is not True,
            'Require a completed confirmation; discovery and calibration reports are refused')
    rounds = report.get('rounds')
    require(type(rounds) is int and rounds >= 10, 'At least10complete paired rounds are required')
    require(report.get('gpu') is False and report.get('pytorch') is False, 'Expected CPU-only native confirmation')
    require(report.get('preflight_same_precision_ids_passed') is True, 'Missing preflight output-identity gate')
    require(report.get('precision') in ('f16', 'f32', 'q4_k_m'), 'Unknown native precision class')
    configs = report['configurations']
    names = [config['name'] for config in configs]
    require(len(names) >= 2 and len(set(names)) == len(names) and baseline in names,
            'Supply an existing explicit baseline among at least2unique configurations')
    require(set(report['prompts']) == set(profiles), 'Confirmation must cover every declared context')
    count = report['tokens']
    require(type(count) is int and count >= 2, 'Invalid fixed output count')
    for profile, length in profiles.items():
        prompt = report['prompts'][profile]
        require(prompt['count'] == length and len(prompt['ids']) == length, 'Prompt length differs from protocol')
        require(len(report['expected_output_ids'][profile]) == count, 'Wrong expected output-ID count')
    visits = {}
    orders = {}
    for visit in report['visits']:
        index, name, order = visit['round'], visit['config'], visit['order_index']
        require(visit.get('phase') == 'scored' and type(index) is int and 0 <= index < rounds
                and name in names and type(order) is int and 0 <= order < len(names),
                'Malformed scored visit')
        require((index, name) not in visits and (index, order) not in orders, 'Duplicate paired visit or order position')
        require(set(visit['profiles']) == set(profiles), 'Visit is missing a context')
        for profile, length in profiles.items():
            row = visit['profiles'][profile]['scored']
            require(row.get('output_token_count') == count and row.get('finish_reason') == 'length'
                    and row.get('eos_observed') is False
                    and row.get('output_token_ids') == report['expected_output_ids'][profile],
                    'Scored workload changed output IDs, count or EOS behavior')
            timing = row.get('backend_timings', {})
            require(timing.get('cache_n') == 0 and timing.get('prompt_n') == length,
                    'Scored request reused a prompt or evaluated a different prompt length')
            for metric in METRICS.values():
                value = row.get(metric)
                require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                        'Nonfinite or nonpositive measured metric')
        visits[index, name] = visit
        orders[index, order] = name
    require(len(visits) == rounds * len(names), 'Missing configuration/round pairs')
    # The producer rotates a configuration order, then reverses it on the next
    # round. Verify that balance rather than accepting a merely labeled report.
    for index in range(0, rounds - 1, 2):
        first = [orders[index, offset] for offset in range(len(names))]
        second = [orders[index + 1, offset] for offset in range(len(names))]
        require(first == list(reversed(second)), 'Adjacent confirmation rounds did not reverse configuration order')
    return names, visits, orders


def percentile(sorted_values, probability):
    index = probability * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (index - lower)


def interval(values, resampled_indices):
    medians = sorted(statistics.median(values[index] for index in sample) for sample in resampled_indices)
    return dict(paired_observations=len(values), median_paired_speedup=statistics.median(values),
                percentile_95_ci=[percentile(medians, .025), percentile(medians, .975)],
                minimum_observed_ratio=min(values), maximum_observed_ratio=max(values))


def quality_evidence(path):
    if path is None:
        return None
    gate = json.loads(path.read_text())
    require(gate.get('completed') is True and gate.get('passed') is True
            and gate.get('calibration_only') is not True and gate.get('mode') not in ('discovery', 'calibration'),
            'Provided quality report must explicitly be completed/passed, not calibration or discovery')
    return dict(path=str(path.resolve()), sha256=sha(path), reported_completed=True, reported_passed=True,
                scope_match_verified=False,
                reported_selected_steps=gate.get('selected_steps'), reported_summary=gate.get('summary'),
                reported_oracle_precision={key: gate.get('oracle', {}).get(key) for key in ('weight_precision', 'kv_precision')},
                limitation='Only the supplied report\'s status was checked. Its build, precision, prompt, position and conversation coverage must be matched manually to the performance comparison; no baseline quality scope is inferred.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--baseline', required=True, help='Exact configuration name; never choose a baseline implicitly')
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261112)
    parser.add_argument('--quality-gate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1000 <= args.resamples <= 100000:
        parser.error('Use1000..100000bootstrap resamples')
    if args.output.exists():
        parser.error('Preserve previous evidence; choose a fresh analysis output')
    input_digest = sha(args.report)
    source = json.loads(args.report.read_text())
    names, visits, orders = validate(source, args.baseline)
    profiles = profiles_for(source)
    quality = quality_evidence(args.quality_gate)
    rng = random.Random(args.seed)
    rounds = source['rounds']
    indices = [[rng.randrange(rounds) for _ in range(rounds)] for _ in range(args.resamples)]
    result = dict(completed=False, source_report=str(args.report.resolve()), source_report_sha256=input_digest,
                  analyzer_sha256=sha(__file__), python_version=sys.version,
                  precision=source['precision'], baseline=args.baseline,
                  context_suite=source.get('context_suite', 'standard'),
                  configurations=source['configurations'], rounds=rounds, tokens=source['tokens'],
                  model=source.get('model'), source_builds=source.get('builds'),
                  bootstrap=dict(method='Paired-round percentile bootstrap of the median ratio',
                                 confidence_level=.95, resamples=args.resamples, seed=args.seed,
                                 resampling_unit='One complete measured round, preserving baseline/candidate pairing',
                                 common_resample_indices_across_contexts_metrics_and_configurations=True),
                  quality_evidence=quality, quality_pass_inferred=False,
                  performance_claim_eligible=False, manual_quality_and_uncertainty_review_required=True,
                  configuration_order_by_round=[[orders[index, order] for order in range(len(names))] for index in range(rounds)],
                  comparisons=[], limitations=[
                      'Ratios above1favor the candidate. Intervals summarize paired observations; they do not automatically establish a performance win.',
                      'Percentile bootstrap assumes measured rounds are sufficiently exchangeable. It does not remove thermal drift, background activity, affinity differences or systematic benchmark bias.',
                      'Intervals are unadjusted for multiple configurations, metrics or contexts. Choosing the best observed result can introduce selection bias.',
                      'A passed quality report, when supplied, is evidence only within its own stated scope. This analyzer does not promote it into a general baseline-quality claim.',
                      'Single-stream decode, TTFT and request latency remain separate metrics; no composite winner or cross-precision speedup is inferred.'])
    for candidate in names:
        if candidate == args.baseline:
            continue
        for profile in profiles:
            pairs = []
            for index in range(rounds):
                reference_visit = visits[index, args.baseline]
                candidate_visit = visits[index, candidate]
                reference = reference_visit['profiles'][profile]['scored']
                measured = candidate_visit['profiles'][profile]['scored']
                ratios = {label: (measured[metric] / reference[metric] if label == 'decode_speedup'
                                  else reference[metric] / measured[metric]) for label, metric in METRICS.items()}
                pairs.append(dict(round=index,
                                  baseline_before_candidate=reference_visit['order_index'] < candidate_visit['order_index'],
                                  baseline={metric: reference[metric] for metric in METRICS.values()},
                                  candidate={metric: measured[metric] for metric in METRICS.values()}, ratios=ratios))
            result['comparisons'].append(dict(candidate=candidate, profile=profile,
                prompt_tokens=profiles[profile], raw_pairs=pairs,
                order_counts=dict(baseline_first=sum(pair['baseline_before_candidate'] for pair in pairs),
                                  candidate_first=sum(not pair['baseline_before_candidate'] for pair in pairs)),
                metrics={metric: interval([pair['ratios'][metric] for pair in pairs], indices) for metric in METRICS}))
    if sha(args.report) != input_digest:
        raise RuntimeError('Confirmation report changed during analysis')
    if quality and sha(args.quality_gate) != quality['sha256']:
        raise RuntimeError('Provided quality report changed during analysis')
    result['completed'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
