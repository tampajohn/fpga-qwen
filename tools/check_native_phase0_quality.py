"""Frozen nine-case native quality/regression comparison; no timing claims.

Use the same config-file schema as benchmark_native_phase0.py. The first
configuration is the explicit reference. All configurations use one precision,
identical prepared prompt IDs, natural EOS and each case's original token cap.
An equivalence gate can pass while suite accuracy remains low: unchanged wrong
answers are retained and never relabeled as improvements.
"""
import argparse
import json
from pathlib import Path

from benchmark_native_phase0 import (Phase0NativeEngine, freeze_build, guard,
                                     parse_configs, sha, verify_build)
from benchmark_matched import EOS
from chat_protocol import ChatProtocol
from conversation_eval import CASES, evaluate_response, suite_fingerprint
from experiment import ROOT, locate_model


def validate_response(result, prompt, maximum):
    tokens = result.get('output_token_ids')
    if (not isinstance(tokens, list) or any(type(token) is not int or not 0 <= token < 151936 for token in tokens)
            or result.get('output_token_count') != len(tokens) or len(tokens) > maximum):
        raise RuntimeError('Invalid native output-token payload or exceeded case cap')
    reason = result.get('finish_reason')
    eos = result.get('eos_observed')
    if reason == 'length':
        if len(tokens) != maximum or eos is not False or any(token in EOS for token in tokens):
            raise RuntimeError('Length-limited native response has inconsistent count/EOS')
    elif reason == 'eos':
        if eos is not True:
            raise RuntimeError('Natural EOS termination was not recorded')
        # An EOS token may be represented only by the stream's stop event.
        # If it is included in returned IDs, no later token may follow it.
        if any(token in EOS for token in tokens[:-1]):
            raise RuntimeError('Native response continued after EOS')
    else:
        raise RuntimeError('Native quality response did not finish naturally or at its declared cap')
    timings = result.get('backend_timings')
    if not isinstance(timings, dict) or timings.get('cache_n') != 0 or timings.get('prompt_n') != len(prompt):
        raise RuntimeError('Quality request reused prompt KV or evaluated different prompt work')
    if not isinstance(result.get('text'), str):
        raise RuntimeError('Missing native response text')
    # Deliberately exclude all measured timings from this quality report.
    return {key: result[key] for key in ('output_token_ids', 'output_token_count', 'text', 'finish_reason', 'eos_observed')}


def summarize(report):
    reference = report['configurations'][0]['name']
    reference_cases = {row['case']: row for row in report['results'] if row['configuration'] == reference}
    if len(reference_cases) != len(CASES):
        raise RuntimeError('Reference did not complete the entire fixed suite')
    report['suite_accuracy'] = []
    report['comparisons'] = []
    report['baseline_wrong_answers'] = [dict(case=row['case'], text=row['result']['text'], grade=row['grade'])
                                        for row in reference_cases.values() if not row['grade']['passed']]
    for config in report['configurations']:
        name = config['name']
        rows = [row for row in report['results'] if row['configuration'] == name]
        if len(rows) != len(CASES) or {row['case'] for row in rows} != set(reference_cases):
            raise RuntimeError('Configuration did not complete all nine unique cases')
        correct = sum(row['grade']['passed'] for row in rows)
        report['suite_accuracy'].append(dict(configuration=name, correct=correct, total=len(CASES),
                                             fraction=correct / len(CASES), all_cases_correct=correct == len(CASES)))
        if name == reference:
            continue
        comparisons = []
        for row in rows:
            before = reference_cases[row['case']]
            first, second = before['result'], row['result']
            checks = [key for key, passed in before['grade']['checks'].items()
                      if passed and row['grade']['checks'].get(key) is not True]
            comparisons.append(dict(case=row['case'],
                output_ids_identical=first['output_token_ids'] == second['output_token_ids'],
                finish_identical=first['finish_reason'] == second['finish_reason'] and first['eos_observed'] == second['eos_observed'],
                text_identical=first['text'] == second['text'], grading_identical=before['grade'] == row['grade'],
                reference_correct=before['grade']['passed'], candidate_correct=row['grade']['passed'],
                new_case_regression=before['grade']['passed'] and not row['grade']['passed'],
                improved_case=not before['grade']['passed'] and row['grade']['passed'],
                regressed_checks=checks))
        exact_trajectory = all(row['output_ids_identical'] and row['finish_identical'] for row in comparisons)
        exact_text = all(row['text_identical'] for row in comparisons)
        exact_grades = all(row['grading_identical'] for row in comparisons)
        no_case_regression = not any(row['new_case_regression'] for row in comparisons)
        no_check_regression = not any(row['regressed_checks'] for row in comparisons)
        report['comparisons'].append(dict(reference=reference, candidate=name, cases=comparisons,
            exact_trajectory_gate_passed=exact_trajectory, text_identical=exact_text,
            grading_identical=exact_grades, no_new_case_regressions=no_case_regression,
            no_new_check_regressions=no_check_regression,
            no_new_regressions=no_case_regression and no_check_regression,
            passed=exact_trajectory and exact_text and exact_grades and no_case_regression and no_check_regression))
    report['exact_trajectory_gate_passed'] = all(row['exact_trajectory_gate_passed'] for row in report['comparisons'])
    report['no_new_regressions'] = all(row['no_new_regressions'] for row in report['comparisons'])
    report['passed'] = all(row['passed'] for row in report['comparisons'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-file', type=Path, required=True)
    parser.add_argument('--precision', choices=('f16', 'f32', 'q4_k_m'), required=True)
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'build/bench')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT / 'build/benchmark-status')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Reuse the existing explicit config parser without introducing another
    # interpretation of build paths, names or requested thread counts.
    args.mode = 'quality'
    args.build_dir = args.configs = args.decode_threads = args.prefill_threads = None
    configs = parse_configs(args, parser)
    if len(configs) < 2 or any(config['name'] in ('.', '..') for config in configs):
        parser.error('Provide at least two distinct named configurations, reference first')
    if len(CASES) != 9:
        raise RuntimeError('Expected the existing frozen nine-case suite')
    guarded = guard(args.guard_state_dir)
    if args.output.exists():
        parser.error('Preserve existing evidence; choose a fresh report path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_root = args.output.parent / (args.output.stem + '-logs')
    if log_root.exists() and any(log_root.iterdir()):
        parser.error('Quality log directory already contains evidence')
    builds = {path: freeze_build(path) for path in dict.fromkeys(config['build_dir'] for config in configs)}
    model = (args.models_dir / f'qwen3-0.6b-{args.precision}.gguf').resolve()
    checkpoint = locate_model(args.checkpoint)
    protocol = ChatProtocol(checkpoint)
    prompts = {case.name: protocol.prepare(dict(session='phase0-quality-' + case.name,
               messages=list(case.messages), max_new_tokens=case.max_new_tokens, decoding='greedy')).ids for case in CASES}
    source_paths = [Path(__file__), ROOT / 'tools/benchmark_native_phase0.py', ROOT / 'tools/benchmark_matched.py',
                    ROOT / 'tools/chat_protocol.py', ROOT / 'tools/conversation_eval.py', ROOT / 'tools/benchmark_guard.py',
                    ROOT / 'tools/experiment.py']
    input_paths = [args.config_file.resolve(), model, *[checkpoint / name for name in
                   ('tokenizer.json', 'tokenizer_config.json', 'config.json')]]
    fingerprint = suite_fingerprint()
    report = dict(completed=False, passed=None, timing_claim=False, performance_claim_eligible=False,
                  gpu=False, pytorch=False, precision=args.precision, configurations=configs,
                  reference_configuration=configs[0]['name'], builds=builds,
                  inputs={str(path): sha(path) for path in input_paths},
                  source_hashes={str(path): sha(path) for path in source_paths},
                  suite_sha256=fingerprint,
                  suite=[dict(name=case.name, category=case.category, messages=list(case.messages), expected=case.expected,
                              max_new_tokens=case.max_new_tokens, prompt_ids=prompts[case.name]) for case in CASES],
                  guard_before=guarded, sessions=[], results=[],
                  decoding='Greedy, seed42, no penalties, no prompt cache, no suffix/speculation; natural EOS or original case cap.',
                  pass_scope='Exact native output trajectories/text/grading and no new regressions on these nine cases; not nine correct answers, full-logit validation or broad quality.',
                  limitations=['Reference is the first named configuration in this report, not an assumed historical score.',
                               'Unchanged wrong answers remain wrong. Matching baseline outputs or passing the equivalence gate is not an accuracy improvement.',
                               'Suite accuracy, case/check regressions and exact trajectory equivalence are reported separately; failed answers and strict grading remain visible.',
                               'Fixed fixture conversations do not exercise browser-cache lifecycle or sampled chat behavior. Numerical tolerance gates remain separate.',
                               'All configurations run sequentially at the same named weight precision and FP16 KV; no cross-precision equivalence or speed claim is made.'])
    def save():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    try:
        save()
        for config in configs:
            guard(args.guard_state_dir)
            session = dict(configuration=config['name'])
            report['sessions'].append(session)
            engine = Phase0NativeEngine(args.precision, config, builds[config['build_dir']], model,
                                        log_root / config['name'])
            try:
                engine.start()  # Startup timing is intentionally not retained here.
                session.update(command=engine.command, pid=engine.process.pid, metadata=engine.metadata,
                               removed_environment_keys=engine.removed_environment_keys)
                for case in CASES:
                    raw = engine.generate(prompts[case.name], case.max_new_tokens, fixed_count=False)
                    result = validate_response(raw, prompts[case.name], case.max_new_tokens)
                    grade = evaluate_response(case, result['text'])
                    record = dict(configuration=config['name'], case=case.name, category=case.category,
                                  result=result, grade=grade, evaluated_prompt_tokens=len(prompts[case.name]),
                                  cached_prompt_tokens=0, declared_token_cap=case.max_new_tokens)
                    report['results'].append(record)
                    save()
                    print(json.dumps(dict(configuration=config['name'], case=case.name,
                                          correct=grade['passed'], text=result['text']), ensure_ascii=False), flush=True)
            finally:
                if hasattr(engine, 'command'):
                    session['command'] = engine.command
                engine.close()
                if hasattr(engine, 'log_path') and engine.log_path.is_file():
                    session['runtime_log'] = dict(path=str(engine.log_path), sha256=sha(engine.log_path))
                save()
        summarize(report)
        for build in builds.values():
            verify_build(build)
        if any(sha(path) != digest for path, digest in report['inputs'].items()):
            raise RuntimeError('Quality model/config/tokenization input changed during evaluation')
        if any(sha(path) != digest for path, digest in report['source_hashes'].items()) or suite_fingerprint() != fingerprint:
            raise RuntimeError('Quality suite, grading or helper sources changed during evaluation')
        report['guard_after'] = guard(args.guard_state_dir)
        report['completed'] = True
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        save()
    if report['passed'] is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
