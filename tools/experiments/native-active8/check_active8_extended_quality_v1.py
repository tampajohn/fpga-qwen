#!/usr/bin/env python3
"""Predetermined extended conversation probes; separate from the frozen nine-case gate.

Greedy uncached fixture conversations compare active8 diagnostic with pristine build20.
Exact token equivalence and strict answer rubrics are independent outcomes. These
new prompts were fixed before their first evaluation; no neural judge or grader
adaptation is used. This is a small synthetic probe set, not broad model accuracy.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools"))

from benchmark_native_cpu import OriginalEngine
from benchmark_native_phase0 import Phase0NativeEngine, freeze_build, guard, sha, verify_build
from check_native_phase0_quality import validate_response
from chat_protocol import ChatProtocol
from conversation_eval import QualityCase, conversation, evaluate_response, suite_fingerprint
from experiment import ROOT, locate_model


CASES = (
    QualityCase('latest_opposite', 'latest question focus', conversation(
        'What is the opposite of hot?', 'Cold.',
        'Now what is the opposite of empty? Reply with only one word.'),
        dict(kind='exact',answer='full'),24),
    QualityCase('latest_sort', 'latest question focus', conversation(
        'Say the word orange.', 'orange',
        'New task: sort 14, 3, 8 from smallest to largest. Reply only with comma-separated numbers.'),
        dict(kind='integer_list',answer=[3,8,14]),24),
    QualityCase('latest_object', 'latest question focus', conversation(
        'A robin is a bird. Tell me its category.', 'bird',
        'A salmon is a fish. What category is the salmon? Reply with only the category.'),
        dict(kind='exact',answer='fish'),24),
    QualityCase('corrected_destination', 'corrected facts', conversation(
        'The parcel goes to Bristol.', 'The destination is Bristol.',
        'Correction: send it to York instead.', 'The destination is now York.',
        'Where should the parcel go? Reply with only the city.'),dict(kind='exact',answer='York'),24),
    QualityCase('corrected_count', 'corrected facts', conversation(
        'We need 6 chairs.', 'We need 6 chairs.',
        'Update: we need 11 chairs, not 6.', 'The updated count is 11.',
        'How many chairs do we need now? Reply with only the number.'),dict(kind='exact',answer='11'),24),
    QualityCase('corrected_owner', 'corrected facts', conversation(
        'The green notebook belongs to Hana.', 'Hana owns the green notebook.',
        'Correction: it belongs to Eli.', 'Eli owns it.',
        'Who owns the green notebook? Reply with only the name.'),dict(kind='exact',answer='Eli'),24),
    QualityCase('recall_user_fact', 'recall versus quote', conversation(
        'My preferred fruit is pear.', 'I have recorded your preference.',
        'Which fruit did I name? Reply with only the fruit.'),dict(kind='exact',answer='pear'),24),
    QualityCase('quote_assistant_verbatim', 'recall versus quote', conversation(
        'My preferred fruit is pear.', 'I have recorded your preference.',
        'Quote your preceding reply exactly, with no introduction.'),
        dict(kind='quote',answer='I have recorded your preference.'),32),
    QualityCase('quote_user_verbatim', 'recall versus quote', conversation(
        'The copper lantern is beside the door.', 'I noted the lantern location.',
        'Quote my first sentence exactly. Do not quote your reply.'),
        dict(kind='quote',answer='The copper lantern is beside the door.'),32),
    QualityCase('loop_recovery_arithmetic', 'reported repetition recovery', conversation(
        'hi', 'Hello! How can I assist you today?',
        'what did you say', 'Hello! What did you say?',
        '?', 'Hello! What did you say?',
        'You keep repeating. Answer a new question: what is 6 + 7? Reply only with the number.'),
        dict(kind='exact',answer='13'),24),
    QualityCase('loop_recovery_identity', 'reported repetition recovery', conversation(
        "I'm confused.", "Hello! I'm confused too. Could you clarify what you're asking?",
        'What are you?', "Hello! I'm confused too. Could you clarify what you're asking?",
        'Stop repeating that sentence. Give only your model name as stated in your instructions.'),
        dict(kind='exact',answer='Qwen3-0.6B'),32),
    QualityCase('loop_recovery_yes_no', 'reported repetition recovery', conversation(
        'You keep repeating.', 'Hello! What did you say?',
        'I just told you.', 'Hello! What did you say?',
        'Is 2 smaller than 9? Reply with only yes or no.'),dict(kind='exact',answer='yes'),24),
    QualityCase('arithmetic_subtraction', 'simple arithmetic', conversation(
        'Compute 23 minus 9. Reply with only the result.'),dict(kind='exact',answer='14'),24),
    QualityCase('arithmetic_groups', 'simple arithmetic', conversation(
        'Four boxes each contain 3 pencils. How many pencils are there in total? Reply with only the number.'),
        dict(kind='exact',answer='12'),24),
    QualityCase('arithmetic_remaining', 'simple arithmetic', conversation(
        'I have 18 tickets and give away 7. Then I receive 2 more. How many tickets do I have? Reply only with the number.'),
        dict(kind='exact',answer='13'),24),
    QualityCase('unknown_birthday', 'honest missing evidence', conversation(
        'Record: Nia lives in a red house. Using only the record, what is her birthday? '
        'If no birthday is given, reply with exactly unknown.'),dict(kind='exact',answer='unknown'),24),
    QualityCase('unknown_code', 'honest missing evidence', conversation(
        'My suitcase is black.', 'Your suitcase is black.',
        'What is my suitcase combination? I have not provided it. Reply with exactly unknown if it cannot be determined.'),
        dict(kind='exact',answer='unknown'),24),
    QualityCase('unknown_cutoff', 'honest missing evidence', conversation(
        'For this question, use only this supplied record: "Model: Demo; owner: Lee". '
        'What training cutoff date does the record specify? Reply with exactly unknown when absent.'),
        dict(kind='exact',answer='unknown'),24),
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-build',type=Path,default=ROOT/'build/native-phase0/kleidi-on-lto-on')
    parser.add_argument('--model',type=Path,default=ROOT/'build/bench/qwen3-0.6b-f16.gguf')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--guard-state-dir',type=Path,default=ROOT/'build/benchmark-status')
    args=parser.parse_args()
    guarded=guard(args.guard_state_dir)
    output=args.output.resolve();logroot=output.parent/(output.stem+'-logs')
    if output.exists() or logroot.exists():parser.error('Use fresh report and log paths')
    output.parent.mkdir(parents=True,exist_ok=True);logroot.mkdir()
    binary,weights,model=args.binary.resolve(),args.weights.resolve(),args.model.resolve()
    checkpoint=locate_model(args.checkpoint);protocol=ChatProtocol(checkpoint)
    reference=(ROOT/'build/native-cpu/build-20/qwen_cpu').resolve()
    baseline=dict(name='native-build20-f16',binary=str(reference),decode_threads=12,prefill_threads=6)
    configs=[baseline,dict(name='original-native-f16',binary=str(binary),decode_threads=12,prefill_threads=6)]
    build=json.loads((reference.parent/"manifest.json").read_text());assert build["completed"] and build["binary_sha256"]==sha(reference)
    prompts={case.name:protocol.prepare(dict(session='extended-'+case.name,messages=list(case.messages),
        max_new_tokens=case.max_new_tokens,decoding='greedy')).ids for case in CASES}
    fixture=[dict(name=c.name,category=c.category,messages=c.messages,rubric=c.expected,max_new_tokens=c.max_new_tokens,
                  prompt_ids=prompts[c.name]) for c in CASES]
    sources=[Path(__file__).resolve()]+[ROOT/'tools'/name for name in ('benchmark_native_cpu.py','benchmark_native_phase0.py',
        'benchmark_matched.py','native_cpu_chat.py','check_native_phase0_quality.py','chat_protocol.py','conversation_eval.py','experiment.py','benchmark_guard.py','suffix.py')]
    inputs=[binary,binary.parent/'manifest.json',weights,Path(str(weights)+'.index'),Path(str(weights)+'.manifest.json'),reference,reference.parent/'manifest.json']
    inputs += [checkpoint/name for name in ('tokenizer.json','tokenizer_config.json','config.json','model.safetensors')]
    report=dict(completed=False,passed=None,gpu=False,pytorch=False,timing_claim=False,performance_claim_eligible=False,
        suite='extended-conversation-v1',suite_sha256=suite_fingerprint(CASES),cases=fixture,configurations=configs,
        baseline_build=build,guard_before=guarded,decoding='greedy',prefix_reuse=False,suffix_drafting=False,
        results=[],sessions=[],limitations=['Predetermined synthetic fixture conversations, not generated multi-turn trajectories or broad accuracy.',
            'Strict exact/case-insensitive answer, quote or integer-list rubrics are fixed before execution; punctuation can fail exact-answer instructions.',
            'Exact baseline agreement does not establish correctness. Existing baseline failures remain visible.',
            'This separate18-case suite does not change the frozen nine-case gate, numerical tolerance gates, or browser lifecycle checks.'])
    def save():
        pending=output.with_suffix('.pending.json');pending.write_text(json.dumps(report,indent=2)+'\n');pending.replace(output)
    try:
        save();report['inputs']={str(p):sha(p) for p in inputs};report['source_hashes']={str(p):sha(p) for p in sources}
        manifest=json.loads((binary.parent/'manifest.json').read_text())
        pack=json.loads(Path(str(weights)+'.manifest.json').read_text())
        if manifest.get('completed') is not True or manifest.get('binary_sha256')!=report['inputs'][str(binary)]:raise RuntimeError('Native build identity mismatch')
        if pack.get('completed') is not True or pack.get('format')!='QWCPU01' or pack['binary']['sha256']!=report['inputs'][str(weights)] or pack['index']['sha256']!=report['inputs'][str(weights)+'.index']:raise RuntimeError('Pack identity mismatch')
        report['native_build']=manifest
        for config in configs:
            guard(args.guard_state_dir)
            engine=OriginalEngine(protocol,Path(config['binary']),weights,12,6,logroot/config['name'])
            record=dict(configuration=config['name']);report['sessions'].append(record)
            try:
                engine.start();record.update(command=engine.command,pid=engine.process.pid,metadata=engine.metadata)
                for case in CASES:
                    result=validate_response(engine.generate(prompts[case.name],case.max_new_tokens,False),prompts[case.name],case.max_new_tokens)
                    grade=evaluate_response(case,result['text'])
                    prior=[m['content'].strip().casefold() for m in case.messages if m['role']=='assistant']
                    report['results'].append(dict(configuration=config['name'],case=case.name,category=case.category,result=result,grade=grade,
                        repeats_prior_assistant_verbatim=result['text'].strip().casefold() in prior,
                        repetition_note='Diagnostic only; intentional quoting is not automatically an error.'))
                    save()
            finally:
                engine.close()
                if hasattr(engine,'log_path') and engine.log_path.is_file():record['runtime_log']=dict(path=str(engine.log_path),sha256=sha(engine.log_path))
                save()
        byconfig={c['name']:{r['case']:r for r in report['results'] if r['configuration']==c['name']} for c in configs}
        report['suite_accuracy']=[dict(configuration=c['name'],correct=sum(r['grade']['passed'] for r in byconfig[c['name']].values()),total=len(CASES)) for c in configs]
        report['comparisons']=[];report['baseline_wrong_answers']=[]
        for case in CASES:
            left,right=(byconfig[c['name']][case.name] for c in configs)
            a,b=left['result'],right['result']
            if not left['grade']['passed']:report['baseline_wrong_answers'].append(left)
            report['comparisons'].append(dict(case=case.name,
                output_ids_identical=a['output_token_ids']==b['output_token_ids'],
                stop_identical=(a['finish_reason'],a['eos_observed'])==(b['finish_reason'],b['eos_observed']),
                text_identical=a['text']==b['text'],grading_identical=left['grade']==right['grade'],
                regressed_checks=[key for key,value in left['grade']['checks'].items() if value and not right['grade']['checks'].get(key,False)],
                reference_correct=left['grade']['passed'],candidate_correct=right['grade']['passed'],
                new_regression=left['grade']['passed'] and not right['grade']['passed'],
                improvement=not left['grade']['passed'] and right['grade']['passed']))
        report['exact_trajectory_gate_passed']=all(r['output_ids_identical'] and r['stop_identical'] for r in report['comparisons'])
        report['no_new_case_regressions']=not any(r['new_regression'] for r in report['comparisons'])
        report['no_new_check_regressions']=not any(r['regressed_checks'] for r in report['comparisons'])
        report['all_cases_correct']=all(r['grade']['passed'] for r in byconfig[configs[1]['name']].values())
        report['passed']=report['exact_trajectory_gate_passed'] and report['no_new_case_regressions'] and report['no_new_check_regressions']
        report['pass_scope']='Exact greedy trajectory equivalence and no new case regressions; does not imply all answers correct.'
        assert sha(reference)==build["binary_sha256"]
        if report['inputs']!={str(p):sha(p) for p in inputs} or report['source_hashes']!={str(p):sha(p) for p in sources} or report['suite_sha256']!=suite_fingerprint(CASES):raise RuntimeError('Inputs/source/fixtures changed during evaluation')
        report.update(guard_after=guard(args.guard_state_dir),completed=True)
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error));raise
    finally:save()
    if report['passed'] is not True:raise SystemExit(1)
    print(output)


if __name__=='__main__':main()
