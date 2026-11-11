"""Declared, objective conversation-quality gates; no neural-framework dependency.

This evaluates behavior, not speed. Gates are frozen independently of model output.
A passing score is a small regression suite, not a general model-quality benchmark.
"""
from dataclasses import dataclass
import argparse
import functools
import hashlib
import json
from pathlib import Path
import re
import threading
from datetime import datetime, timezone


@dataclass(frozen=True)
class QualityCase:
    name: str
    category: str
    messages: tuple[dict, ...]
    expected: dict
    max_new_tokens: int = 48


def conversation(*items):
    return tuple(dict(role='user' if i % 2 == 0 else 'assistant', content=text)
                 for i, text in enumerate(items))


CASES = (
    QualityCase('fact_recall', 'history retrieval', conversation(
        'My locker code is 7319. Remember it for this conversation.',
        'Your locker code is 7319.',
        'What is my locker code? Reply with just the four digits.'),
        {'kind': 'exact', 'answer': '7319'}, 16),
    QualityCase('corrected_fact', 'history correction', conversation(
        'The meeting is in Oslo.', 'The meeting is in Oslo.',
        'Correction: the meeting moved to Lima.', 'The meeting is now in Lima.',
        'Which city is the meeting in now? Reply with only the city.'),
        {'kind': 'exact', 'answer': 'Lima'}, 16),
    QualityCase('assistant_quote', 'assistant history retrieval', conversation(
        'Give me a short greeting.', 'Welcome aboard, Maya!',
        'Quote your previous reply exactly. Do not add anything else.'),
        {'kind': 'quote', 'answer': 'Welcome aboard, Maya!'}, 24),
    QualityCase('task_switch', 'latest instruction', conversation(
        'Hello.', 'Hello! How can I help?',
        'I am confused.', 'Could you clarify what you mean?',
        'New task: sort these numbers from smallest to largest: 9, 2, 5. '
        'Reply only with the three numbers separated by commas.'),
        {'kind': 'integer_list', 'answer': [2, 5, 9]}, 24),
    QualityCase('json_extraction', 'structured extraction', conversation(
        'Extract the name and color from this record: "Name: Mira; Color: teal". '
        'Return only a JSON object with keys "name" and "color".'),
        {'kind': 'json', 'answer': {'name': 'Mira', 'color': 'teal'}}, 48),
    QualityCase('two_step_arithmetic', 'arithmetic', conversation(
        'There are 17 red beads and 8 blue beads. Five beads are removed. '
        'How many beads remain? Reply with only the number.'),
        {'kind': 'exact', 'answer': '20'}, 16),
    QualityCase('distractor_recall', 'selective retrieval', conversation(
        'The project codename is Cedar. Its room is 42 and its launch month is May.',
        'The project is Cedar, in room 42, launching in May.',
        'Ignore the room and month. What is the project codename? Reply with one word.'),
        {'kind': 'exact', 'answer': 'Cedar'}, 16),
    QualityCase('evidence_only', 'grounded unknown', conversation(
        'Record: Ravi owns a blue bicycle. The record does not state his age. '
        'Using only this record, what is Ravi\'s age? Reply "unknown" if it is absent.'),
        {'kind': 'exact', 'answer': 'unknown'}, 16),
    QualityCase('new_question_after_repetition', 'repetition recovery', conversation(
        'Hello.', 'Hello! How can I assist you today?',
        'What did you say?', 'Hello! What did you say?',
        'I am confused.', 'Hello! I am confused too. Could you clarify what you are asking?',
        'Answer this new question: which is larger, 12 or 7? Reply with just the number.'),
        {'kind': 'exact', 'answer': '12'}, 16),
)


# Separate from the general quality score: identity is given by the system prompt.
REPORTED_RECOVERY = QualityCase('reported_recovery', 'reported identity regression', conversation(
    'hi', 'Hello! How can I assist you today?',
    'what did you say', 'Hello! What did you say?',
    '?', 'Hello! What did you say?',
    "i'm confused", "Hello! I'm confused too. Could you clarify what you're asking?",
    'what are you'),
    {'kind': 'contains_any', 'answer': ['assistant', 'qwen', 'language model']}, 48)
EXTRA_CASES = (REPORTED_RECOVERY,)


def evaluate_response(case: QualityCase, text: str) -> dict:
    """Strict instruction checks; no answer-dependent grading or model judge."""
    cleaned = text.strip()
    expected = case.expected
    answer = expected['answer']
    kind = expected['kind']
    checks = {'nonempty': bool(cleaned)}
    if kind == 'exact':
        checks['answer_present'] = bool(re.search(r'(?<!\w)'+re.escape(answer)+r'(?!\w)', cleaned, re.IGNORECASE))
        checks['exact_answer'] = cleaned.casefold() == answer.casefold()
    elif kind == 'quote':
        # Quote marks are acceptable, but prefixes/explanations are not.
        checks['verbatim_quote'] = cleaned.strip('"“”') == answer
    elif kind == 'integer_list':
        checks['only_comma_separated_integers'] = bool(re.fullmatch(r'\d+\s*,\s*\d+\s*,\s*\d+', cleaned))
        checks['correct_order'] = [int(n) for n in re.findall(r'\d+', cleaned)] == answer
    elif kind == 'contains_any':
        checks['identity_present'] = any(term in cleaned.casefold() for term in answer)
    elif kind == 'json':
        try:
            decoded = json.loads(cleaned)
        except (ValueError, TypeError):
            decoded = None
        checks['valid_json_only'] = isinstance(decoded, dict)
        checks['exact_fields_and_values'] = decoded == answer
    else:
        raise ValueError(f'Unknown grader kind: {kind}')
    return dict(passed=all(checks.values()), checks=checks,
                details=f'{kind} target: {json.dumps(answer, ensure_ascii=False)}')


def suite_fingerprint(cases=CASES):
    data = [dict(name=c.name, category=c.category, messages=c.messages,
                 expected=c.expected, max_new_tokens=c.max_new_tokens) for c in cases]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def cache_warmup_messages(case: QualityCase):
    """Prior user-ended request; fixture assistant text is retained only in full input."""
    if len(case.messages) < 3:
        raise ValueError('A cache check requires at least two user turns')
    return case.messages[:-2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['rtl', 'llama-f32', 'llama-q4'], default='rtl')
    parser.add_argument('--workers', type=int, choices=[0, 1, 2, 4, 8], default=8)
    parser.add_argument('--decoding', choices=['chat', 'greedy'], default='chat')
    parser.add_argument('--seeds', default='7,42,123')
    parser.add_argument('--cases', default='', help='Comma-separated names; empty means all')
    parser.add_argument('--presence-penalty', type=float, help='RTL-only controlled sampler ablation, 0..2')
    parser.add_argument('--repeatability', action='store_true', help='Repeat from an empty cache with the same seed')
    parser.add_argument('--cache-check', action='store_true', help='Compare warmed conversation cache against fresh results')
    parser.add_argument('--output', type=Path, default=Path('build/conversation-quality.json'))
    parser.add_argument('--fail-on-quality', action='store_true')
    args = parser.parse_args()
    if args.presence_penalty is not None and (args.backend != 'rtl' or args.decoding != 'chat' or not 0 <= args.presence_penalty <= 2):
        parser.error('--presence-penalty requires RTL chat decoding and a value from 0 to 2')
    seeds = [int(s) for s in args.seeds.split(',')]
    if any(not 0 <= seed <= 4294967295 for seed in seeds):
        parser.error('Seeds must be unsigned 32-bit integers')
    names = args.cases.split(',') if args.cases else [case.name for case in CASES]
    available = CASES + EXTRA_CASES
    unknown = set(names)-{case.name for case in available}
    if unknown:
        parser.error(f'Unknown cases: {sorted(unknown)}')
    cases = [case for case in available if case.name in names]
    # Lazy imports make graders usable in tests without tokenizers or model files.
    try:
        from .rtl_chat import RTLChatEngine
        from .chat_backends import BackendRouter
    except ImportError:
        from rtl_chat import RTLChatEngine
        from chat_backends import BackendRouter
    engine = RTLChatEngine(workers=args.workers)
    if args.presence_penalty is not None:
        # Only the transport's explicit sampler argument changes. Model weights,
        # inputs, sampling arithmetic, and the production default remain intact.
        engine.sim.token = functools.partial(engine.sim.token, presence_penalty=args.presence_penalty)
    router = BackendRouter(engine)
    binary = Path(__file__).resolve().parents[1] / ('build/full/Vqwen_behavioral' if args.backend == 'rtl' else 'build/llama-cpu/bin/llama-server')
    artifact_metadata = dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                             model_config_sha256=hashlib.sha256((engine.sim.model/'config.json').read_bytes()).hexdigest(),
                             started_at=datetime.now(timezone.utc).isoformat())
    records = []
    def generate(case, seed, session, messages=None, max_new=None):
        payload = dict(backend=args.backend, session=session, messages=list(messages or case.messages),
                       max_new_tokens=max_new or case.max_new_tokens, max_draft=0,
                       simulation_workers=args.workers, decoding=args.decoding, seed=seed)
        final = None
        for event in router.stream(router.prepare(payload), threading.Event()):
            if event['type'] == 'done':
                final = event
        if final is None:
            raise RuntimeError('Backend did not complete a reply')
        return final
    try:
        for case in cases:
            for seed in seeds:
                router.close()
                final = generate(case, seed, f'{case.name}-{seed}-fresh')
                record = dict(case=case.name, category=case.category, seed=seed,
                              text=final['text'], finish_reason=final['finish_reason'],
                              grade=evaluate_response(case, final['text']))
                if args.repeatability:
                    router.close()
                    repeated = generate(case, seed, f'{case.name}-{seed}-repeat')
                    record['same_seed_fresh_identical'] = repeated['text'] == final['text']
                    record['repeat_text'] = repeated['text']
                if args.cache_check and len(case.messages) > 1:
                    router.close()
                    session = f'{case.name}-{seed}-cache'
                    generate(case, seed, session, messages=cache_warmup_messages(case), max_new=16)
                    warmed = generate(case, seed, session)
                    record['cached_fresh_identical'] = warmed['text'] == final['text']
                    record['cached_text'] = warmed['text']
                    record['reused_prefix_tokens'] = warmed['metrics']['reused_prefix_tokens']
                records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        router.close()
    invariant_results = [value for record in records for key, value in record.items()
                         if key in ('same_seed_fresh_identical', 'cached_fresh_identical')]
    invariants = all(invariant_results) if invariant_results else None
    report = dict(backend=args.backend, workers=args.workers if args.backend == 'rtl' else None,
                  cpu_threads=max(1, args.workers) if args.backend == 'rtl' else 6, decoding=args.decoding,
                  presence_penalty=args.presence_penalty if args.presence_penalty is not None else (1.5 if args.decoding == 'chat' else 0),
                  seeds=seeds, artifacts=artifact_metadata, suite_sha256=suite_fingerprint(), selected_cases_sha256=suite_fingerprint(cases), quality_passes=sum(r['grade']['passed'] for r in records),
                  quality_total=len(records), implementation_invariant_checks=len(invariant_results),
                  implementation_invariants_passed=invariants, results=records,
                  caveats=['Small declared objective regression suite; not a broad model capability score.',
                           'Different backend RNG and penalty scopes prevent claiming sampled token equivalence.',
                           'Performance metrics are deliberately omitted; this runner is for quality.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(f"Quality: {report['quality_passes']}/{report['quality_total']}; invariants: {invariants}; {args.output}")
    if invariants is False or (args.fail_on_quality and report['quality_passes'] != report['quality_total']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
