"""Transport/diagnostics shared by the experimental runtime-dispatch checks."""
import contextlib
import hashlib
import json
import os
from pathlib import Path

from rtl_model import RTLModel
from compare_query_value_pairs import read_dump_logits
from check_phase_pools_real import telemetry

ENV = 'QWEN_EXPERIMENTAL_QUERY_VALUE_PAIRS'
POLICY = 'fp32_fma_b8_aligned_v1'
PATTERN = [9707, 11, 358, 40, 1097, 264, 1296, 198]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def artifact(binary, expected_sha):
    binary = Path(binary).resolve()
    if sha(binary) != expected_sha:
        raise RuntimeError('Executable differs from explicitly pinned SHA256')
    manifest = json.loads((binary.parent.parent / 'manifest.json').read_text())
    if manifest['binary_sha256'] != expected_sha:
        raise RuntimeError('Manifest/executable mismatch')
    for key in ('attention_query_value_dispatch', 'attention_query_value_pairs',
                'attention_qk2', 'attention_head_pairs', 'attention_value_pairs',
                'static_kv', 'batch_attention', 'fp32_prefill_fma',
                'phase_pools_experiment', 'fp32_fma_canonical_prefix'):
        if manifest['options'].get(key) is not True:
            raise RuntimeError(f'Missing required experimental capability: {key}')
    return dict(path=str(binary), sha256=expected_sha, options=manifest['options'])


@contextlib.contextmanager
def selector_environment(selector):
    previous = os.environ.get(ENV)
    os.environ[ENV] = selector
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = previous


def policy(model, selector):
    actual = json.loads(model.command('query_value_policy'))
    minimum = -1 if selector == 'OFF' else 0 if selector == 'ON' else int(selector.split(':')[1])
    expected = dict(selector=selector, min_context=minimum,
                    context_metric='paired_query_common_kv_tokens', requires_second_query=True)
    if any(type(actual.get(k)) is not type(v) or actual[k] != v for k, v in expected.items()):
        raise RuntimeError(f'Wrong runtime query-value policy: {actual}')
    return actual


def start(model, selector):
    model.phase_pool_selector = 'PREFILL_12'
    # This standalone experiment starts models sequentially, never from HTTP.
    # Scope environment to Popen; restore it before starting any other model.
    with selector_environment(selector):
        model.start()
    actual = json.loads(model.command('precision'))
    expected = dict(precision_profile='fp32_b1_fp32_fma_prefill', scalar_projection='fp32',
                    prefill_projection='fp32', verification_projection='fp32', attention_kv='fp64',
                    canonical_prefix_policy=POLICY, scalar_accumulation='separate',
                    prefill_accumulation='fused', verification_accumulation='separate',
                    vocabulary_projection='fp32', vocabulary_accumulation='separate',
                    max_verification_block=4)
    if any(type(actual.get(k)) is not type(v) or actual[k] != v for k, v in expected.items()):
        raise RuntimeError(f'Wrong numerical profile: {actual}')
    return dict(precision=actual, query_value=policy(model, selector), phase=telemetry(model, True))


def make_models(binary, count=3):
    return [RTLModel(binary=binary, workers=8, lanes=8, block_size=4, prefill_size=8)
            for _ in range(count)]


def fill(model, tokens):
    for offset in range(0, len(tokens), 8):
        model.prefill(tokens[offset:offset + 8])


def repeat_tokens(count):
    return (PATTERN * ((count + 7) // 8))[:count]


def exact(models, outputs, directory, label):
    if any(output != outputs[0] for output in outputs[1:]):
        raise AssertionError(f'{label}: IDs/positions differ')
    bits = [read_dump_logits(m, Path(directory) / f'{i}.bin').tobytes()
            for i, m in enumerate(models)]
    if any(value != bits[0] for value in bits[1:]):
        raise AssertionError(f'{label}: full-vocabulary logit bits differ')
    return dict(label=label, outputs=outputs[0], full_vocabulary_bit_exact=True,
                logits_sha256=hashlib.sha256(bits[0]).hexdigest())


def route_call(model, selector, prefix, count, call, scalar=False):
    before = policy(model, selector)
    result = call()
    after = policy(model, selector)
    minimum = before['min_context']
    paired = 0 if scalar else sum(query + 1 < count and minimum >= 0
                                  and prefix + query + 1 >= minimum
                                  for query in range(0, count, 2))
    fallback = 0 if scalar else (count + 1) // 2 - paired
    expected = dict(paired_head_pairs=448 * paired, fallback_head_pairs=448 * fallback)
    actual = {key: after[key] - before[key] for key in expected}
    if actual != expected:
        raise AssertionError(f'Actual SV route mismatch {selector} prefix={prefix} count={count}: {actual} != {expected}')
    return result, dict(selector=selector, prefix=prefix, count=count, scalar=scalar,
                        expected=expected, actual=actual)
