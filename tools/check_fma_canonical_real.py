"""Existing cache/suffix suites on one frozen fused-profile capability binary."""
import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from weakref import WeakKeyDictionary
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

QUERY_VALUE_ENV = 'QWEN_EXPERIMENTAL_QUERY_VALUE_PAIRS'


def query_value_selector(value):
    if value in ('OFF', 'ON'):
        return value
    prefix = 'MIN_CONTEXT:'
    if value.startswith(prefix):
        digits = value[len(prefix):]
        if 1 <= len(digits) <= 4 and digits.isascii() and digits.isdigit() and 1 <= int(digits) <= 4096:
            return value
    raise argparse.ArgumentTypeError('Use OFF, ON or MIN_CONTEXT:N (1..4096)')


@contextmanager
def query_value_scope(selector, observations):
    if selector is None:
        yield
        return
    from tools.rtl_model import RTLModel
    previous = os.environ.get(QUERY_VALUE_ENV)
    original_start = RTLModel.start
    seen = WeakKeyDictionary()
    minimum = -1 if selector == 'OFF' else 0 if selector == 'ON' else int(selector.split(':')[1])
    expected = dict(selector=selector, min_context=minimum,
                    context_metric='paired_query_common_kv_tokens', requires_second_query=True)

    def checked_start(model, *args, **kwargs):
        result = original_start(model, *args, **kwargs)
        actual = json.loads(model.command('query_value_policy'))
        pid = model.process.pid
        prior_pid = seen.get(model)
        valid = (isinstance(actual, dict)
                 and all(type(actual.get(k)) is type(v) and actual[k] == v for k, v in expected.items())
                 and all(type(actual.get(k)) is int and actual[k] >= 0
                         for k in ('paired_head_pairs', 'fallback_head_pairs')))
        observations.append(dict(instance=id(model), pid=pid, new_process=pid != prior_pid,
                                 restarted=prior_pid is not None and pid != prior_pid,
                                 policy=actual, validated=valid))
        seen[model] = pid
        if not valid:
            raise RuntimeError(f'Query-value lifecycle selector mismatch: {actual}')
        return result

    os.environ[QUERY_VALUE_ENV] = selector
    try:
        with patch.object(RTLModel, 'start', checked_start):
            yield
    finally:
        if previous is None:
            os.environ.pop(QUERY_VALUE_ENV, None)
        else:
            os.environ[QUERY_VALUE_ENV] = previous


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--phase-selector',choices=['OFF','PREFILL_12'],default='PREFILL_12')
    p.add_argument('--query-value-selector', type=query_value_selector,
                   help='Assert runtime dispatch policy at every initial/restarted model start')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();binary=a.binary.resolve();before=digest(binary)
    a.output.parent.mkdir(parents=True,exist_ok=True);stem=str(a.output.with_suffix(''))
    os.environ.update(SV_CANONICAL_BINARY=str(binary),SV_CANONICAL_BLOCK_SIZE='4',
        SV_CANONICAL_PROFILE='fp32_b1_fp32_fma_prefill',SV_CANONICAL_FIXED_WORKERS='8',
        SV_FP32_VERIFY_BINARY=str(binary),SV_FP32_VERIFY_PROFILE='fp32_b1_fp32_fma_prefill',
        SV_PHASE_POOL_SELECTOR=a.phase_selector,CANONICAL_REPORT=stem+'-canonical.json',
        CANONICAL_DECODE_REPORT=stem+'-decode-exclusion.json',FP32_SUFFIX_REPORT=stem+'-suffix.json')
    sources=[Path(__file__),ROOT/'tests/test_canonical_prefix_real.py',ROOT/'tests/test_suffix_real.py',
             ROOT/'tools/rtl_chat.py',ROOT/'tools/canonical_prefix.py',ROOT/'tools/rtl_model.py']
    hashes={str(x.relative_to(ROOT)):digest(x) for x in sources}
    suites=['tests.test_canonical_prefix_real.RealCanonicalPrefixTests','tests.test_suffix_real.RealFP32SuffixTests']
    log=io.StringIO()
    class Tee:
        def write(self,value):sys.stdout.write(value);log.write(value)
        def flush(self):sys.stdout.flush()
    observations=[]
    with query_value_scope(a.query_value_selector, observations):
        result=unittest.TextTestRunner(stream=Tee(),verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(suites))
    stable=digest(binary)==before and hashes=={str(x.relative_to(ROOT)):digest(x)for x in sources}
    selector_checked = a.query_value_selector is None or bool(observations) and all(x['validated'] for x in observations)
    report=dict(completed=True,passed=result.wasSuccessful() and not result.skipped and stable and selector_checked,
        binary=str(binary),binary_sha256=before,artifacts_unchanged=stable,source_hashes=hashes,
        precision_profile='fp32_b1_fp32_fma_prefill',policy='fp32_fma_b8_aligned_v1',
        query_value_selector=a.query_value_selector,query_value_start_observations=observations,
        query_value_selector_checked=selector_checked if a.query_value_selector is not None else None,
        phase_pool_selector=a.phase_selector,worker_scope='fixed8 scalar/verification; phase-selected prefill',
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=result.skipped,
        log=log.getvalue(),timing_claim=False)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
