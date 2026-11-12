"""Existing cache/suffix suites on one frozen fused-profile capability binary."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--phase-selector',choices=['OFF','PREFILL_12'],default='PREFILL_12')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();binary=a.binary.resolve();before=digest(binary)
    a.output.parent.mkdir(parents=True,exist_ok=True);stem=str(a.output.with_suffix(''))
    os.environ.update(SV_CANONICAL_BINARY=str(binary),SV_CANONICAL_BLOCK_SIZE='4',
        SV_CANONICAL_PROFILE='fp32_b1_fp32_fma_prefill',SV_CANONICAL_FIXED_WORKERS='8',
        SV_FP32_VERIFY_BINARY=str(binary),SV_FP32_VERIFY_PROFILE='fp32_b1_fp32_fma_prefill',
        SV_PHASE_POOL_SELECTOR=a.phase_selector,CANONICAL_REPORT=stem+'-canonical.json',
        CANONICAL_DECODE_REPORT=stem+'-decode-exclusion.json',FP32_SUFFIX_REPORT=stem+'-suffix.json')
    sources=[Path(__file__),ROOT/'tests/test_canonical_prefix_real.py',ROOT/'tests/test_suffix_real.py',
             ROOT/'tools/rtl_chat.py',ROOT/'tools/canonical_prefix.py']
    hashes={str(x.relative_to(ROOT)):digest(x) for x in sources}
    suites=['tests.test_canonical_prefix_real.RealCanonicalPrefixTests','tests.test_suffix_real.RealFP32SuffixTests']
    log=io.StringIO()
    class Tee:
        def write(self,value):sys.stdout.write(value);log.write(value)
        def flush(self):sys.stdout.flush()
    result=unittest.TextTestRunner(stream=Tee(),verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(suites))
    stable=digest(binary)==before and hashes=={str(x.relative_to(ROOT)):digest(x)for x in sources}
    report=dict(completed=True,passed=result.wasSuccessful() and not result.skipped and stable,
        binary=str(binary),binary_sha256=before,artifacts_unchanged=stable,source_hashes=hashes,
        precision_profile='fp32_b1_fp32_fma_prefill',policy='fp32_fma_b8_aligned_v1',
        phase_pool_selector=a.phase_selector,worker_scope='fixed8 scalar/verification; phase-selected prefill',
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=result.skipped,
        log=log.getvalue(),timing_claim=False)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
