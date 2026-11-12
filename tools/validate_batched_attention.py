"""Run existing exact-route, canonical-cache and suffix gates on a frozen artifact."""
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    paths=[args.reference.resolve(),args.candidate.resolve()]
    hashes=[hashlib.sha256(p.read_bytes()).hexdigest()for p in paths]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    stem=args.output.with_suffix('')
    os.environ.update(SV_BATCH_ATTENTION_BINARY=str(paths[1]),SV_BATCH_ATTENTION_REFERENCE=str(paths[0]),
        SV_CANONICAL_BINARY=str(paths[1]),SV_CANONICAL_BLOCK_SIZE='4',
        CANONICAL_REPORT=str(stem)+'-canonical.json',CANONICAL_DECODE_REPORT=str(stem)+'-decode-exclusion.json',
        SV_FP32_VERIFY_BINARY=str(paths[1]),FP32_SUFFIX_REPORT=str(stem)+'-suffix.json')
    suites=['tests.test_batched_attention_real.BatchedAttentionParity',
            'tests.test_canonical_prefix_real.RealCanonicalPrefixTests',
            'tests.test_suffix_real.RealFP32SuffixTests']
    log=io.StringIO()
    class Tee:
        def write(self,text):sys.stdout.write(text);log.write(text)
        def flush(self):sys.stdout.flush()
    suite=unittest.defaultTestLoader.loadTestsFromNames(suites)
    result=unittest.TextTestRunner(stream=Tee(),verbosity=2).run(suite)
    stable=hashes==[hashlib.sha256(p.read_bytes()).hexdigest()for p in paths]
    report=dict(completed=True,passed=result.wasSuccessful() and not result.skipped and stable,
        binaries=[dict(path=str(p),sha256=h)for p,h in zip(paths,hashes)],artifacts_unchanged=stable,
        suites=suites,tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=result.skipped,
        log=log.getvalue(),exactness='Full finite151936 binary64 logit bytes and IDs at matched logical positions; no tolerance',
        coverage=['prefill1..8 tails andoverwrite','85/127/128/129/255/256/452 contexts andcontinuations',
            'workers1/3/17 sampledB4','futuretoken exclusion aftertruncate','prefillcancel/restart',
            'existing33canonicalcomparisons andmatchingdecode exclusion','existing14suffixcomparisons+cancelrecovery'],
        timing_claim=False,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in
            [Path(__file__),ROOT/'tests/test_batched_attention_real.py',ROOT/'tests/test_canonical_prefix_real.py',ROOT/'tests/test_suffix_real.py']})
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
