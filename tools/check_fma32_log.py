"""Check actual generated-SV FMA outputs against test-only exact rational math."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from fp32_rounding_oracle import fused_multiply_add


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=re.findall(r'^FMA32 ([0-9a-fA-F]{8}) ([0-9a-fA-F]{8}) ([0-9a-fA-F]{8}) ([0-9a-fA-F]{8})$',args.log.read_text(),re.M)
    if len(rows)!=274:raise ValueError(f'Expected274 actualoutputrows, found{len(rows)}')
    cases=[]
    for index,row in enumerate(rows):
        a,b,c,actual=map(lambda value:int(value,16),row)
        expected=fused_multiply_add(a,b,c)
        nan=(expected&0x7fffffff)>0x7f800000
        passed=(actual&0x7fffffff)>0x7f800000 if nan else actual==expected
        cases.append(dict(index=index,inputs=[a,b,c],expected=expected,actual=actual,nan_classification_only=nan,passed=passed))
    report=dict(passed=all(case['passed']for case in cases),cases=cases,source_log=str(args.log),
                log_sha256=hashlib.sha256(args.log.read_bytes()).hexdigest(),
                checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                oracle_sha256=hashlib.sha256(Path(__file__).with_name('fp32_rounding_oracle.py').read_bytes()).hexdigest(),
                distinct_input_triples=len({tuple(case['inputs']) for case in cases}),
                coverage='18 edge cases plus256 deterministic bounded finite inputs; not uniform IEEE input coverage')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise SystemExit(1)
    print(f'PASS {len(cases)} generated-SV outputs against exact-rational FMA oracle')


if __name__=='__main__':main()
