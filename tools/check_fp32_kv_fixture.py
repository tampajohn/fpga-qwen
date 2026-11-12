"""Storage-only exact rounding checks; never a model or performance evaluator."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import struct
import subprocess
from fp32_rounding_oracle import round_binary32, as_float

def bits(value):
    return struct.unpack('<Q', struct.pack('<d', value))[0]

def expected(value):
    # Preserve binary64 signed zero; rational values cannot represent its sign.
    if value == 0:
        return bits(value)
    try:
        return bits(as_float(round_binary32(Fraction.from_float(value))))
    except OverflowError:
        return bits(float("-inf") if value<0 else float("inf"))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    args=parser.parse_args()
    cases=[]
    def add(enabled,index,read,key,value,want_key,want_value,description):
        cases.append((f'{enabled} {index} {read} {bits(key):x} {bits(value):x}',
                      [int(bool(enabled and 0<=index<16)),int(0<=read<16),want_key,want_value],description))
    for i in range(16):
        add(0,0,i,0.,0.,0,0,'initialized read before write '+str(i))
    fixtures=[0.,-0.,1.+2.**-24,1.+3.*2.**-24,-1.-2.**-24,
              2.**-149,2.**-150,3.*2.**-150,-2.**-150,
              2.**-126-2.**-150,2.**-126,12345.678901,2.**128,-2.**128]
    for i,value in enumerate(fixtures):
        add(1,i,i,value,-value,expected(value),expected(-value),'same-edge write/read '+repr(value))
    add(1,15,15,3.5,-7.,expected(3.5),expected(-7.),'last legal address')
    add(1,-1,15,99.,99.,expected(3.5),expected(-7.),'negative write rejected, last cell retained')
    add(1,16,15,99.,99.,expected(3.5),expected(-7.),'past-end write rejected, last cell retained')
    add(0,15,-1,0.,0.,0,0,'negative read rejected')
    add(0,15,16,0.,0.,0,0,'past-end read rejected')
    add(0,15,15,99.,99.,expected(3.5),expected(-7.),'disabled write retains value')
    for i in range(16):
        key,value=(fixtures[i],-fixtures[i]) if i<len(fixtures) else ((3.5,-7.) if i==15 else (0.,0.))
        add(0,0,i,0.,0.,expected(key),expected(value),'all slots retained after invalid writes '+str(i))
    run=subprocess.run([str(args.binary.resolve())],input='\n'.join(c[0] for c in cases)+'\n',text=True,capture_output=True,check=True)
    lines=run.stdout.splitlines()
    assert len(lines)==len(cases),(lines,run.stderr)
    for line,(_,want,description) in zip(lines,cases):
        fields=line.split(); got=[int(fields[0]),int(fields[1]),int(fields[2],16),int(fields[3],16)]
        assert got==want,(description,got,want)
    print(json.dumps({'completed':True,'cases':len(cases),'checks':[c[2]for c in cases],
                      'rootp_widening':'harness asserts every valid read bit-exact to SV real ports',
                      'limits':'finite fixture only; tiny16-word capacity; no full-model quality or timing claim'},indent=2))
if __name__=='__main__':main()
