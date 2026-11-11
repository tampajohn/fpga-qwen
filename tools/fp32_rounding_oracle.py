"""Tiny test-only exact-rational IEEE binary32 rounding oracle; never inference.

Finite values, round-to-nearest/ties-to-even. Exact cancellation yields +0;
negative-zero arithmetic and overflow are outside these bounded fixtures.
"""
from fractions import Fraction
import struct


def rounded_quotient(n,d):
    q,r=divmod(n,d)
    return q+(2*r>d or (2*r==d and q%2==1))


def round_binary32(value):
    value=Fraction(value)
    if not value:return 0
    sign=0x80000000 if value<0 else 0
    n,d=abs(value.numerator),value.denominator
    exponent=n.bit_length()-d.bit_length()
    below_power=(n<d*(1<<exponent)) if exponent>=0 else (n*(1<<-exponent)<d)
    if below_power:exponent-=1
    scale=149 if exponent< -126 else 23-exponent
    if scale>=0:mantissa=rounded_quotient(n<<scale,d)
    else:mantissa=rounded_quotient(n,d<<-scale)
    if exponent< -126:
        return sign|mantissa  # Can round up into the smallest normal number.
    if mantissa==(1<<24):mantissa>>=1;exponent+=1
    if exponent>127:raise OverflowError('Fixture overflow is unsupported')
    return sign|((exponent+127)<<23)|(mantissa-(1<<23))


def exact_value(bits):
    sign=-1 if bits>>31 else 1
    exponent=(bits>>23)&255;mantissa=bits&0x7fffff
    if exponent==255:raise ValueError('Finite fixtures only')
    if exponent:mantissa|=1<<23;power=exponent-150
    else:power=-149
    return sign*Fraction(mantissa)*(Fraction(2)**power)


def as_float(bits):return struct.unpack('<f',struct.pack('<I',bits))[0]
def add(a,b):return round_binary32(exact_value(a)+exact_value(b))
def multiply(a,b):return round_binary32(exact_value(a)*exact_value(b))


def projection_fixture(checkpoint,base,rows,width,batch):
    """Bounded diagnostic projection only, no full-model host execution."""
    if not 1<=rows<=7 or not 8<=width<=24 or width%8 or batch not in (1,4,8):raise ValueError('Oracle fixture exceeds tiny bounds')
    lanes=4 if batch==8 else 8
    activations=[[round_binary32(Fraction.from_float(((j*7+t*11)%61-30)/17.0))for t in range(batch)]for j in range(width)]
    with open(checkpoint,'rb')as stream:
        stream.seek(base*2);weights=struct.unpack('<'+'H'*(rows*width),stream.read(rows*width*2))
    outputs=[]
    for row in range(rows):
        partial=[[0]*batch for _ in range(lanes)]
        for column in range(width):
            weight=weights[row*width+column]<<16
            for t in range(batch):partial[column%lanes][t]=add(partial[column%lanes][t],multiply(activations[column][t],weight))
        for t in range(batch):
            total=0
            for lane in range(lanes):total=add(total,partial[lane][t])
            outputs.append(as_float(total))
    return outputs
