import unittest
from fractions import Fraction
from tools.fp32_rounding_oracle import round_binary32,exact_value,add,multiply

class RoundingOracleTests(unittest.TestCase):
    def test_ties_even_and_sign(self):
        self.assertEqual(round_binary32(1+Fraction(1,1<<24)),0x3f800000)
        self.assertEqual(round_binary32(1+Fraction(3,1<<24)),0x3f800002)
        self.assertEqual(round_binary32(-1-Fraction(3,1<<24)),0xbf800002)
    def test_subnormal_boundary(self):
        self.assertEqual(round_binary32(Fraction(1,1<<149)),1)
        self.assertEqual(round_binary32(Fraction(1,1<<150)),0)
        self.assertEqual(round_binary32(Fraction(3,1<<150)),2)
        self.assertEqual(round_binary32(Fraction(1,1<<126)),0x00800000)
    def test_known_rounded_operations(self):
        self.assertEqual(add(0x3f800000,0xbf800000),0)
        self.assertEqual(multiply(0x3f800001,0x3f800001),0x3f800002)
        self.assertEqual(add(0x4b800000,0x3f800000),0x4b800000) # 2**24 +1 rounds down.
        self.assertEqual(add(0x4b800000,0x40400000),0x4b800002) # 2**24 +3 rounds up.
    def test_exact_roundtrip(self):
        for bits in (1,0x007fffff,0x00800000,0x3f800001,0xbf800001,0x7f7fffff):
            self.assertEqual(round_binary32(exact_value(bits)),bits)
