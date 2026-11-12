"""Independent finite-rational and exceptional classification checks for FMA."""
import unittest
from tools.fp32_rounding_oracle import fused_multiply_add as fma,add,multiply

class FmaOracleTests(unittest.TestCase):
    def test_fused_differs_from_separate(self):
        a,b,c=0x3f800001,0x3f7ffffe,0xbf800000
        self.assertEqual(fma(a,b,c),0xa8800000)
        self.assertEqual(add(multiply(a,b),c),0)

    def test_ties_subnormal_and_cancellation(self):
        cases=[(0x3f800000,0x3f800000,0x33800000,0x3f800000),
               (0x3f800001,0x3f800000,0x33800000,0x3f800002),
               (1,0x3f000000,0,0),(1,0x3f000000,1,2),
               (0x80000001,0x3f000000,0,0x80000000),
               (0x3f800000,0x3f800000,0xbf800000,0)]
        for a,b,c,expected in cases:
            with self.subTest(a=a,b=b,c=c):self.assertEqual(fma(a,b,c),expected)

    def test_overflow_rescued_before_rounding(self):
        self.assertEqual(fma(0x7f7fffff,0x40000000,0xff7fffff),0x7f7fffff)
        self.assertEqual(fma(0x7f7fffff,0x40000000,0),0x7f800000)
        self.assertEqual(fma(0xff7fffff,0x40000000,0),0xff800000)

    def test_signed_zero(self):
        for a,b,c,result in [(0x80000000,0x3f800000,0x80000000,0x80000000),
                              (0x80000000,0xbf800000,0x80000000,0),
                              (0x80000000,0x3f800000,0,0)]:
            self.assertEqual(fma(a,b,c),result)

    def test_nan_infinity_classification(self):
        for a,b,c in [(0x7f800000,0,0),(0x7f800000,0x3f800000,0xff800000),
                      (0x7fc12345,0x3f800000,0),(0x7f812345,0x3f800000,0)]:
            self.assertGreater(fma(a,b,c)&0x7fffffff,0x7f800000)
        self.assertEqual(fma(0x7f800000,0xbf800000,0),0xff800000)
        self.assertEqual(fma(0x3f800000,0x3f800000,0x7f800000),0x7f800000)

    def test_invalid_bit_patterns(self):
        for invalid in (-1,1<<32,True,1.0):
            with self.assertRaises(ValueError):fma(invalid,0,0)
