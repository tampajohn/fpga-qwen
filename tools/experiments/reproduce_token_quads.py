"""Build the archived candidate in a checkout where its patch was applied."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--compiler-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    if 'FP32_TOKEN_QUADS' not in (checkout / 'rtl/fp32_projection_kernel.sv').read_text():
        parser.error('Apply fp32-token-quads.patch to the isolated reproduction checkout first')
    if args.output.exists():
        parser.error('Use a new output directory to preserve existing artifacts')
    sys.path.insert(0, str(checkout / 'tools'))
    from probe_fp32_projection import build
    build(args.output.resolve(), 8, 'fp32', args.compiler_root.resolve(),
          unroll_count=1, token_quads=True)


if __name__ == '__main__':
    main()
