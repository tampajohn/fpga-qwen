"""Build isolated projection variants without touching the live simulator binary."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', choices=['scalar', 'row'], default='scalar')
    parser.add_argument('--lto', action='store_true')
    parser.add_argument('--unchecked', action='store_true', help='Validate each full projection span once, then omit per-weight checks')
    parser.add_argument('--opt', choices=['O3', 'Os'], default='O3')
    args = parser.parse_args()
    tag = f'{args.storage}-{args.opt}' + ('-lto' if args.lto else '') + ('-unchecked' if args.unchecked else '')
    directory = ROOT / 'build/variants' / tag
    projection, full = directory / 'projection', directory / 'full'
    projection.mkdir(parents=True, exist_ok=True)
    full.mkdir(parents=True, exist_ok=True)
    flags = '-std=c++17 -' + args.opt + (' -flto=thin' if args.lto else '')
    defines = {'scalar': [], 'row': ['+define+BULK_ROW_WEIGHTS']}[args.storage]
    if args.unchecked:
        defines += ['+define+UNCHECKED_WEIGHTS']
    with (directory/'build.log').open('w') as log:
        subprocess.run(['verilator', '--cc', '--build', '-j', '4', '--top-module',
                        'projection_worker', '--Mdir', str(projection), '-Wall', '-Wno-BLKSEQ',
                        '-CFLAGS', flags, '-MAKEFLAGS', f'OPT_FAST=-{args.opt}', '-O3',
                        *defines, str(ROOT/'rtl/projection_worker.sv')],
                       check=True, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        subprocess.run(['verilator', '--cc', '--exe', '--build', '-j', '4', '--top-module',
                        'qwen_behavioral', '--Mdir', str(full), '-I'+str(ROOT/'rtl'),
                        '-Wno-fatal', '--public-flat-rw', '-CFLAGS', flags+' -I'+str(projection),
                        '-MAKEFLAGS', f'OPT_FAST=-{args.opt}', '-O3',
                        *(['-LDFLAGS', '-flto=thin'] if args.lto else []),
                        str(ROOT/'rtl/qwen_behavioral.sv'), str(ROOT/'sim/full_model.cpp'),
                        str(projection/'Vprojection_worker__ALL.a')],
                       check=True, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    print(full/'Vqwen_behavioral', flush=True)


if __name__ == '__main__':
    main()
