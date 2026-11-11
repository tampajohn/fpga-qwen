"""Build isolated, explicitly tagged SV variants without touching the live binary."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', choices=['scalar', 'row'], default='scalar')
    parser.add_argument('--lto', action='store_true')
    parser.add_argument('--unchecked', action='store_true', help='Validate each full projection span once, then omit per-weight checks')
    parser.add_argument('--opt', choices=['O3', 'Os'], default='O3')
    parser.add_argument('--fp-contract', choices=['on', 'off', 'fast'], default='on')
    parser.add_argument('--fast-math', action='store_true', help='Experimental relaxed IEEE semantics; requires separate quality validation')
    parser.add_argument('--unroll', action='store_true')
    parser.add_argument('--masked-activation', action='store_true')
    parser.add_argument('--fast-attention', action='store_true')
    parser.add_argument('--parallel-attention', action='store_true')
    parser.add_argument('--compact-projection', action='store_true')
    parser.add_argument('--pool', choices=['cv', 'hybrid'], default='cv')
    parser.add_argument('--tag', default='', help='Extra alphanumeric/dash suffix for a reproducible experiment')
    parser.add_argument('--print-path', action='store_true', help='Print output path without compiling')
    args = parser.parse_args()
    if args.tag and not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        parser.error('--tag accepts only letters, numbers, underscores, and dashes')
    if args.unchecked and args.storage != 'scalar':
        parser.error('--unchecked applies only to scalar projection reads')
    tag = f'{args.storage}-{args.opt}' + ('-lto' if args.lto else '') + ('-unchecked' if args.unchecked else '')
    if args.fp_contract != 'on': tag += '-contract-'+args.fp_contract
    if args.fast_math: tag += '-fast-math'
    if args.unroll: tag += '-unroll'
    if args.masked_activation: tag += '-masked-activation'
    if args.fast_attention: tag += '-fast-attention'
    if args.parallel_attention: tag += '-parallel-attention'
    if args.compact_projection: tag += '-compact-projection'
    if args.pool != 'cv': tag += '-pool-'+args.pool
    if args.tag: tag += '-'+args.tag
    directory = ROOT / 'build/variants' / tag
    projection, full = directory / 'projection', directory / 'full'
    binary = full/'Vqwen_behavioral'
    if args.print_path:
        print(binary)
        return
    projection.mkdir(parents=True, exist_ok=True)
    full.mkdir(parents=True, exist_ok=True)
    flags = '-std=c++17 -' + args.opt + (' -flto=thin' if args.lto else '')
    if args.fast_math: flags += ' -ffast-math'
    if args.fp_contract != 'on': flags += ' -ffp-contract='+args.fp_contract
    if args.unroll: flags += ' -funroll-loops'
    if args.pool == 'hybrid': flags += ' -DPROJECTION_HYBRID_WAIT'
    if args.parallel_attention: flags += ' -DPARALLEL_ATTENTION'
    defines = {'scalar': [], 'row': ['+define+BULK_ROW_WEIGHTS']}[args.storage]
    if args.unchecked: defines += ['+define+UNCHECKED_WEIGHTS']
    if args.masked_activation: defines += ['+define+MASK_ACTIVATION_INDEX']
    main_defines = []
    if args.parallel_attention:
        defines += ['+define+PARALLEL_ATTENTION']
        main_defines += ['+define+PARALLEL_ATTENTION']
    if args.fast_attention: main_defines += ['+define+FAST_ATTENTION']
    if args.compact_projection: main_defines += ['+define+COMPACT_PROJECTION_BUFFER']
    commands = [
        ['verilator', '--cc', '--build', '-j', '4', '--top-module',
         'projection_worker', '--Mdir', str(projection), '-Wall', '-Wno-BLKSEQ',
         '-CFLAGS', flags, '-MAKEFLAGS', f'OPT_FAST=-{args.opt}', '-O3',
         *defines, str(ROOT/'rtl/projection_worker.sv')],
        ['verilator', '--cc', '--exe', '--build', '-j', '4', '--top-module',
         'qwen_behavioral', '--Mdir', str(full), '-I'+str(ROOT/'rtl'),
         '-Wno-fatal', '--public-flat-rw', '-CFLAGS', flags+' -I'+str(projection),
         '-MAKEFLAGS', f'OPT_FAST=-{args.opt}', '-O3', *main_defines,
         *(['-LDFLAGS', '-flto=thin'] if args.lto else []),
         str(ROOT/'rtl/qwen_behavioral.sv'), str(ROOT/'sim/full_model.cpp'),
         str(projection/'Vprojection_worker__ALL.a')]]
    sources = ['rtl/projection_worker.sv', 'rtl/qwen_behavioral.sv', 'rtl/sampling.svh',
               'sim/full_model.cpp', 'sim/projection_pool.h', 'tools/build_projection_variant.py']
    def source_hashes():
        return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    before = source_hashes()
    with (directory/'build.log').open('w') as log:
        for command in commands:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    after = source_hashes()
    if before != after:
        raise RuntimeError('Source changed during build; rebuild before benchmarking this artifact')
    manifest = dict(options=vars(args), commands=commands, sources=after,
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    verilator=subprocess.check_output(['verilator', '--version'], text=True).strip(),
                    compiler=subprocess.check_output(['c++', '--version'], text=True).splitlines()[0])
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(binary, flush=True)


if __name__ == '__main__':
    main()
