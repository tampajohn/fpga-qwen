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
    parser.add_argument('--unsigned-indices', action='store_true')
    parser.add_argument('--cpu', choices=['generic', 'native'], default='generic')
    parser.add_argument('--fast-attention', action='store_true')
    parser.add_argument('--parallel-attention', action='store_true')
    parser.add_argument('--attention-dot-lanes8', action='store_true')
    parser.add_argument('--batched-prefill', action='store_true')
    parser.add_argument('--block-unroll', action='store_true')
    parser.add_argument('--compact-projection', action='store_true')
    parser.add_argument('--prefill-b8', action='store_true')
    parser.add_argument('--group-projections', action='store_true')
    parser.add_argument('--static-kv', action='store_true')
    parser.add_argument('--pool', choices=['cv', 'hybrid'], default='cv')
    parser.add_argument('--scalar-group', action='store_true')
    parser.add_argument('--tag', default='', help='Extra alphanumeric/dash suffix for a reproducible experiment')
    parser.add_argument('--print-path', action='store_true', help='Print output path without compiling')
    args = parser.parse_args()
    if args.scalar_group and not args.group_projections:parser.error('--scalar-group requires --group-projections')
    if args.tag and not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        parser.error('--tag accepts only letters, numbers, underscores, and dashes')
    if args.attention_dot_lanes8 and not args.parallel_attention:
        parser.error('--attention-dot-lanes8 requires --parallel-attention')
    if args.prefill_b8 and not args.batched_prefill:
        parser.error('--prefill-b8 requires --batched-prefill')
    if args.group_projections and not args.batched_prefill:
        parser.error('--group-projections requires --batched-prefill')
    if args.block_unroll and not args.batched_prefill:
        parser.error('--block-unroll requires --batched-prefill')
    if args.batched_prefill and (not args.parallel_attention or args.storage!='scalar'):
        parser.error('--batched-prefill requires --parallel-attention and scalar storage')
    if args.unchecked and args.storage != 'scalar':
        parser.error('--unchecked applies only to scalar projection reads')
    tag = f'{args.storage}-{args.opt}' + ('-lto' if args.lto else '') + ('-unchecked' if args.unchecked else '')
    if args.fp_contract != 'on': tag += '-contract-'+args.fp_contract
    if args.fast_math: tag += '-fast-math'
    if args.unroll: tag += '-unroll'
    if args.masked_activation: tag += '-masked-activation'
    if args.unsigned_indices: tag += '-unsigned-indices'
    if args.cpu == 'native': tag += '-cpu-native'
    if args.fast_attention: tag += '-fast-attention'
    if args.parallel_attention: tag += '-parallel-attention'
    if args.attention_dot_lanes8: tag += '-attention-dot-lanes8'
    if args.batched_prefill: tag += '-batched-prefill'
    if args.block_unroll: tag += '-block-unroll'
    if args.compact_projection: tag += '-compact-projection'
    if args.group_projections: tag += '-group-projections'
    if args.static_kv: tag += '-static-kv'
    if args.prefill_b8: tag += '-prefill-b8'
    if args.pool != 'cv': tag += '-pool-'+args.pool
    if args.scalar_group: tag += '-scalar-group'
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
    if args.scalar_group: flags += ' -DSCALAR_GROUP'
    if args.cpu == 'native': flags += ' -mcpu=native'
    if args.fast_math: flags += ' -ffast-math'
    if args.fp_contract != 'on': flags += ' -ffp-contract='+args.fp_contract
    if args.unroll: flags += ' -funroll-loops'
    if args.pool == 'hybrid': flags += ' -DPROJECTION_HYBRID_WAIT'
    if args.parallel_attention: flags += ' -DPARALLEL_ATTENTION'
    if args.batched_prefill: flags += ' -DBATCH_PREFILL'
    if args.group_projections: flags += ' -DGROUP_PROJECTIONS'
    if args.static_kv: flags += ' -DSTATIC_KV'
    if args.prefill_b8: flags += ' -DPREFILL_B8 -DWIDE_BATCH=8 -DWIDE_FOUR_LANES'
    defines = {'scalar': [], 'row': ['+define+BULK_ROW_WEIGHTS']}[args.storage]
    if args.attention_dot_lanes8: defines += ['+define+ATTENTION_DOT_LANES8']
    if args.unchecked: defines += ['+define+UNCHECKED_WEIGHTS']
    if args.masked_activation: defines += ['+define+MASK_ACTIVATION_INDEX']
    if args.block_unroll: defines += ['+define+BLOCK_UNROLL']
    if args.unsigned_indices: defines += ['+define+UNSIGNED_PROJECTION_INDEX']
    main_defines = []
    if args.scalar_group: main_defines += ['+define+SCALAR_GROUP']
    if args.parallel_attention:
        defines += ['+define+PARALLEL_ATTENTION']
        main_defines += ['+define+PARALLEL_ATTENTION']
    if args.batched_prefill:
        defines += ['+define+BATCH_PREFILL']
        main_defines += ['+define+BATCH_PREFILL']
    if args.fast_attention: main_defines += ['+define+FAST_ATTENTION']
    if args.compact_projection: main_defines += ['+define+COMPACT_PROJECTION_BUFFER']
    if args.group_projections: main_defines += ['+define+GROUP_PROJECTIONS']
    if args.static_kv: main_defines += ['+define+STATIC_KV']
    if args.prefill_b8:
        defines += ['+define+WIDE_BATCH=8', '+define+WIDE_FOUR_LANES']
        main_defines += ['+define+PREFILL_B8']
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
               'sim/full_model.cpp', 'sim/projection_pool.h', 'sim/projection_descriptors.h', 'tools/build_projection_variant.py']
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
