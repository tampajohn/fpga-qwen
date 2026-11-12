"""Build isolated, explicitly tagged SV variants without touching the live binary."""
import argparse
import hashlib
import json
import os
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
    parser.add_argument('--batch-attention', action='store_true', help='Experimental exact per-query batched attention dispatch')
    parser.add_argument('--attention-dot-lanes8', action='store_true')
    parser.add_argument('--batched-prefill', action='store_true')
    parser.add_argument('--block-unroll', action='store_true')
    parser.add_argument('--compact-projection', action='store_true')
    parser.add_argument('--prefill-b8', action='store_true')
    parser.add_argument('--group-projections', action='store_true')
    parser.add_argument('--static-kv', action='store_true')
    parser.add_argument('--pool', choices=['cv', 'hybrid'], default='cv')
    parser.add_argument('--scalar-group', action='store_true')
    parser.add_argument('--fp32-projection-b1', action='store_true', help='Experimental FP32 scalar projections with FP64 prefill/attention; no prefix reuse or block verification')
    parser.add_argument('--fp32-verification-b4', action='store_true', help='Experimental explicit FP32 B4 verification; FP64 prefill unchanged')
    parser.add_argument('--fp32-prefill-b8', action='store_true', help='Experimental FP32 B8 prefill and B4 tails; distinct precision profile')
    parser.add_argument('--shortreal-compiler', default='', help='Reviewed patched Verilator root for isolated FP32 archive')
    parser.add_argument('--tag', default='', help='Extra alphanumeric/dash suffix for a reproducible experiment')
    parser.add_argument('--print-path', action='store_true', help='Print output path without compiling')
    args = parser.parse_args()
    if args.fp32_prefill_b8 and not args.fp32_verification_b4:parser.error('--fp32-prefill-b8 requires --fp32-verification-b4')
    if args.fp32_verification_b4 and not args.fp32_projection_b1:parser.error('--fp32-verification-b4 requires --fp32-projection-b1')
    if args.fp32_projection_b1:
        if not (args.lto and args.parallel_attention and args.batched_prefill and args.prefill_b8 and args.scalar_group and args.storage=='scalar' and args.opt=='O3'):
            parser.error('FP32 B1 requires scalar O3/LTO, parallel attention, B8 prefill and scalar grouping')
        if args.fast_math or args.fp_contract!='on':parser.error('FP32 B1 retains original stock FP64 contraction flags')
        if not args.shortreal_compiler and not args.print_path:parser.error('FP32 B1 needs explicit reviewed --shortreal-compiler')
    if args.scalar_group and not args.group_projections:parser.error('--scalar-group requires --group-projections')
    if args.tag and not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        parser.error('--tag accepts only letters, numbers, underscores, and dashes')
    if args.batch_attention and not (args.parallel_attention and args.prefill_b8):parser.error('--batch-attention requires parallel attention and B8 prefill')
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
    if args.fp32_projection_b1: tag += '-fp32-b1'
    if args.fp32_verification_b4: tag += '-verify-b4'
    if args.fp32_prefill_b8: tag += '-fp32-prefill-b8'
    if args.batch_attention: tag += '-batch-attention'
    if args.tag: tag += '-'+args.tag
    directory = ROOT / 'build/variants' / tag
    projection, full = directory / 'projection', directory / 'full'
    fp32 = directory / 'fp32'
    fp32_b4 = directory / 'fp32-b4'
    fp32_b8 = directory / 'fp32-b8'
    binary = full/'Vqwen_behavioral'
    if args.print_path:
        print(binary)
        return
    projection.mkdir(parents=True, exist_ok=True)
    full.mkdir(parents=True, exist_ok=True)
    flags = '-std=c++17 -' + args.opt + (' -flto=thin' if args.lto else '')
    if args.scalar_group: flags += ' -DSCALAR_GROUP'
    if args.fp32_projection_b1: flags += ' -DFP32_PROJECTION_B1 -I'+str(fp32)
    if args.fp32_prefill_b8: flags += ' -DFP32_PREFILL_B8 -I'+str(fp32_b8)
    if args.fp32_verification_b4: flags += ' -DFP32_VERIFICATION_B4 -I'+str(fp32_b4)
    if args.cpu == 'native': flags += ' -mcpu=native'
    if args.fast_math: flags += ' -ffast-math'
    if args.fp_contract != 'on': flags += ' -ffp-contract='+args.fp_contract
    if args.unroll: flags += ' -funroll-loops'
    if args.pool == 'hybrid': flags += ' -DPROJECTION_HYBRID_WAIT'
    if args.batch_attention: flags += ' -DBATCH_ATTENTION'
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
    if args.batch_attention:
        defines += ['+define+BATCH_ATTENTION']
        main_defines += ['+define+BATCH_ATTENTION']
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
         str(projection/'Vprojection_worker__ALL.a'),
         *([str(fp32/'Vfp32_projection_kernel__ALL.a')]if args.fp32_projection_b1 else []),
         *([str(fp32_b4/'Vfp32_verification_kernel__ALL.a')] if args.fp32_verification_b4 else []),
         *([str(fp32_b8/'Vfp32_prefill_kernel__ALL.a')] if args.fp32_prefill_b8 else [])]]
    stock_env=os.environ.copy()
    stock_env.pop('VERILATOR_ROOT',None);stock_env.pop('VERILATOR_BIN',None)
    environments=[stock_env for _ in commands]
    fp32_info=None
    if args.fp32_projection_b1:
        compiler_root=Path(args.shortreal_compiler).resolve();compiler_binary=compiler_root/'build-local/src/verilator_bin'
        patched_env=stock_env.copy();patched_env['VERILATOR_ROOT']=str(compiler_root);patched_env['VERILATOR_BIN']=str(compiler_binary)
        fp32.mkdir(parents=True,exist_ok=True)
        command=[str(compiler_root/'bin/verilator'),'--cc','--build','-j','4','--top-module','fp32_projection_kernel','--Mdir',str(fp32),
            '-Wall','-Wno-BLKSEQ','-Wno-fatal','-CFLAGS','-std=c++17 -O3 -flto=thin -fno-fast-math -ffp-contract=off',
            '-MAKEFLAGS','OPT_FAST=-O3','-O3','+define+KERNEL_BATCH=1','+define+KERNEL_ROWS=151936',str(ROOT/'rtl/fp32_projection_kernel.sv')]
        commands.insert(0,command);environments.insert(0,patched_env)
        if args.fp32_verification_b4:
            fp32_b4.mkdir(parents=True,exist_ok=True)
            verify_command=command.copy()
            verify_command[verify_command.index('--Mdir')+1]=str(fp32_b4)
            verify_command[verify_command.index('+define+KERNEL_BATCH=1')]='+define+KERNEL_BATCH=4'
            verify_command[verify_command.index('+define+KERNEL_ROWS=151936')]='+define+KERNEL_ROWS=3072'
            verify_command[1:1]=['--prefix','Vfp32_verification_kernel']
            commands.insert(1,verify_command);environments.insert(1,patched_env)
        if args.fp32_prefill_b8:
            fp32_b8.mkdir(parents=True,exist_ok=True)
            prefill_command=command.copy()
            prefill_command[prefill_command.index('--Mdir')+1]=str(fp32_b8)
            prefill_command[prefill_command.index('+define+KERNEL_BATCH=1')]='+define+KERNEL_BATCH=8'
            prefill_command[prefill_command.index('+define+KERNEL_ROWS=151936')]='+define+KERNEL_ROWS=3072'
            prefill_command[1:1]=['--prefix','Vfp32_prefill_kernel','--unroll-count','1']
            commands.insert(2,prefill_command);environments.insert(2,patched_env)
        stock_include=Path(subprocess.check_output(['verilator','--getenv','VERILATOR_ROOT'],env=stock_env,text=True).strip())/'include'
        headers={}
        for name in ('verilated.h','verilated_types.h','verilated_threads.h'):
            a=hashlib.sha256((stock_include/name).read_bytes()).hexdigest();b=hashlib.sha256((compiler_root/'include'/name).read_bytes()).hexdigest()
            if a!=b:raise RuntimeError('Runtime ABI header differs: '+name)
            headers[name]=a
        fp32_info=dict(compiler_binary_sha256=hashlib.sha256(compiler_binary.read_bytes()).hexdigest(),
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=compiler_root,text=True).strip(),runtime_header_hashes=headers,
            policy='FP32 archives only use strict flags; stock worker/main/runtime retain original O3 ThinLTO/contraction. B1 FP32; attention/KV FP64. Prefill precision and verification capacity are specified by the adjacent manifest fields.', verification_batch=4 if args.fp32_verification_b4 else 1, prefill_projection='fp32' if args.fp32_prefill_b8 else 'fp64', prefill_unroll_count=1 if args.fp32_prefill_b8 else None)
    sources = ['rtl/projection_worker.sv', 'rtl/qwen_behavioral.sv', 'rtl/sampling.svh',
               'sim/full_model.cpp', 'sim/projection_pool.h', 'sim/projection_descriptors.h', 'tools/build_projection_variant.py']
    if args.fp32_projection_b1:sources.append('rtl/fp32_projection_kernel.sv')
    def source_hashes():
        return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    before = source_hashes()
    with (directory/'build.log').open('w') as log:
        for command,environment in zip(commands,environments):
            subprocess.run(command,env=environment,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    after = source_hashes()
    if before != after:
        raise RuntimeError('Source changed during build; rebuild before benchmarking this artifact')
    if fp32_info and fp32_info['compiler_binary_sha256']!=hashlib.sha256(compiler_binary.read_bytes()).hexdigest():raise RuntimeError('FP32 compiler changed during build')
    manifest = dict(fp32=fp32_info,options=vars(args), commands=commands, sources=after,
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    verilator=subprocess.check_output(['verilator', '--version'],env=stock_env,text=True).strip(),
                    compiler=subprocess.check_output(['c++', '--version'], text=True).splitlines()[0])
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(binary, flush=True)


if __name__ == '__main__':
    main()
