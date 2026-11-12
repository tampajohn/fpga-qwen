#!/usr/bin/env python3
"""Guarded direct-API B2 verification correctness gate; no timing claims.

Builds a frozen copy of the original core with a separate C++ test entry point.
Default acceptance matrix: prefixes0/1/511/512/513/3966 × workers1/12/64 ×
commit1/2. --smoke is explicitly partial and cannot satisfy acceptance.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from benchmark_native_phase0 import guard, sha

ROOT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def save(path, report):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=ROOT/'build/native-cpu/weights.bin')
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--source-directory', type=Path, default=ROOT/'engine',
                        help='Core source directory to freeze and test; defaults to engine')
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT/'build/benchmark-status')
    args = parser.parse_args()
    guarded = guard(args.guard_state_dir)
    require(sys.platform == 'darwin' and platform.machine() == 'arm64', 'Apple ARM is required')
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    output = directory/'report.json'
    workers = [1] if args.smoke else [1, 12, 64]
    prefixes = [0, 1] if args.smoke else [0, 1, 511, 512, 513, 3966]
    report = dict(completed=False, passed=False, full_acceptance_matrix_passed=False,
                  smoke=args.smoke, gpu=False, pytorch=False, performance_claim_eligible=False,
                  guard_before=guarded, workers=workers, prefill_workers=6,
                  prefixes=prefixes, commits=[1, 2], runs=[],
                  prefix_token_formula='32 + (index * 37) % 1000', verification_ids=[9707, 11],
                  ordinary_continuation_ids=[42, 99],
                  scope='Direct core API: both full-vocabulary logits and normalized hidden rows must match two sequential steps bit-for-bit. Commit rollback is checked with different ordinary continuation IDs.',
                  limitations=['Synthetic predetermined valid-token prefixes, not conversation-quality fixtures.',
                               'Validation errors must preserve state/pending views. Reset and truncate must abort pending work.',
                               'Execution-error reset contract is source-reviewed separately; no OOM, hardware fault, or failure injection is attempted.',
                               'No CLI sampler, browser integration, drafting acceptance rate, or performance claim.'])
    try:
        weights = args.weights.resolve()
        paths = [weights, Path(str(weights)+'.index'), Path(str(weights)+'.manifest.json')]
        report['inputs'] = {str(path): sha(path) for path in paths}
        manifest = json.loads(paths[2].read_text())
        require(manifest.get('completed') is True and manifest.get('format') == 'QWCPU01'
                and manifest['binary']['sha256'] == report['inputs'][str(weights)]
                and manifest['index']['sha256'] == report['inputs'][str(paths[1])], 'Weight manifest mismatch')
        source_directory = args.source_directory.resolve()
        files = [source_directory/'core.cpp', source_directory/'core.hpp', ROOT/'tools/native_verify_check.cpp']
        require(all(path.is_file() for path in files), 'Missing selected core source or API test harness')
        report['source_directory'] = str(source_directory)
        report['source_scope'] = 'Selected core.cpp/core.hpp frozen beside the unchanged API test harness; this gate builds its own executable, not the chat binary.'
        helpers = [Path(__file__).resolve(), ROOT/'tools/benchmark_native_phase0.py',
                   ROOT/'tools/benchmark_guard.py']
        report['source_hashes'] = {str(path): sha(path) for path in files+helpers}
        frozen = directory/'source'
        frozen.mkdir()
        for path in files:
            shutil.copyfile(path, frozen/path.name)
            require(sha(frozen/path.name) == report['source_hashes'][str(path)], 'Source changed while copying')
        report['frozen_source_hashes'] = {str(frozen/path.name): sha(frozen/path.name) for path in files}
        env = dict(os.environ)
        removed = []
        for name in list(env):
            if name.startswith(('DYLD_', 'GGML_', 'LLAMA_ARG_')) or name == 'VECLIB_MAXIMUM_THREADS':
                removed.append(name)
                env.pop(name)
        env['VECLIB_MAXIMUM_THREADS'] = '6'
        report.update(removed_environment_names=sorted(removed), requested_veclib_maximum_threads=6,
                      thread_scope='Core decode workers explicit; prefill workers6. Accelerate thread request is not a pinning guarantee.')
        compiler = shutil.which('clang++')
        require(compiler is not None, 'clang++ is required')
        binary = directory/'native_verify_check'
        command = [compiler, '-std=c++20', '-O3', '-mcpu=native', '-flto=thin',
                   '-ffp-contract=fast', '-fno-math-errno', '-pthread',
                   str(frozen/'core.cpp'), str(frozen/'native_verify_check.cpp'),
                   '-framework', 'Accelerate', '-o', str(binary)]
        report['build'] = dict(command=command, compiler=subprocess.check_output([compiler, '--version'], text=True, env=env))
        save(output, report)
        with (directory/'build.log').open('x') as log:
            built = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=180)
        report['build']['returncode'] = built.returncode
        require(built.returncode == 0, 'Compilation failed; inspect build.log')
        linked = subprocess.check_output(['otool', '-L', str(binary)], text=True, env=env)
        require(not any(name in '\n'.join(linked.splitlines()[1:]).lower()
                        for name in ('metal', 'cuda', 'torch', 'llama', 'ggml')), 'Unexpected neural/GPU runtime linkage')
        report['build'].update(binary=str(binary), binary_sha256=sha(binary), linked_libraries=linked)
        report['inputs'][str(binary)] = sha(binary)
        save(output, report)
        for worker in workers:
            guard(args.guard_state_dir)
            stdout, stderr = directory/f'workers-{worker}.log', directory/f'workers-{worker}.stderr.log'
            command = [str(binary), str(weights), str(worker)] + (['--smoke'] if args.smoke else [])
            row = dict(workers=worker, command=command, completed=False)
            report['runs'].append(row)
            save(output, report)
            with stdout.open('x') as out, stderr.open('x') as err:
                process = subprocess.run(command, stdout=out, stderr=err, env=env, timeout=3600)
            row.update(returncode=process.returncode, stdout_sha256=sha(stdout), stderr_sha256=sha(stderr))
            require(process.returncode == 0, f'API gate failed for workers{worker}; inspect logs')
            result = json.loads(stdout.read_text().splitlines()[-1])
            require(result.get('completed') is True and result.get('passed') is True
                    and result.get('workers') == worker and result.get('prefill_workers') == 6
                    and result.get('prefix_commit_cases') == len(prefixes)*2
                    and result.get('edge_cases_passed') is True, 'Incomplete API gate result')
            row.update(completed=True, result=result)
            save(output, report)
        for group in ('inputs', 'source_hashes', 'frozen_source_hashes'):
            require(all(sha(Path(path)) == digest for path, digest in report[group].items()), 'Frozen artifact changed: '+group)
        report.update(completed=True, passed=True, full_acceptance_matrix_passed=not args.smoke,
                      guard_after=guard(args.guard_state_dir))
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        save(output, report)
    print(output)


if __name__ == '__main__':
    main()
