"""Build/run the original public-API native CPU profiling client.

This is a sparse-callback perturbation diagnostic, not a scoreboard benchmark.
Build and run are deliberately separate. Run must be a direct child of the
serving checkout's benchmark_guard.py and name that guard's state directory.
Prompt/continuation files are JSON integer arrays. Continuation contains exactly
tokens-1 consumed IDs; with it the client teacher-forces instead of generating.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from benchmark_guard import BenchmarkGate

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/profile_native_cpu.cpp'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cpu_build(build):
    cache = build / 'CMakeCache.txt'
    values = {}
    for line in cache.read_text().splitlines():
        if line and not line.startswith(('#', '//')) and '=' in line and ':' in line.split('=', 1)[0]:
            key, value = line.split('=', 1)
            values[key.split(':', 1)[0]] = value
    for name in ('METAL', 'CUDA', 'VULKAN', 'OPENCL', 'HIP', 'MUSA', 'SYCL'):
        if values.get('GGML_' + name) != 'OFF':
            raise RuntimeError('CPU-only build requires GGML_' + name + '=OFF')
    for name in ('CANN', 'WEBGPU', 'HEXAGON', 'VIRTGPU', 'RPC', 'OPENVINO'):
        if values.get('GGML_' + name, 'OFF') != 'OFF':
            raise RuntimeError('Unexpected non-CPU backend: ' + name)
    if values.get('BUILD_SHARED_LIBS') != 'ON' or values.get('GGML_BACKEND_DL') != 'OFF':
        raise RuntimeError('Client helper requires shared libraries and GGML_BACKEND_DL=OFF')
    if values.get('GGML_CPU') != 'ON':
        raise RuntimeError('CPU backend must be enabled')
    return dict(path=str(cache), sha256=sha(cache), options=values)


def native_libraries(build):
    paths = sorted({p.resolve() for pattern in ('libllama*.dylib', 'libggml*.dylib')
                    for p in (build / 'bin').glob(pattern)})
    if not paths or not any(p.name.startswith('libllama') for p in paths):
        raise RuntimeError('No built llama/ggml shared libraries found')
    return {str(path): sha(path) for path in paths}


def build_client(args):
    source_tree = args.llama_source.resolve()
    build = args.llama_build.resolve()
    directory = args.directory.resolve()
    if directory.exists():
        raise RuntimeError('Use a new client build directory; artifacts are immutable')
    cache = cpu_build(build)
    if Path(cache['options']['CMAKE_HOME_DIRECTORY']).resolve() != source_tree:
        raise RuntimeError('Headers/source checkout differs from the native build source directory')
    libraries = native_libraries(build)
    compiler = shutil.which(args.cxx)
    if not compiler:
        raise RuntimeError('C++ compiler not found')
    inputs = [SOURCE, Path(__file__).resolve(), source_tree / 'include/llama.h',
              source_tree / 'ggml/include/ggml.h', source_tree / 'ggml/include/ggml-backend.h',
              source_tree / 'ggml/include/ggml-cpu.h',
              source_tree / 'vendor/nlohmann/json.hpp']
    hashes = {str(p): sha(p) for p in inputs}
    directory.mkdir(parents=True)
    binary = directory / 'profile_native_cpu'
    command = [compiler, '-std=c++17', '-O3', '-DNDEBUG', '-Wall', '-Wextra',
               '-I' + str(source_tree / 'include'), '-I' + str(source_tree / 'ggml/include'),
               '-I' + str(source_tree / 'vendor'), str(SOURCE), '-L' + str(build / 'bin'),
               '-Wl,-rpath,' + str(build / 'bin'), '-lllama', '-lggml', '-lggml-cpu', '-lggml-base', '-o', str(binary)]
    with (directory / 'build.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    if hashes != {str(p): sha(p) for p in inputs} or libraries != native_libraries(build) or cache != cpu_build(build):
        raise RuntimeError('Build inputs changed during client compilation')
    manifest = dict(binary=str(binary), binary_sha256=sha(binary), command=command,
                    compiler_version=subprocess.check_output([compiler, '--version'], text=True),
                    llama_source=str(source_tree), llama_build=str(build),
                    llama_commit=subprocess.check_output(['git', '-C', str(source_tree), 'rev-parse', 'HEAD'], text=True).strip(),
                    source_hashes=hashes, libraries=libraries, native_build=cache,
                    diagnostic_only=True, source_policy='Original profiling client; linked public llama/ggml APIs, no copied kernels.')
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(binary)


def verify_manifest(binary):
    manifest = json.loads((binary.parent / 'manifest.json').read_text())
    if manifest['binary'] != str(binary) or manifest['binary_sha256'] != sha(binary):
        raise RuntimeError('Profiling client differs from frozen manifest')
    build = Path(manifest['llama_build'])
    if manifest['libraries'] != native_libraries(build) or manifest['native_build'] != cpu_build(build):
        raise RuntimeError('Linked native build differs from client manifest')
    return manifest


def guard(directory):
    directory = directory.resolve()
    status = BenchmarkGate(directory).status()
    metadata = json.loads((directory / 'run.json').read_text())
    if not status['running'] or metadata.get('pid') != os.getppid():
        raise RuntimeError('Run directly under serving-checkout benchmark_guard.py')
    return metadata


def run_client(args):
    guarded = guard(args.guard_state_dir)
    binary = args.binary.resolve()
    manifest = verify_manifest(binary)
    directory = args.directory.resolve()
    if directory.exists():
        raise RuntimeError('Use a new run directory; reports/dumps are immutable')
    directory.mkdir(parents=True)
    model, prompt = args.model.resolve(), args.prompt_ids.resolve()
    input_files = [model, prompt]
    command = [str(binary), '--model', str(model), '--prompt-ids', str(prompt),
               '--threads', str(args.threads), '--batch-threads', str(args.batch_threads),
               '--tokens', str(args.tokens), '--warmups', str(args.warmups),
               '--repetitions', str(args.repetitions), '--callback', args.callback,
               '--flash-attn', args.flash_attn, '--context', str(args.context),
               '--batch-size', str(args.batch_size), '--output', str(directory / 'client.json')]
    if args.continuation_ids:
        continuation = args.continuation_ids.resolve()
        input_files.append(continuation)
        command += ['--continuation-ids', str(continuation)]
    if args.dump_logits:
        command += ['--dump-logits-dir', str(directory / 'logits')]
    inputs = {str(path): sha(path) for path in input_files}
    report = dict(completed=False, diagnostic_only=True, guard=guarded, command=command,
                  inputs=inputs, client_manifest=manifest, helper_sha256=sha(__file__),
                  limits='No stack profiling or cross-mode comparison here. Compare callback-on/off separately; neither is a scoreboard result.')
    environment = os.environ.copy()
    removed = []
    for key in list(environment):
        if key.startswith(('DYLD_', 'GGML_KLEIDIAI_', 'LLAMA_ARG_')) or key in ('GGML_BACKEND_PATH', 'GGML_TOTAL_THREADS'):
            removed.append(key)
            environment.pop(key)
    report['removed_environment_keys'] = sorted(removed)
    report['native_environment'] = {k: v for k, v in environment.items()
                                    if k.startswith(('GGML_', 'OMP_', 'VECLIB_', 'DYLD_'))}
    try:
        with (directory / 'stdout.log').open('w') as stdout, (directory / 'stderr.log').open('w') as stderr:
            process = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr)
        report['exit_code'] = process.returncode
        client_path = directory / 'client.json'
        if client_path.exists():
            report['client_report'] = json.loads(client_path.read_text())
        if process.returncode:
            raise RuntimeError('Profiling client failed; inspect client.json and stderr.log')
        if not report['client_report']['completed']:
            raise RuntimeError('Client report is incomplete')
        if manifest != verify_manifest(binary) or inputs != {str(p): sha(p) for p in input_files}:
            raise RuntimeError('Frozen executable/library/input changed during diagnostic')
        report['guard_after'] = guard(args.guard_state_dir)
        actual_dumps = set((directory / 'logits').glob('*.f32'))
        expected_dumps = set()
        rows = report['client_report']['runs']
        if len(rows) != args.warmups + args.repetitions:
            raise RuntimeError('Wrong diagnostic request count')
        for row in rows:
            if len(row['logit_dumps']) != (args.tokens if args.dump_logits else 0):
                raise RuntimeError('Wrong per-request logit dump count')
            for step, item in enumerate(row['logit_dumps']):
                path = Path(item['path'])
                expected_path = directory / 'logits' / f"run-{row['index']}-step-{step}.f32"
                if (path != expected_path or item['step'] != step or item['dtype'] != 'float32'
                        or item['byte_order'] != 'little' or item['shape'] != [151936]
                        or item['bytes'] != 151936 * 4 or path.stat().st_size != 151936 * 4
                        or item['context_tokens'] != len(report['client_report']['prompt_ids']) + step):
                    raise RuntimeError('Invalid full-logit dump metadata or size')
                expected_dumps.add(path)
        if actual_dumps != expected_dumps:
            raise RuntimeError('Dump files differ from declared outputs')
        report['logit_dumps'] = {str(p): sha(p) for p in sorted(actual_dumps)}
        report['completed'] = True
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(directory / 'report.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    build = modes.add_parser('build')
    build.add_argument('--llama-source', type=Path, required=True)
    build.add_argument('--llama-build', type=Path, required=True)
    build.add_argument('--directory', type=Path, required=True)
    build.add_argument('--cxx', default='clang++')
    run = modes.add_parser('run')
    for name in ('binary', 'model', 'prompt-ids', 'guard-state-dir', 'directory'):
        run.add_argument('--' + name, type=Path, required=True)
    run.add_argument('--continuation-ids', type=Path)
    run.add_argument('--dump-logits', action='store_true')
    for name, default in (('threads', 8), ('batch-threads', 12), ('tokens', 32), ('warmups', 1),
                          ('repetitions', 3), ('context', 4096), ('batch-size', 512)):
        run.add_argument('--' + name, type=int, default=default)
    run.add_argument('--callback', choices=('off', 'on'), required=True)
    run.add_argument('--flash-attn', choices=('off', 'on', 'auto'), default='auto')
    args = parser.parse_args()
    (build_client if args.mode == 'build' else run_client)(args)


if __name__ == '__main__':
    main()
