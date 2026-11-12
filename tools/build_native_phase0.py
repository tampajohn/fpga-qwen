"""Build an isolated, provenance-recorded CPU baseline; never replace live binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
GPU_BACKENDS = ('METAL', 'CUDA', 'VULKAN', 'OPENCL', 'HIP', 'MUSA', 'SYCL')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kleidiai', choices=['ON', 'OFF'], required=True)
    p.add_argument('--lto', choices=['ON', 'OFF'], required=True)
    p.add_argument('--jobs', type=int, default=6)
    a = p.parse_args()
    if not 1 <= a.jobs <= 18:
        p.error('jobs must be 1..18')
    source = ROOT / 'build/llama.cpp'
    directory = ROOT / f'build/native-phase0/kleidi-{a.kleidiai.lower()}-lto-{a.lto.lower()}'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'manifest.json').exists():
        raise RuntimeError('Frozen build already has a manifest; choose a separate build configuration')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
    commit = git('rev-parse', 'HEAD')
    if git('status', '--porcelain'):
        raise RuntimeError('Baseline source must be clean')
    configure = ['cmake', '-S', str(source), '-B', str(directory),
                 '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                 '-DGGML_NATIVE=ON', '-DGGML_BLAS=ON', '-DGGML_BLAS_VENDOR=Apple',
                 '-DGGML_OPENMP=OFF', f'-DGGML_CPU_KLEIDIAI={a.kleidiai}',
                 f'-DGGML_LTO={a.lto}', '-DLLAMA_BUILD_TESTS=OFF',
                 '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_SERVER=ON',
                 '-DLLAMA_BUILD_TOOLS=ON', '-DLLAMA_BUILD_APP=OFF',
                 '-DLLAMA_BUILD_UI=OFF', '-DLLAMA_OPENSSL=OFF',
                 *[f'-DGGML_{name}=OFF' for name in GPU_BACKENDS]]
    build = ['cmake', '--build', str(directory), '--parallel', str(a.jobs),
             '--target', 'llama-server', 'llama-bench']
    record = dict(completed=False, source_commit=commit, commands=[configure, build],
                  script_sha256=sha(Path(__file__)), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  note='CPU-only baseline candidate, not a tuned performance claim. OpenMP explicitly off to match previously unavailable runtime; Accelerate enabled.')
    try:
        for label, command in [('configure', configure), ('build', build)]:
            print(label, directory, flush=True)
            with (directory / f'{label}.log').open('w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        cache = (directory / 'CMakeCache.txt').read_text()
        for name in GPU_BACKENDS:
            if f'GGML_{name}:BOOL=OFF' not in cache:
                raise RuntimeError(f'GPU backend not disabled: {name}')
        if git('rev-parse', 'HEAD') != commit or git('status', '--porcelain'):
            raise RuntimeError('Source changed during build')
        record.update(completed=True, cache_sha256=sha(directory / 'CMakeCache.txt'),
                      build_options=[line for line in cache.splitlines() if line.startswith('GGML_') and ':BOOL=' in line],
                      artifacts={str(path.relative_to(directory)): sha(path) for path in (directory / 'bin').iterdir()
                                 if path.is_file() and (path.name in ('llama-server', 'llama-bench') or '.dylib' in path.name)})
    finally:
        (directory / ('manifest.json' if record['completed'] else 'failed-build.json')).write_text(json.dumps(record, indent=2) + '\n')
    print('Completed', directory, flush=True)


if __name__ == '__main__':
    main()
