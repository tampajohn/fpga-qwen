"""Build the original M5 CPU engine into a new immutable artifact directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from benchmark_native_phase0 import guard
from experiment import ROOT


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT / 'build/benchmark-status')
    args = parser.parse_args()
    guarded = guard(args.guard_state_dir)
    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        parser.error('This engine targets Apple ARM CPU hardware')
    dest = args.directory.resolve()
    if dest.exists():
        parser.error('Choose a new directory; existing builds are immutable')
    source = ROOT / 'engine'
    files = sorted([*source.glob('*.cpp'), *source.glob('*.hpp')])
    if not (source / 'core.cpp').is_file() or not (source / 'main.cpp').is_file():
        parser.error('Core and CLI sources must exist before building')
    inputs = {str(path): sha(path) for path in [*files, Path(__file__).resolve()]}
    dest.mkdir(parents=True)
    binary = dest / 'qwen_cpu'
    compiler = shutil.which('clang++')
    if not compiler:
        raise RuntimeError('clang++ not found')
    command = [compiler, '-std=c++20', '-O3', '-mcpu=native', '-flto=thin',
               '-ffp-contract=fast', '-fno-math-errno', '-pthread',
               *[str(path) for path in files if path.suffix == '.cpp'],
               '-framework', 'Accelerate', '-o', str(binary)]
    environment = os.environ.copy()
    removed = []
    for key in list(environment):
        if key.startswith(('DYLD_', 'GGML_', 'LLAMA_ARG_')):
            removed.append(key)
            environment.pop(key)
    report = dict(completed=False, gpu=False, pytorch=False, guard=guarded,
                  command=command, source_sha256=inputs, removed_environment_keys=removed,
                  compiler=subprocess.check_output([compiler, '--version'], text=True),
                  started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    try:
        with (dest / 'build.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        report['exit_code'] = result.returncode
        if result.returncode:
            raise RuntimeError(f'Compiler failed; inspect {dest / "build.log"}')
        for path, digest in inputs.items():
            if sha(Path(path)) != digest:
                raise RuntimeError('Source changed during build: ' + path)
        linked = subprocess.check_output(['otool', '-L', str(binary)], text=True)
        if any(name in linked.lower() for name in ('metal.framework', 'libllama', 'libggml', 'libtorch')):
            raise RuntimeError('Unexpected GPU or external neural runtime dependency')
        report.update(completed=True, binary=str(binary), binary_sha256=sha(binary), linked_libraries=linked)
    except Exception as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        (dest / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(dest / 'manifest.json')


if __name__ == '__main__':
    main()
