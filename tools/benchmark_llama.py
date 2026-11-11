#!/usr/bin/env python3
"""Record a local CPU-only llama.cpp baseline, without changing the chat backend."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build'


def main():
    cmake = (BUILD / 'llama-cpu/CMakeCache.txt').read_text()
    for backend in ['METAL', 'CUDA', 'VULKAN', 'OPENCL']:
        if f'GGML_{backend}:BOOL=OFF' not in cmake:
            raise RuntimeError(f'{backend} was not disabled at build time')
    binary = BUILD / 'llama-cpu/bin/llama-bench'
    dest = BUILD / 'bench'
    report = dict(cpu=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                  llama_commit=subprocess.check_output(['git', '-C', str(BUILD / 'llama.cpp'),
                                                         'rev-parse', 'HEAD'], text=True).strip(),
                  gpu_backends_compiled=False, reference_app_reported_tokens_per_second=40,
                  context_depth=512, generated_tokens=128, repetitions=3,
                  notes=['llama-bench synthetic decode; excludes sampling/tokenization/browser/RTL overhead',
                         'The app reference is user-reported, not a matched-prompt benchmark',
                         'Both llama.cpp runs use default FP16 KV; app KV is FP32. This llama-bench rejects f32 KV.',
                         'Both models converted from the same locally cached Qwen3-0.6B checkpoint',
                         'No speculative decoding in llama-bench'], runs=[])
    # Same four threads as the app, and the tool's untouched default thread count.
    for precision, kv in [('f32', 'f16'), ('q4_k_m', 'f16')]:
        for setting in ['4', 'default']:
            cmd = [str(binary), '-m', str(dest / f'qwen3-0.6b-{precision}.gguf'),
                   '-p', '0', '-n', '128', '-d', '512', '-r', '3', '-o', 'json',
                   '-ngl', '0', '-nkvo', '1', '-nopo', '1', '-ctk', kv, '-ctv', kv]
            if setting != 'default':
                cmd += ['-t', setting]
            print(f'Benchmarking {precision}, threads={setting}, CPU only...', flush=True)
            start = time.monotonic()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            (dest / f'{precision}-{setting}.json').write_text(result.stdout)
            (dest / f'{precision}-{setting}.log').write_text(result.stderr)
            result.check_returncode()
            rows = json.loads(result.stdout)
            if len(rows) != 1 or rows[0]['n_gpu_layers'] != 0:
                raise AssertionError('Unexpected benchmark configuration')
            row = rows[0]
            report['runs'].append(dict(precision=precision, thread_setting=setting, command=cmd,
                                       wall_seconds=time.monotonic()-start, result=row))
            print(f"  {row['avg_ts']:.2f} ± {row['stddev_ts']:.2f} tok/s; "
                  f"actual threads={row['n_threads']}", flush=True)
    (dest / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Wrote {dest / "comparison.json"}')


if __name__ == '__main__':
    main()
