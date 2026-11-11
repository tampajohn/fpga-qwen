"""Measure persistent Verilator row workers and verify full-logit identity."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
try:
    from .rtl_model import RTLModel
    from .experiment import ROOT
except ImportError:
    from rtl_model import RTLModel
    from experiment import ROOT


def main():
    records = []
    golden = None
    # Fixed token positions and input vectors across all worker counts.
    sequence = [9707, 11, 1917, 0, 9707, 11, 1917, 0, 9707]
    for workers in [0, 1, 2, 4, 8]:
        model = RTLModel(workers=workers)
        try:
            model.start()
            model.token(9707, logits=False)
            model.token(11, logits=False)
            times, hashes, choices = [], [], []
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'logits.bin'
                for token in sequence:
                    start = time.perf_counter()
                    best, _, _ = model.token(token)
                    times.append(time.perf_counter()-start)
                    choices.append(best)
                    assert model.command(f'dump {path}') == 'dumped'
                    hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            if golden is None:
                golden = hashes
            assert hashes == golden, f'Full-logit mismatch for {workers} workers'
            matrices = {}
            if workers:
                for name, slot, rows, width, repetitions in [
                    ('gate', 11, 3072, 1024, 20), ('down', 13, 1024, 3072, 20),
                    ('vocabulary', 2, 151936, 1024, 3)]:
                    samples = []
                    for _ in range(3):
                        response = model.command(f'bench {slot} {rows} {width} {repetitions}').split()
                        samples.append(float(response[0])/repetitions)
                    matrices[name] = dict(median_seconds=statistics.median(samples), samples_seconds=samples)
            record = dict(workers=workers, mode='serial' if workers==0 else 'partitioned',
                          token_seconds=times, median_tokens_per_second=1/statistics.median(times),
                          mean_tokens_per_second=len(times)/sum(times),
                          full_logits_bit_identical=True, choices=choices, matrices=matrices)
            records.append(record)
            print(json.dumps(record), flush=True)
        finally:
            model.close()
    report = dict(cpu=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                  gpu=False, pytorch=False, model='Qwen3-0.6B', precision='SV real / BF16 weights',
                  note='Short-context token forward timings, including IPC; 2 warmup prompt tokens, 9 measured tokens. No FPGA speed prediction.',
                  runs=records)
    path = ROOT / 'build/parallel-benchmark.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Saved {path}', flush=True)


if __name__ == '__main__':
    main()
