"""Cold/steady timings and full-logit errors for isolated SV projection variants."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import numpy as np
try:
    from .rtl_model import prepare_layout
    from .experiment import ROOT
except ImportError:
    from rtl_model import prepare_layout
    from experiment import ROOT


def run(binary, workers, lanes, sequence):
    model, layout = prepare_layout(None)
    process = subprocess.Popen([str(binary), str(model/'model.safetensors'), str(layout), str(workers)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, bufsize=1)
    def command(line):
        process.stdin.write(line+'\n');process.stdin.flush()
        reply = process.stdout.readline().strip()
        if not reply:
            raise RuntimeError(process.stderr.read())
        return reply
    try:
        if process.stdout.readline().strip() != 'ready':
            raise RuntimeError(process.stderr.read())
        if lanes != 1:
            assert command(f'lanes {lanes}') == 'configured'
        timings, outputs, ids = [], [], []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'logits.bin'
            for token in sequence:
                start = time.perf_counter()
                response = command(f'token {token} 1').split()
                timings.append(time.perf_counter()-start)
                ids.append(int(response[0]))
                assert command(f'dump {path}') == 'dumped'
                outputs.append(np.fromfile(path, dtype=np.float64))
        rss = int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(process.pid)], text=True).strip())
        return dict(binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    workers=workers, lanes=lanes, cold_first_token_seconds=timings[0],
                    steady_token_seconds=timings[1:], steady_tps=1/statistics.median(timings[1:]),
                    final_rss_mib=rss/1024, greedy_ids=ids), outputs
    finally:
        process.terminate()
        process.wait(timeout=5)
        process.stdin.close();process.stdout.close();process.stderr.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binaries', nargs='+', type=Path)
    parser.add_argument('--reference', type=Path, default=ROOT/'build/full/Vqwen_behavioral')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path, default=ROOT/'build/projection-probe.json')
    args = parser.parse_args()
    sequence = [9707, 11, 1917, 0, 9707]
    reference, expected = run(args.reference, 0, 1, sequence)
    results = [reference]
    print(json.dumps(reference), flush=True)
    for binary in args.binaries:
        for lanes in (1, 8):
            record, outputs = run(binary, args.workers, lanes, sequence)
            record['max_absolute_logit_error'] = max(float(np.max(np.abs(a-b))) for a,b in zip(outputs, expected))
            record['bit_identical_logits'] = all(a.tobytes()==b.tobytes() for a,b in zip(outputs, expected))
            record['greedy_ids_match'] = record['greedy_ids']==reference['greedy_ids']
            for actual, golden in zip(outputs, expected):
                np.testing.assert_allclose(actual, golden, atol=1e-9, rtol=1e-9)
            assert record['greedy_ids_match'], record
            results.append(record)
            print(json.dumps(record), flush=True)
    args.output.write_text(json.dumps(dict(token_sequence=sequence, runs=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
