"""Numerical probe only: stock Verilator SV versus an 80-digit Decimal oracle.

No throughput claim and no model/runtime integration. The absolute 1e-12
fixture budget applies only to these bounded inputs (|V| <= 1.75), not to
model-quality acceptance or equivalence to the existing attention profile.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def reference(scenario, length, query):
    with localcontext() as ctx:
        ctx.prec = 80
        factor = (Decimal(1), Decimal(64), Decimal(1) / 1024)[scenario]
        visible = max(1, length - 4 + query + 1)
        scores = []
        for t in range(visible):
            scores.append(sum((Decimal((query * 13 + j * 7) % 19 - 9) / 16 * factor)
                              * (Decimal((t * 11 + j * 3) % 23 - 11) / 16 * factor)
                              for j in range(16)) / 4)
        weights = [(score - max(scores)).exp() for score in scores]
        denominator = sum(weights)
        return [sum(weight * Decimal((t * 5 + j * 17) % 29 - 14) / 8
                    for t, weight in enumerate(weights)) / denominator for j in range(16)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/online-attention-probe/result.json')
    args = parser.parse_args()
    directory = ROOT / 'build/online-attention-probe'
    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'rtl/online_attention_probe.sv'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    compiler_version = subprocess.check_output(['verilator', '--version'], text=True).strip()
    command = ['verilator', '--binary', '--top-module', 'online_attention_probe',
               '--Mdir', str(directory / 'obj'), '-j', '2', '-Wno-fatal', str(source)]
    with (directory / 'build.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    binary = directory / 'obj/Vonline_attention_probe'
    run = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    (directory / 'output.txt').write_text(run.stdout)
    expected = {(s, n, q, j): value for s in range(3) for n in (1, 15, 16, 17, 64, 65)
                for q in range(4) for j, value in enumerate(reference(s, n, q))}
    seen = set()
    errors = [0.0, 0.0]
    for line in run.stdout.splitlines():
        if not line.startswith('RESULT '):
            continue
        fields = line.split()
        assert len(fields) == 7, line
        key = tuple(map(int, fields[1:5]))
        assert key in expected and key not in seen, key
        seen.add(key)
        for i, number in enumerate(fields[5:7]):
            value = float(number)
            assert math.isfinite(value), (key, number)
            with localcontext() as ctx:
                ctx.prec = 80
                error = float(abs(Decimal.from_float(value) - expected[key]))
            errors[i] = max(errors[i], error)
    assert seen == set(expected), (len(seen), len(expected))
    assert max(errors) <= 1e-12, errors
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    report = dict(passed=True, outputs=len(seen), scenarios=['normal', 'large_scores', 'small_scores'],
                  lengths=[1, 15, 16, 17, 64, 65], tile=16, queries=4, head_dimension=16,
                  oracle_precision_digits=80, fixture_absolute_error_budget=1e-12,
                  max_absolute_errors=dict(three_pass=errors[0], tiled_online=errors[1]),
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  source_sha256=source_hash, compiler_version=compiler_version,
                  causal_scope='Each query sees max(1, length-4+q+1) keys; length1 repeats a one-key prefix across all queries. No zero-visible query cases.',
                  build_command=command, timing_claim=False, model_quality_claim=False,
                  algorithm_source='https://arxiv.org/abs/2205.14135')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
