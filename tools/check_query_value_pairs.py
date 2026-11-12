"""Exact full-model gates for paired-query V loads. No timing claim."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from rtl_model import RTLModel
from compare_final_fma import read_dump_logits
from check_phase_pools_real import telemetry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.reference.resolve(), args.candidate.resolve()]
    assert sha(paths[0]) == '6c9ab44bf49b9f35ebf1a86897c284640041fb2a1a811f38184bff306f9e1b89'
    manifest = json.loads((paths[1].parent.parent / 'manifest.json').read_text())
    assert manifest['options']['attention_query_value_pairs'] is True
    assert manifest['binary_sha256'] == sha(paths[1])
    models = [RTLModel(binary=p, workers=8, lanes=8, block_size=4, prefill_size=8) for p in paths]
    report = dict(completed=False, timing_claim=False, artifacts=[dict(path=str(p), sha256=sha(p)) for p in paths],
                  cases=[], source_sha256=sha(__file__))
    pattern = [9707, 11, 358, 40, 1097, 264, 1296, 198]
    try:
        for model in models:
            model.phase_pool_selector = 'PREFILL_12'
            model.start()
        capabilities = [json.loads(m.command('precision')) for m in models]
        assert capabilities[0] == capabilities[1]
        assert capabilities[0]['canonical_prefix_policy'] == 'fp32_fma_b8_aligned_v1'
        report['capabilities'] = capabilities
        with tempfile.TemporaryDirectory() as directory:
            def equal(label, outputs):
                assert outputs[0] == outputs[1], (label, outputs)
                arrays = [read_dump_logits(m, Path(directory) / str(i)).tobytes() for i, m in enumerate(models)]
                assert arrays[0] == arrays[1], label
                report['cases'].append(dict(label=label, outputs=outputs[0], bit_exact=True,
                                           logits_sha256=hashlib.sha256(arrays[0]).hexdigest()))

            for prefix in (0, 63, 451):
                for model in models:
                    model.reset()
                    tokens = (pattern * 57)[:prefix]
                    for offset in range(0, prefix, 8):
                        model.prefill(tokens[offset:offset + 8])
                for count in range(1, 9):
                    for model in models:
                        assert model.command(f'truncate {prefix}') == 'truncated'
                        model.prefill(pattern[:count])
                    equal(f'prefix{prefix}-tail{count}', [m.token(198)[:2] for m in models])
                for count in range(1, 5):
                    for logits in (False, True):
                        for model in models:
                            assert model.command(f'truncate {prefix}') == 'truncated'
                        blocks = [[(x[0], x[1]) for x in m.block(pattern[:count], logits=logits,
                                   uniforms=[.031, .72, .999, .43][:count] if logits else None)] for m in models]
                        if logits:
                            equal(f'prefix{prefix}-sampled-block{count}', blocks)
                        else:
                            assert blocks[0] == blocks[1]
                        equal(f'prefix{prefix}-block{count}-logits{logits}-continuation',
                              [m.token(198)[:2] for m in models])
                print(f'PASS prefix {prefix}', flush=True)
        report['phase_telemetry'] = [telemetry(m, True) for m in models]
        for data in report['phase_telemetry']:
            counts = data['dispatch_counts']
            assert counts[1][1] > 0 and counts[0][0] > 0 and counts[0][2] > 0
            assert counts[0][1] == counts[1][0] == counts[1][2] == 0
        assert all(sha(p) == record['sha256'] for p, record in zip(paths, report['artifacts']))
        report['completed'] = True
    finally:
        for model in models:
            model.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS', len(report['cases']), 'full-vocabulary comparisons')


if __name__ == '__main__':
    main()
