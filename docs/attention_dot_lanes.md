# Optional eight-lane attention score reduction

`--attention-dot-lanes8` changes only the parallel SV attention worker's Q·K
reduction over 128 head features. Eight independent FP64 partial sums replace
one serial sum; the final reduction visits lanes in order. Head partitioning,
causal limits, softmax, value accumulation and KV storage are unchanged. Scalar
worker-zero fallback is unchanged. This is floating-point reassociation, not
bit-identical arithmetic, and remains disabled by default.

Three alternating-order pairs use identical B8 prefill, eight workers, 85/452
prompt tokens and 32 greedy output tokens, with empty logical KV per request.
Every warm/scored output sequence matched. Warm final full-vocabulary errors
were below 4.8e-14 against the current B8 baseline (declared tolerance 1e-6).
Both processes remained resident; no OS cache eviction was attempted.

| Context | Decode throughput ratios per pair | Median total-request speed ratio |
|---|---|---:|
| 85 tokens | 1.040, 1.026, 0.983 | 1.019× |
| 452 tokens | 1.067, 1.077, 1.266 | 1.050× |

The longer-context result is more promising; short decode has a regression in
one pair and the third long pair is noticeably larger than the first two.
These are small local samples, not confidence intervals or general guarantees.
An earlier run overlapped another agent's correctness work and was cancelled;
its timings are excluded. Other agents held heavy work during the completed run.
Raw timing evidence: [paired report](../reports/attention-dot-paired.json).

The full sixteen-case, 292-target distribution probe preserved every top-1 ID,
with maximum logit difference 1.29e-13 and loss/KL differences at floating-point
roundoff scale. This does not establish broad model quality. Separate tests
covered 1/3/8/17-worker head partitions, sampled selections, truncation, reset,
restart and deterministic replay. [Distribution report](../reports/attention-dot-quality.json).
No default or live change follows from these results alone.

Reproduce by adding `--attention-dot-lanes8` to the B8 builder flags. The option
requires `--parallel-attention`. Use `tools/compare_attention_dot.py --reference
BASELINE --candidate CANDIDATE --pairs 3 --output build/attention-dot-paired.json`;
both executables must support B8 prefill. Set `SV_DOT_REFERENCE` and
`SV_DOT_BINARY` to run `python -m unittest tests.test_attention_dot -v`.
The [build manifest](../reports/attention-dot-manifest.json) records exact flags,
sources, compiler and executable hash.
