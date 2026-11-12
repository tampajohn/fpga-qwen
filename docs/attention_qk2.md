# Optional paired-query QK SIMD

`--attention-qk2` requires `--batch-attention` and excludes `--attention-dot-lanes8`. It changes only the stock FP64 worker's batched QK loop. Neither scalar decode nor model precision changes. Defaults remain unchanged.

Pairs of queries share one key load at each ascending feature index. Their FP64 sums remain independent; no horizontal reduction or reassociation is introduced. Both consume their common causal prefix; only the valid second query consumes its extra causal timestep. Softmax and value accumulation keep the original per-query time order. Odd tails compute an unused second common-prefix sum using zero-padded query storage, but never publish it. Score scratch is persistent per-worker SV storage, avoiding per-dispatch array initialization.

Candidate SHA256 `c5b414d226bafba3dc9419961e9adc626cf02ab7280aeefba159385410ffedf0` passed 38 full-vocabulary bit-exact comparisons against ordinary batched attention `07e22976`, covering counts 1–8, prefix lengths 0/1/7/8/63/452, truncation/overwrite, and sampled B4 counts 1–4 with workers 1/3/17. See [gate report](../reports/attention-qk2-gates.json). This bounded gate is not a full 4096-capacity run or a performance claim.

Independent linked assembly review identified the actual common-prefix j-loop at `0x1000255fc–0x100025644`: one shared scalar key load (`0x100025624`), adjacent query pair load (`0x100025630`), and `fmla.2d` (`0x100025634`). The accumulator remains in a vector register through all 128 features, with no calls, spills or horizontal reduction in the loop. The extra causal timestep uses scalar `fmadd`; ordinary single-query attention retains its scalar FMA policy. Dynamic KV deque lookup remains once per feature.

Reproduce gates with `python tools/check_attention_qk2.py --candidate <QK2-binary> --reference <ordinary-batched-attention-binary>`. Paired performance must compare those same profiles with `tools/compare_attention_qk2.py`, preserving warm worker pools and requiring exact output IDs/full-logit bits. No speed claim follows merely from observing SIMD instructions.

The subsequent [independent validation](../reports/qk2-independent.json) passed all five suites: every batch/worker partition, causal tails and rollback, canonical prefix reuse, exclusion of scalar-decode states from reuse, and seeded suffix replay. IDs and all finite vocabulary-logit bytes matched at corresponding positions. This report does not claim a capacity test or timing result.

## Paired end-to-end timing

Three alternating warmed pairs per workload, 8 workers and 32 outputs, compared `07e22976` against `c5b414d2`. Both use batched FP64 attention, FP64 B8 prefill projections and FP32 scalar projections. No prefix cache or suffix drafting was used. Full output IDs and full-vocabulary logit bits matched throughout. [Raw report](../reports/attention-qk2-paired.json).

| Workload | Prefill ratio median (range) | Total request ratio median (range) |
| --- | --- | --- |
| 85-token short | 0.999 (0.926–1.054) | 1.018 (0.974–1.061) |
| 452-token long | 1.052 (1.039–1.062) | 1.037 (1.035–1.042) |

Ratios above 1 favor QK2. The short result is neutral/mixed; all three long prefill and total-request pairs improve modestly. Unchanged scalar decode varied in opposite directions between workloads (median ratios 1.051 short / 0.949 long), so it is not evidence of a decode optimization. Retain this as optional rather than automatically changing defaults. QK2-specific actual 4096-capacity validation and integration/browser checks remain pending if promotion is pursued; the earlier barrier-only capacity run does not validate this new loop.
