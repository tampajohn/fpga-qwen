# Optional paired-query QK SIMD

`--attention-qk2` requires `--batch-attention` and excludes `--attention-dot-lanes8`. It changes only the stock FP64 worker's batched QK loop. Neither scalar decode nor model precision changes. Defaults remain unchanged.

Pairs of queries share one key load at each ascending feature index. Their FP64 sums remain independent; no horizontal reduction or reassociation is introduced. Both consume their common causal prefix; only the valid second query consumes its extra causal timestep. Softmax and value accumulation keep the original per-query time order. Odd tails compute an unused second common-prefix sum using zero-padded query storage, but never publish it. Score scratch is persistent per-worker SV storage, avoiding per-dispatch array initialization.

Candidate SHA256 `c5b414d226bafba3dc9419961e9adc626cf02ab7280aeefba159385410ffedf0` passed 38 full-vocabulary bit-exact comparisons against ordinary batched attention `07e22976`, covering counts 1–8, prefix lengths 0/1/7/8/63/452, truncation/overwrite, and sampled B4 counts 1–4 with workers 1/3/17. See [gate report](../reports/attention-qk2-gates.json). This bounded gate is not a full 4096-capacity run or a performance claim.

Independent linked assembly review identified the actual common-prefix j-loop at `0x1000255fc–0x100025644`: one shared scalar key load (`0x100025624`), adjacent query pair load (`0x100025630`), and `fmla.2d` (`0x100025634`). The accumulator remains in a vector register through all 128 features, with no calls, spills or horizontal reduction in the loop. The extra causal timestep uses scalar `fmadd`; ordinary single-query attention retains its scalar FMA policy. Dynamic KV deque lookup remains once per feature.

Reproduce gates with `python tools/check_attention_qk2.py --candidate <QK2-binary> --reference <ordinary-batched-attention-binary>`. Paired performance must compare those same profiles with `tools/compare_attention_qk2.py`, preserving warm worker pools and requiring exact output IDs/full-logit bits. No speed claim follows merely from observing SIMD instructions.
