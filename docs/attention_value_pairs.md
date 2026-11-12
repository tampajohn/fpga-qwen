# Optional paired-head value accumulation

This extends the preserved head-pair experiment at `d9120db`. Add `--attention-value-pairs` to its existing `--attention-head-pairs` build. Both softmax denominators are completed in their original order, then each timestep supplies one V load to two independent SV accumulators. Each head/feature retains ascending-t FMA order and operand order `scale * value + accumulator`. A compact persistent `[128][2]` SV array holds paired outputs; one final scatter restores head-major output layout. The original op6 worker eligibility and fallback are unchanged. Default builds do not select this option.

Artifact SHA256 `43c108188228587ae994d2598ae45d81614c2e36299713dbf6e310f11bfca165` passed the existing 49-case full-vocabulary bitwise gate against ordinary batched attention `07e22976`, including sampled token IDs/positions, prefix lengths 0/1/7/8/63/452, worker eligibility/fallback, overwrite and restart. [Gate report](../reports/attention-value-pair-gates.json). Actual full-capacity and independent cache/suffix checks remain promotion gates.

Independent final assembly review confirmed vector scale division at `0x100025438` and the value feature loop `0x100025440–0x100025490`: shared V load at `0x100025474`, compact vector accumulator load at `0x100025478`, `fmla.2d` at `0x10002547c`, and vector store at `0x100025480`. Large output-array bounds checks occur only in the final scatter. One shared deque lookup remains per feature; there are no calls, horizontal reductions or stack accesses in the feature loop. This is structural evidence, not a speed claim.

Reproduce correctness with `tools/check_attention_head_pairs.py --candidate <V2-binary> --reference <ordinary-batched-attention-binary>`. Use `tools/compare_attention_value_pairs.py` for runtime-validated paired end-to-end timing with warm worker pools, fixed output IDs and exact full-vocabulary logits. Do not infer a performance improvement from SIMD instructions alone.

## Paired end-to-end timing

Three alternating warmed pairs per workload compared `43c10818` against ordinary batched attention `07e22976`, using 8 workers, B8 prefill and 32 greedy outputs. Prefix reuse and suffix drafting were disabled. Every fixed output ID and final full-vocabulary logit bit matched. [Raw report](../reports/attention-value-pairs-paired.json).

| Workload | Decode ratio median (range) | Median decode tokens/s, baseline → candidate | Total request ratio median (range) |
| --- | --- | --- | --- |
| 85-token short | 0.987 (0.976–1.142) | 57.87 → 57.53 | 1.009 (0.966–1.038) |
| 452-token long | 1.178 (1.174–1.248) | 40.12 → 47.74 | 1.024 (1.015–1.032) |

Ratios above 1 favor the candidate; median absolute timings and median paired ratios need not match exactly. All three long decode and total-request pairs improved. Short decode remained mixed/neutral, so this is not a universal throughput claim. The prefill path was unchanged and its long median ratio was 0.994. This supports retaining V2 for independent route/cache/suffix and actual-capacity validation. Combining it with QK2 prefill or static KV requires a separate build, exactness gates and paired measurements; isolated ratios must not be multiplied to predict the combined result.
