# Optional batched FP64 attention dispatch

`--batch-attention` requires parallel attention and B8 prefill. It reserves worker operation 5 and leaves projection operations 0/2/3/4 and single-token attention operation 1 unchanged. The first candidate amortizes dispatch barriers; it does not share KV loads across query tokens or change precision.

For each layer, the main SV module normalizes/rotates each query/key and publishes all block KV. The worker then computes each query using the original per-head/per-query loops with causal end `initial_position + query_index`. Publishing future KV must not make it visible to earlier queries. Softmax and value accumulation retain their original order. The original contiguous result scratch is reused per query, then scattered once into fixed feature-major stride 8 output. Inactive output lanes can retain prior values; only active token lanes are consumed by main SV.

The host validates layer, count 1–8, start/window within 4096, rows 16 and width 2048 before job publication. Operation 5 reads KV rather than weights and bypasses weight-span validation. Query copies are 2048×8 doubles; each worker copies its disjoint head range with stride 128×8. Main waits for all worker reads before further KV mutation. Macro-off behavior is unchanged.

This changes the 451-token prefill prefix from 12,628 attention dispatches to 1,596, but does not remove its arithmetic. The preceding profile measured attention at about 40% of long prefill eval; a performance benefit must be measured separately, not inferred from dispatch counts.

Initial candidate SHA256 `07e22976b2c72325e9ff2a62ca7e12e733762a1b8ac957d8987fb1cd511bbed7` matched the frozen ordinary-attention hybrid's full vocabulary bits after 8 prefill tokens and one scalar prediction. Transport tests cover all counts, worker partitions 1/3/17, boundary validation, CV/hybrid waits and existing projection routes. Independent full-model gates subsequently passed: all counts 1–8, boundaries 127/128/129/255/452, future-KV poisoning, truncate/overwrite, cancellation/restart, cached/fresh session parity, sampled B4 verification and suffix replay. An additional 36 count/partition combinations across workers 1/3/17 matched IDs, positions and full-vocabulary bits. See the [independent gate report](../reports/batched-attention-independent.json) and [partition report](../reports/batched-attention-partitions.json).

## Paired end-to-end result

Three alternating pairs per workload compared deployed hybrid `97572eb3` against candidate `07e22976`, with 8 workers, canonical B8 prefill, 32 greedy outputs, and no prefix reuse or suffix drafting. Direct closed-file dumps preserve worker pools after warmup. Every fixed output ID and all final full-vocabulary logit bits matched. See the [raw paired report](../reports/batched-attention-paired.json).

| Workload | Prefill ratio median (range) | Median prefill seconds, baseline → candidate | Total request ratio median (range) |
| --- | --- | --- | --- |
| 85-token short | 1.148 (1.105–1.356) | 0.827 → 0.721 | 1.062 (1.047–1.262) |
| 452-token long | 1.168 (1.070–1.186) | 5.725 → 4.932 | 1.147 (1.045–1.160) |

Ratios above 1 favor the candidate; medians of per-run wall times need not equal median paired ratios. All six prefill and total-request comparisons improved. Median TTFT ratios were 1.141 short / 1.167 long. Decode uses the unchanged scalar path and was mixed (median ratios 0.972 / 0.996); no decode speedup is claimed. The first short pair showed broader host variation, so retain the ranges rather than treating the largest result as typical.

## Main integration and deployment

The optional browser **SystemVerilog · FP32 hybrid** backend now selects batched attention. The ordinary FP64 profile remains the default. Main artifact SHA256 `4d45e071a8a8aec353b638356c90dd42f9c90e69aaee929725e17e20dec7ebc3` passed all five [integration suites](../reports/batched-attention-main-integration.json), including exact cache, suffix and worker-partition checks. Its [browser check](../reports/batched-attention-main-browser.json) verified backend switching, 136 reused prefix tokens, matching suffix on/off output, cancellation/restart and no JavaScript errors.

The [actual context-capacity gate](../reports/batched-attention-capacity.json) filled both the main artifact and the previous hybrid reference to 4096 tokens through FP64 B8 prefill. FP32 B4 verification at positions 4093–4096 then matched IDs and all 151936 logit bits. A valid single-token tail succeeded and overflow was rejected. Prefill itself has no logit output; the report distinguishes its capacity acknowledgment from the final verification comparison. These are correctness checks, not additional timing measurements. The localhost server was restarted with this artifact after the gates passed.
