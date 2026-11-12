# Optional batched FP64 attention dispatch

`--batch-attention` requires parallel attention and B8 prefill. It reserves worker operation 5 and leaves projection operations 0/2/3/4 and single-token attention operation 1 unchanged. The first candidate amortizes dispatch barriers; it does not share KV loads across query tokens or change precision.

For each layer, the main SV module normalizes/rotates each query/key and publishes all block KV. The worker then computes each query using the original per-head/per-query loops with causal end `initial_position + query_index`. Publishing future KV must not make it visible to earlier queries. Softmax and value accumulation retain their original order. The original contiguous result scratch is reused per query, then scattered once into fixed feature-major stride8 output. Inactive output lanes can retain prior values; only active token lanes are consumed by main SV.

The host validates layer, count1–8, start/window within4096, rows16 and width2048 before job publication. Operation5 reads KV rather than weights and bypasses weight-span validation. Query copies are2048×8 doubles; each worker copies its disjoint head range with stride128×8. Main waits for all worker reads before further KV mutation. Macro-off behavior is unchanged.

This changes the 451-token prefill prefix from12,628 attention dispatches to1,596, but does not remove its arithmetic. The preceding profile measured attention at about40% of long prefill eval; a performance benefit must be measured separately, not inferred from dispatch counts.

Initial candidate SHA256 `07e22976b2c72325e9ff2a62ca7e12e733762a1b8ac957d8987fb1cd511bbed7` matched the frozen ordinary-attention hybrid's full vocabulary bits after8prefill tokens and one scalar prediction. Transport tests cover all counts, worker partitions1/3/17, boundary validation, CV/hybrid waits and existing projection routes. Independent long-context/causal/cache/suffix gates are required before timing/promotion.
