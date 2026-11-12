# Combined attention candidate

This isolated composition starts from main `cdbf60a`: QK2 prefill attention, paired-head scalar QK/value accumulation, and fixed-size FP64 KV arrays. It restores preserved head-pair source from `d9120db` and V2 source from `32bb4c1`; the separate V2 measurement remains preserved from `8f6d723`. Original failed/mixed QK-only head-pair reports are unchanged. No default or live-server configuration changes are part of this source integration.

Build the selected hybrid B4 profile with `--batch-attention --attention-qk2 --attention-head-pairs --attention-value-pairs --static-kv`. FP32 scalar/B4 projection archives retain their strict compiler flags; stock FP64 worker/main use their existing flags. Operation 5 handles paired-query prefill QK, operation 6 handles paired-head QK/V for workers 1/2/4/8, and other worker counts retain operation 1 fallback. Both KV arrays still store FP64 values with the original dimensions. Cache policy and precision-profile identifiers remain unchanged.

Initial frozen artifact: `f17ae654f0c64ac47b63225a3406edc8bcd114cc86e67d56610090bf5d7c075d`. The initial 8-token prefill plus scalar prediction matched deployed `4d45e071` in every full-vocabulary logit bit. Independent full route/cache/suffix gates are recorded separately; actual capacity remains a promotion gate after a positive combined timing result.

## Compiler findings and fallback disclosure

The tested QK2 and V2 SV task bodies were retained byte-for-byte during composition. Independent linked assembly review confirms direct static KV loads and SIMD in operation 5 QK2 and operation 6 QK/V2. Operation 5 common-prefix QK at `0x100023f50–0x100023f70` uses a direct K load, adjacent query pair load and `fmla.2d`. Operation6 QK at `0x100024f94–0x100024fa8` similarly shares K. Its V2 vector path at `0x100025204–0x10002523c` loads four direct V values and uses compact accumulator `ld2/st2` plus independent FP64 vector FMAs; a scalar-V shared-head fallback loop remains. Deque lookups are absent from these hot loops.

Fixed-array compilation also changed fallback operation 1 QK: the compiler emits vector products followed by ordered scalar additions instead of the older scalar fused multiply-add chain. Therefore this document does **not** claim equality of every intermediate operation. Targeted final-output checks at workers 3/7/17 passed both short contexts and a 451-token prefix followed by scalar predictions at positions 452/453, with identical IDs/positions and all 151936 logit bits. See [fallback evidence](../reports/attention-combined-fallback.json). The exact executed [diagnostic script](../tools/experiments/combined-fallback-long-check.py) prints booleans; all six resulting cases were explicitly checked true afterward. No numerical tolerance was relaxed.

## Combined measurement

Use `tools/compare_attention_combined.py` against the deployed barrier-only hybrid. It validates both runtime precision profiles and the candidate's five intended build options, preserves warm worker pools during direct full-logit dumps, and requires fixed-count greedy IDs and full-vocabulary bits to match. Isolated QK2, V2 and fixed-KV ratios must not be multiplied to predict this composition. No combined speed claim is made before that measurement completes.
