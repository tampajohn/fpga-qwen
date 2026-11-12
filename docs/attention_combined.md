# Combined attention in the optional hybrid backend

This composition starts from main `cdbf60a`: QK2 prefill attention, paired-head scalar QK/value accumulation, and fixed-size FP64 KV arrays. It restores preserved head-pair source from `d9120db` and V2 source from `32bb4c1`; the separate V2 measurement remains preserved from `8f6d723`. Original failed/mixed QK-only head-pair reports are unchanged. The optional browser hybrid now selects this composition; ordinary FP64 remains the default.

Build the selected hybrid B4 profile with `--batch-attention --attention-qk2 --attention-head-pairs --attention-value-pairs --static-kv`. FP32 scalar/B4 projection archives retain their strict compiler flags; stock FP64 worker/main use their existing flags. Operation 5 handles paired-query prefill QK, operation 6 handles paired-head QK/V for workers 1/2/4/8, and other worker counts retain operation 1 fallback. Both KV arrays still store FP64 values with the original dimensions. Cache policy and precision-profile identifiers remain unchanged.

Frozen artifact: `f17ae654f0c64ac47b63225a3406edc8bcd114cc86e67d56610090bf5d7c075d`. The initial 8-token prefill plus scalar prediction matched the previous deployed `4d45e071` in every full-vocabulary logit bit. All five [independent route/cache/suffix suites](../reports/combined-attention-independent.json) subsequently passed.

## Compiler findings and fallback disclosure

The tested QK2 and V2 SV task bodies were retained byte-for-byte during composition. Independent linked assembly review confirms direct static KV loads and SIMD in operation 5 QK2 and operation 6 QK/V2. Operation 5 common-prefix QK at `0x100023f50–0x100023f70` uses a direct K load, adjacent query pair load and `fmla.2d`. Operation6 QK at `0x100024f94–0x100024fa8` similarly shares K. Its V2 vector path at `0x100025204–0x10002523c` loads four direct V values and uses compact accumulator `ld2/st2` plus independent FP64 vector FMAs; a scalar-V shared-head fallback loop remains. Deque lookups are absent from these hot loops.

Fixed-array compilation also changed fallback operation 1 QK: the compiler emits vector products followed by ordered scalar additions instead of the older scalar fused multiply-add chain. Therefore this document does **not** claim equality of every intermediate operation. Targeted final-output checks at workers 3/7/17 passed both short contexts and a 451-token prefix followed by scalar predictions at positions 452/453, with identical IDs/positions and all 151936 logit bits. See [fallback evidence](../reports/attention-combined-fallback.json). The exact executed [diagnostic script](../tools/experiments/combined-fallback-long-check.py) prints booleans; all six resulting cases were explicitly checked true afterward. No numerical tolerance was relaxed.

## Combined measurement

Use `tools/compare_attention_combined.py` against the deployed barrier-only hybrid. It validates both runtime precision profiles and the candidate's five intended build options, preserves warm worker pools during direct full-logit dumps, and requires fixed-count greedy IDs and full-vocabulary bits to match. Isolated QK2, V2 and fixed-KV ratios must not be multiplied to predict this composition. The completed measurement is preserved in [the paired report](../reports/attention-combined-paired.json).

Three alternating pairs per context compared frozen combined `f17ae654` directly with deployed `4d45e071`, with fresh 85/452-token prompts and 32 greedy outputs. Every warm/scored output ID and diagnostic full-vocabulary logit bit matched. Long-context prefill speedup ratios were 1.111/1.150/1.142, decode 1.305/1.253/1.251, and total 1.135/1.164/1.157. Their medians were 1.142× prefill, 1.253× decode and 1.157× total. Short total ratios were 0.918/1.000/1.027 (median 1.000×); short prefill median was 0.967× and decode 1.026×. Thus this evidence supports a long-context gain, not a universal speed improvement. All measurements used eight workers; fallback worker-count throughput was not scored.

Independent broad route/cache/suffix checks passed under session 90252; the targeted fallback report supplements them. The separate [actual 4096-token capacity check](../reports/combined-attention-capacity.json) passed scalar decoding at position 4096, batched verification at positions 4093–4096, valid-tail handling and overflow rejection. Both scalar and batched comparisons matched all vocabulary-logit bytes against `4d45e071`; this exercises both new attention paths at the maximum position.

## Fresh CPU-only native comparison

The [matched comparison](../reports/combined-attention-native.json) uses the same frozen combined artifact, identical 85/452-token prompts, 32 greedy outputs, eight workers/threads and two warm repetitions. All engine output sequences matched. Native requests processed the full prompt with zero cached tokens. Startup and first requests are recorded separately; these are warm-process measurements without page-cache eviction.

| Engine | Short decode, tokens/s | Long decode, tokens/s | Short TTFT, seconds | Long TTFT, seconds |
| --- | ---: | ---: | ---: | ---: |
| Combined SV hybrid | 60.17 | 49.02 | 0.717 | 4.294 |
| llama.cpp FP16 weights | 139.38 | 126.56 | 0.194 | 0.626 |
| llama.cpp FP32 weights | 94.87 | 86.66 | 0.153 | 0.388 |

The combined backend remains behind both native baselines. Native inference uses FP16 KV and mixed arithmetic; the SV hybrid retains FP64 attention/KV and its declared FP32 projection routes. The native build enables Apple CPU BLAS, while Metal/CUDA/Vulkan/OpenCL are disabled and runtime GPU offload is zero. The SV backend uses no host neural kernels. This comparison does not imply identical arithmetic, memory precision, or CPU instruction selection.

The [installed-artifact record](../reports/combined-attention-install.json) verifies that main received the exact tested executable without recompilation. Its [isolated browser check](../reports/combined-attention-browser.json) passed backend switching, 136 reused prefix tokens, suffix on/off text parity, cancellation/restart and no JavaScript errors. The localhost server was restarted with that artifact after the full-context gate passed; the deployment record includes its healthy backend catalog and zero-GPU/no-PyTorch status.

## Paired-query value loads: experimental follow-up

The optional `--attention-query-value-pairs` flag shares value loads between the
two queries already paired by QK2. Each query retains its existing global
softmax, division and ascending-position value accumulation. The second query's
extra causal position is accumulated last; an odd final query ignores its unused
partner. Scalar paired-head attention is unchanged in source. This is not online
softmax or a FlashAttention integration.

Frozen candidate `5153c95534e5aae1689c426f81925d270fcb2d47c2568dcc258dbf0fdd244bf4`
passed 60 exact full-vocabulary comparisons against deployed `6c9ab44b`: all eight
prefill tails at prefixes 0/63/451, sampled verification, no-logit verification
and subsequent scalar continuation. The linked value loop uses FP64 SIMD on
independent outputs, including shared value loads; no horizontal reduction is
introduced. Build, gates and assembly evidence are in the `query-value-pairs-*`
reports. Actual maximum capacity and canonical/suffix lifecycle remain pending
for this candidate.

The [guarded full-model comparison](../reports/query-value-pairs-timing.json) used
three alternating pairs of 85/452-token prompts with 32 greedy outputs. Every
pair reproduced its own full logits and matched the deployed reference exactly.
Long-prompt median prefill speedup was 1.078x and total speedup 1.074x; all three
long totals improved (1.050–1.101x). Short prefill improved 1.029x, but total
speedup was 0.983x (0.861–0.991x), with slower decode in all three short pairs.
These are per-pair ratio medians, not multiplied component gains. The cause of
the decode regression is unproven. No default or UI promotion is made.

### Context-aware dispatch experiment

`--attention-query-value-dispatch` adds both value paths to one executable.
Trusted process configuration `QWEN_EXPERIMENTAL_QUERY_VALUE_PAIRS` selects `OFF`
(default), `ON`, or `MIN_CONTEXT:N` (1–4096). SystemVerilog branches on the actual
common causal KV window for each query pair and requires a second query; an odd
singleton retains the original path. This is experimental and not selected by
the browser. The `query_value_policy` command reports policy and actual SV route
counters across both worker pools.

Frozen executable `a3459b7dbf1d08e47923005283449cd5b8bd090f304fea94cc6a97b056a2be18`
compiled successfully. Forced OFF passed 60 exact full-vocabulary comparisons
against deployed `6c9ab44b`, including tails, sampled/no-logit verification and
scalar continuation at prefixes 0/63/451. See the `query-value-dispatch-build`
and `query-value-dispatch-reference-gates` reports. The [three-policy gate](../reports/query-value-dispatch-gates.json) passed 169
exact full-vocabulary comparisons and 771 actual route-count checks across
OFF/ON/MIN_CONTEXT:256. It includes threshold neighborhoods, odd tails,
verification, scalar continuation, full B8 prefill to position 4096, overflow
rejection and restart. Canonical/suffix lifecycle and performance checks remain
pending. No crossover or speed improvement is established for this binary.
