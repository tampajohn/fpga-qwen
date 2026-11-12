# Opt-in canonical prefix reuse for hybrid inference

The hybrid engine can reuse completed FP64 B8 prompt blocks while preserving exact fresh-request behavior in the tested cases. The opt-in hybrid UI backend enables this policy with a capability-verified executable. FP64 remains the default backend. Optional FP32 B4 suffix verification never advances the canonical boundary; only acknowledged full FP64 prefill blocks do.

`RTLChatEngine(..., precision_profile='fp32_b1_fp64_prefill', block_size=1, prefill_size=8, lanes=8, workers=8, reuse_prefix=False, canonical_prefix_reuse=True)` selects the narrow policy. The runtime must explicitly advertise `canonical_prefix_policy: "fp64_b8_aligned_v1"`; older hybrid binaries reject this request before prefill. General `prefix_reuse_supported` remains false. No runtime capability or filename silently turns caching on.

The engine tracks a canonical boundary separately from total KV/token history. At the next request it computes:

`reuse = floor(min(shared_prefix, canonical_end, new_prompt_length - 1) / 8) * 8`

It truncates to that boundary and recomputes the rest. Only an acknowledged, validated full eight-token prefill block advances the boundary. Partial tails and every FP32 decode token remain ineligible—even when their token IDs exactly match the next prompt. Growing a partial tail into a full block therefore recomputes the tail using the correct route and reduction order. A mismatch inside a block rolls back to its beginning; shortening a prompt also caps reuse before the new final decode position.

The canonical boundary belongs to a model-process epoch and session. Reset, process loss/restart, cancellation, generator closure or protocol failure clears it. Worker repartitioning retains the fixed precision/lane policy and does not change existing KV. `canonical_reusable_tokens` is reported separately from total KV tokens. No model arithmetic runs in the host helper.

## Evidence

[The real gate report](../reports/canonical-prefix-gates.json) contains 33 fresh-versus-cached comparisons covering every partial-tail length, growth/shrink, changed tokens around block boundaries, new turns/sessions, one-token and one-output requests, worker counts 1/3/17, process loss, cancellation and generator closure. All 151,936 finite logits match bitwise at each comparison, and generated token IDs/cache histories match. These cases reused 568 tokens in total.

[The focused decode-exclusion check](../reports/canonical-prefix-decode-exclusion.json) constructs the next prompt from the exact previous cached token IDs plus one new token. The shared prefix is 36 tokens, but the canonical boundary is 32: only 32 are reused and all four matching FP32 states are recomputed. The result matches fresh inference bitwise and in generated IDs. This specifically rules out incorrectly reusing matching generated states.

Twenty-seven host/cache/suffix/transport tests pass, including route-aware mock histories, malformed/missing capability, bad acknowledgments, EOS, the 4096-token request budget and unchanged default behavior. The real checks make no speed claim.

The tested policy-capable binary is `bf8242221d07675916cd6711869d3568cb24d8655824d875dee5d1eefda943a7`. Its [manifest](../reports/canonical-prefix-build-manifest.json) has the same model/worker/compiler source hashes as the validated `d27a2ffc` hybrid except for the capability JSON in `sim/full_model.cpp`; no neural arithmetic changed.

## Paired latency measurement

Twelve alternating greedy pairs compared the same hybrid executable with reuse off and on. Each pair first primed a fresh session, then measured either a repeated prompt or an appended turn, with an eight-token output budget and eight workers. All output IDs and full-vocabulary logits matched. The [raw report](../reports/canonical-prefix-paired.json) records individual timings and safe reuse boundaries.

| Workload | Median first-token time, reuse off | Reuse on | Reused tokens |
|---|---:|---:|---:|
| Short repeated prompt | 0.810 s | 0.065 s | 80 |
| Short appended turn | 1.605 s | 0.861 s | 80 |
| Long repeated prompt | 5.865 s | 0.102 s | 448 |
| Long appended turn | 7.311 s | 1.364 s | 448 |

These measurements isolate warm-prefix reuse, not faster cold prefill or decode. They are not comparisons against native llama.cpp prefix caching. Priming is excluded from follow-up latency; end-to-end claims must account for whether a reusable prefix actually exists.

Reproduce host checks with `uv run --locked python -m unittest tests.test_canonical_prefix tests.test_suffix_chat tests.test_hybrid_backend tests.test_prefill_transport`. Set `SV_CANONICAL_BINARY` to the explicitly policy-capable binary and run `uv run --locked python -m unittest tests.test_canonical_prefix_real` for both real gates. The ordinary hybrid binary/default remains usable without this option.
