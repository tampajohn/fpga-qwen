# Isolated FP32 suffix verification

The B4 artifact uses FP32 projections for verification and B1 decode, with the
existing FP64 B8 prefill and FP64 attention/KV. The host accepts B4 for the hybrid
profile only with canonical prefix reuse enabled and general prefix reuse
explicitly disabled. It checks runtime `max_verification_block: 4`,
`verification_projection: fp32` and the canonical policy before model work.
The B1 runtime contract remains separate and unchanged. No UI/server selection
is changed by this experiment.

`tests.test_suffix_real.RealFP32SuffixTests` exercises greedy and seeded chat
(seeds 42 and 43), draft limits one and three, fresh and reused canonical
prefixes, forced wrong proposals, cancellation after a drafted update and
restart. All 14 complete on/off comparisons match output IDs, text, accepted
cache tokens and full logits at an identical fixed-token continuation. Recorded
comparisons include 59 accepted drafts out of 141 proposed drafts. This is
route correctness, not a new general model-quality result.

The initial raw final-logit comparison failed. Its diagnostic is retained in
`reports/fp32-suffix-logit-alignment.json`: for the greedy draft-three case the
serial buffer described position 107, while the speculative buffer described
position 109 even though both retained KV caches ended at 107 after EOS. A
shared token-198 continuation returned position 108 in both, with all 151,936
logits bit-identical. The probe is explicitly truncated afterward, without
advancing host cache metadata or sampling RNG. Raw buffer identity at unequal
positions is not used as an arithmetic gate.

The existing `tools/benchmark_suffix_chat.py --hybrid-verification` runner
measures the same binary with suffix drafting off/on and fresh logical KV per
request. It requires identical input/output IDs, finish reason, retained cache
and full continuation logits; continuation probes and binary dumps occur after
timing. Prefill, first-token, decode and total latency are separate. No native
backend equivalence is claimed. Timing results will determine whether exposing
this suffix option is worthwhile; it remains unavailable in the deployed hybrid
until integration is separately authorized and tested.


The six paired runs completed with every equality gate passing. On repeated
code, the median off/on ratio was 1.080 for total request time and 1.179 for
decode time (39 outputs, 24 of 24 proposed tokens accepted). Ordinary prose was
essentially flat: median ratios 0.988 total and 0.992 decode (34 outputs, two of
three proposals accepted). This supports an optional repetition-oriented suffix
mode, not default-on drafting or a general inference speedup. Exact paired rows
and separate timing components are in `reports/fp32-suffix-paired.json`; startup
and continuation probes are excluded and all requests began with empty logical
KV. The benchmark used seed-42 sampled decoding, three alternating pairs per
workload, and respected EOS rather than forcing the 64-token budget.
