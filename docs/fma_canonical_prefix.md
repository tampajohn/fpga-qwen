# Experimental fused-prefill canonical prefix capability

This opt-in policy is `fp32_fma_b8_aligned_v1`, restricted to the
`fp32_b1_fp32_fma_prefill` profile. Build with
`--fp32-fma-canonical-prefix` in addition to the existing fused-prefill flags.
The option changes capability metadata only: it does not change neural arithmetic.
Without it, existing builds continue advertising `canonical_prefix_policy: none`.
No UI or default backend is changed.

The engine must use B8 prefill, B4 verification, eight reduction lanes, positive
workers, and general prefix reuse disabled. Explicit canonical reuse records only
acknowledged complete eight-token prefill blocks. All partial prefill tails
(counts 1–7), scalar decode and verification KV are excluded, even when their
input IDs match a subsequent prompt. Reuse starts at an eight-token boundary so
cold and reused execution retain identical prefill grouping. B4 tails and B1/B4
decode/verification must be recomputed.

The process identity and profile/policy/segmentation signature bind the reusable
boundary. A new process or changed signature clears it. Session changes reset
state. Runtime policy/profile failures, cancellation and generator closure clear
all host cache metadata and close the simulator. Cache-disabled execution accepts
either this explicit policy or `none`; cache-enabled execution requires this
policy exactly. The old FP64 policy and unknown policies are rejected for FMA.
`prefix_reuse_supported` stays false because arbitrary token-prefix reuse is unsafe.

## Validation before promotion

Source preparation includes mock checks for all tails, matching generated IDs,
changed signatures/processes, populated-cache capability failure and retry, and
cache-disabled behavior with both capable and archived none-policy binaries.
These mocks do not establish numerical parity or speed.

After independent source review, build one selected combined candidate with this
capability. Adapt the existing canonical real suite to explicit profile/policy
parameters and compare cache-on versus cache-off using the same binary. Require
identical IDs, retained positions, finite full-vocabulary logit bytes, and token
cache metadata. Include all tail lengths; mutations around 7/8 and 15/16;
shrinking/growing prompts; matching generated-token prefixes; session, worker,
cancel and restart transitions; and seeded/greedy suffix acceptance/rejection.
When a logit dump belongs to a rejected speculative future, compare after the same
continuation at the same retained position, then reset the diagnostic engines.

Use the chosen combined artifact for any required maximum-context gate, avoiding
redundant capacity fills on superseded binaries. Only after exact route gates may
an exclusive paired cache-on/off benchmark measure benefit. Cross-profile FMA
versus FP64 equivalence is not claimed. Frozen binaries are never relabeled.
