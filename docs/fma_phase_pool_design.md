# Proposed composition: fused prefill and persistent phase pools

Status: source/options plan only, based on merged main `d4d9f1e`. No combined binary has been built, and no default/UI changes are proposed here. The [build plan](../tools/experiments/fma-phase-build-plan.json) contains the exact argument vector, reviewed compiler SHA, source hashes, predicted output path and frozen FMA-only reference SHA. `$WORKSPACE` denotes the local sandbox root. `--print-path` was used solely to validate argument compatibility and obtain the path; it did not build anything.

## Routing and precision

Existing merged code can compose the two independent features without new neural arithmetic. Use `--fp32-prefill-b8 --fp32-prefill-fma --phase-pools-experiment` alongside the existing combined QK2/head-pair V2/static-FP64-KV hybrid flags. The strict FMA compiler is required; do not add global contraction/fast-math flags or alter the nonfused archives.

| Explicit wire purpose | Selected pool when enabled | Projection and attention routes |
|---|---:|---|
| Prefill count1..4 | 12 | Operation2 dedicated fused FP32 B4 tail; operation5 FP64 attention |
| Prefill count5..8 | 12 | Operation3 fused FP32 B8; operation5 FP64 attention |
| Verification count1..4, including logits0 | 8 | Operation4 separate nonfused FP32 B4; operation0 nonfused vocabulary; operation5 FP64 attention |
| Scalar token/sample, including logits0 | 8 | Operation0 separate nonfused B1/vocabulary; operation6 eligible paired-head FP64 attention |

All workers share one immutable weight mapping and one coordinator-owned main SV model with FP64 KV. Worker counts change row/head partitions, not reduction order within a matrix row or attention query. Uneven partitions at twelve workers still require direct numerical gates. Both pools construct the existing FMA worker models, including extra tail/prefill scratch even when idle; quantify startup/RSS rather than assume zero overhead.

The precision profile remains `fp32_b1_fp32_fma_prefill`, with fused prefill and separate scalar/verification/vocabulary accumulation. Scheduler metadata is an independent field. Canonical prefix policy must stay `none`: general and canonical KV reuse remain disabled. The existing phase UI adapter hardcodes FP64 prefill plus canonical prefix reuse and **must not be inherited or reused** for this profile. A future trusted adapter should instantiate `RTLChatEngine` with the FMA profile, workers8/lanes8/B8 prefill/B4 verification, both reuse modes disabled, then set `sim.phase_pool_selector='PREFILL_12'` before launch. Verify both precision capabilities and phase telemetry after every restart. No adapter implementation belongs to this plan.

## Merge and build checks

Main already merged the relevant changes. The main remaining integration risk is semantic: do not let the dedicated fused operation2 tail fall into the nonfused operation4 verification branch, or let the purpose selector infer behavior from token count. Existing selected-pool logic and verification-purpose consistency assertion should remain intact. Fixed workers8/lanes8 controls remain true no-ops; other settings fail closed. The two ownership guards must still join both pools before the shared attention root or weights are destroyed.

Builder tag hashing is applied after the phase/FMA options, avoiding path-length collisions. The actual manifest must record both flags, distinct strict module commands, stock worker/main flags, reviewed compiler identity and all source hashes. Inspect the final linked functions: fused arithmetic only in prefill/tail modules, separate B1/verification/vocabulary math preserved, existing stock FP64 attention still selected. Preserve frozen reference013a and phase-only162bed artifacts.

## Minimal gates before scoring

1. Reuse policy and FMA scheduler mocks, extending transport coverage to twelve-worker uneven and empty partitions. Prove op2/op3 go only to the prefill pool and op4/op0 only to decode/verification, independent of count/logits. Check pool constructor count2, per-pool counters, no-op configuration and exception cleanup scope.
2. Compare the combined artifact directly with FMA-only013a, expecting exact full-vocabulary bits and IDs under identical canonical fresh routes. Cover prefill counts1..8, tails crossing63/64 and127/128,85/452 contexts and scalar continuation; include sampled B4 verification and logits0 followed by a scalar head. Test OFF and PREFILL_12 in the same combined binary. Do not weaken exactness to the older FP64-vs-FMA descriptive tolerance.
3. Verify actual precision JSON, `none` cache policy, fixed8 controls and positive/forbidden purpose counters. Repeat requests, truncate/replace and restart/cancel with no prefix reuse. Run existing FMA suffix-on/off seeded and forced-rejection gates. New RTLModel clears ambient selectors, so harnesses must configure the trusted per-process property explicitly.
4. Run the authored292-target corpus and nine free-running cases on the combined FMA profile versus013a; exact preservation of the FMA-only behavior is the target. Comparisons against FP64 deployment are cross-precision and remain descriptive. A passing tiny corpus is not broad quality assurance.
5. Before any optional deployment, use the actual4096 boundary gate with maximum scalar and operation5 attention, valid tail and overflow checks. The previous phase-only FP64-prefill capacity pass cannot substitute for the new fused prefill route.

## Measurements if FMA-only benefit survives

First compare OFF versus PREFILL_12 in one combined binary to isolate the worker-routing change while keeping arithmetic and resident pools fixed. Then compare FMA-only013a directly against combined PREFILL_12 to quantify the deployment composition and idle-pool/host overhead. Keep arithmetic-only FMA-versus-FP64 evidence separate; never multiply these ratios. Use alternating85/452×32 requests, direct warm-pool dumps, full-logit comparisons after **every** scored pair outside clocks, exact IDs/consumed positions and purpose-counter deltas. Report prefill, TTFT, decode, total, startup and memory separately. Twelve workers may not remain optimal after faster fused projections; the earlier FP64-prefill worker sweep is a hypothesis, not evidence for the composed kernel. No scoring is authorized by this design document.
