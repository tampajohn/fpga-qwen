The isolated `--fp32-prefill-fma` option requires FP32 B8 prefill and existing B4 verification. It introduces a distinct profile, `fp32_b1_fp32_fma_prefill`, and leaves the deployed model, UI and defaults unchanged. Prefix reuse (including canonical reuse) is disabled. This starts from the combined-attention model, independently of the optional phase-pool experiment.

The route is explicit in existing operation numbers, not inferred from whether output logits are requested:

| Purpose | Operation | Archive | Arithmetic |
|---|---:|---|---|
| Scalar token/final prompt/decode and every vocabulary projection | 0 | Vfp32_projection_kernel | Separate-rounded FP32, eight column lanes |
| Prefill tail1..4 | 2 | Vfp32_prefill_tail_kernel (new) | Explicit fused FP32, eight column lanes |
| Prefill5..8 | 3 | Vfp32_prefill_kernel | Explicit fused FP32, four column lanes, retained loops |
| Speculative verification1..4 | 4 | Vfp32_verification_kernel | Separate-rounded FP32, eight column lanes |

Prefill commands produce no logits. The final prompt token is processed by the unchanged scalar path and its vocabulary projection remains separate-rounded FP32. Verification also uses the unchanged separate-rounded vocabulary projection. Attention, normalization, RoPE, nonlinearities and KV storage retain existing FP64 behavior.

Each worker owns a new fused-tail module/context; its padded activation copy and active result scatter use the existing stride8 coordinator layout and stride4 kernel layout. Group dispatch copies input once before processing descriptors. No host model math is added. Existing pool partitioning, validation, synchronous publication and lifetimes are retained. B4 verification never uses the fused-tail module.

All FP32 archives use reviewed compiler3a123151 under strict no-fast-math/no-contraction flags, while only the two prefill archives define `FP32_EXPLICIT_FMA`. Ordinary B1/B4 output equivalence under this compiler was demonstrated by the earlier microkernel gates. The new module compiler and its flags remain separate from the stock FP64 main/attention compilation. The builder rejects any other compiler hash for this first bounded artifact.

Required gates before any model timing: final-linked archive/oracle checks; same initial-KV B1/B4 verification equivalence to the separate model; teacher-forced292-target and nine free-running cases versus deployed combined and separate FP32-prefill references; fixed canonical segmentation reset/restart and suffix on/off, rejection, truncation/cancellation checks. Cross-rounding model differences are descriptive NLL/KL/logit measurements, not a fabricated exactness or arbitrary tolerance gate. Within the new fixed route, replay and suffix verification require exact bits. Arbitrary rechunking or comparing fused prefill with scalar feeding is not an exactness requirement because lane count and accumulation rounding differ.

The frozen full-model candidate is `013a56b2271120f6ca39cf38f74398b2d918e7d8c5e7dfe0b37679bd8317ec32`; the matched separately rounded reference is `ecc943d7390c3e84a156ac04f05d95ee8d98dc88049aac8f4a80d039bd305545`. Build manifests and all raw validation evidence are retained under `reports/fma-prefill-*` and `reports/fma-preserved-routes.json`.

Correctness driver 42742 exited successfully. The preserved-route gate compared 76 full-vocabulary distributions exactly after scalar steps, block-final outputs and scalar continuations from identical initial KV; this does not assert comparison of every intermediate block distribution. Another 24 verification cases passed, as did fixed-route tails 1–8, reset/restart and 14 suffix comparisons covering greedy/sampled acceptance, rejection, cancellation and retained-KV continuation. All three real test methods passed. Prefix reuse remains disabled.

Both original 292-target evaluations preserved every top-1 prediction. Against matched separate FP32 prefill, mean NLL delta was −7.73e−8 nats, mean KL 1.60e−12 and maximum logit difference 4.86e−5. Against deployed FP64 prefill, the corresponding values were −1.54e−7, 1.65e−12 and 5.15e−5. These are descriptive rounding differences, not evidence of improved quality. Both nine-case free-running comparisons produced identical token IDs/text and unchanged 4/9 strict passes; that limited pass rate remains a model-quality limitation.

Actual linked code retains separate multiply/add instructions in B1 scalar and B4 verification archives; only B8 prefill and the new B4 tail archive contain fused token-pair instructions. Tail output capacity is 3072×4 doubles, while the B1 vocabulary output capacity remains 151936. No full-model performance conclusion is made before the paired experiment. The comparator checks each profile's own full-logit reproducibility and requires matching fixed-count output IDs across profiles. Startup is measured separately in seconds using `time.perf_counter()` around model startup, precision validation and lane configuration. Direct closed-file dumps preserve warmed pools.

The full-model paired experiment completed successfully (driver 43485, exit 0). Each comparison used three alternating pairs for 85- and 452-token prompts with 32 generated tokens, eight workers, warmed resident pools and logical KV reset. All twelve pairs preserved output IDs/positions and each profile reproduced its own full-logit bits. Startup and post-request dumps were excluded from request timing. Raw evidence is in `reports/fma-prefill-separate-paired.json` and `reports/fma-prefill-deployed-paired.json`.

| Reference | Prompt | Median paired prefill ratio | TTFT ratio | Decode ratio | Total ratio |
|---|---:|---:|---:|---:|---:|
| Matched separate FP32 prefill | 85 | 1.249 | 1.230 | 0.965 | 1.131 |
| Matched separate FP32 prefill | 452 | 1.183 | 1.182 | 0.913 | 1.130 |
| Deployed FP64 prefill | 85 | 1.244 | 1.238 | 0.888 | 1.052 |
| Deployed FP64 prefill | 452 | 1.216 | 1.216 | 0.945 | 1.190 |

Ratios above one favor the FMA candidate; these are medians of per-pair ratios, not ratios of separately aggregated medians. All twelve prefill comparisons improved. All six matched-reference totals improved. Deployed-reference long totals improved in every pair, while short totals were mixed (1.081, 1.052, 0.980). Decode generally slowed despite unchanged scalar arithmetic; separate linking, additional module state, host scheduling and cache behavior remain possible influences, so this experiment does not isolate a causal explanation for that regression. One matched long decode ratio was 0.670. No worker placement, hard process isolation or OS cache eviction is claimed.

For scale, candidate prefill medians were 0.552/3.571 seconds in the matched-reference session and 0.591/3.535 seconds in the deployed-reference session. The separate sessions should not be pooled into a single baseline. This supports an optional faster-prefill experiment, especially on the longer prompt; it does not establish universal total-latency improvement, improved model quality or a reason to replace the deployed default. Cache reuse is still disabled, and fresh-request timings do not compare repeated-context cached service behavior.
