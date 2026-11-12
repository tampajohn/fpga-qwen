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
