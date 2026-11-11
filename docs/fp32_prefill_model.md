# Experimental FP32 prefill profile

`--fp32-prefill-b8` additionally requires `--fp32-projection-b1 --fp32-verification-b4`. It builds a separate strict shortreal B8 archive with `--unroll-count 1`; B1 and B4 retain their existing compilation options. Stock main, attention and KV arithmetic remain FP64. Defaults do not change.

The explicit `fp32_b1_fp32_prefill` profile routes scalar projection operation 0 to FP32 B1, prefill operation 2 (counts 1–4) to FP32 B4, operation 3 (counts 5–8) to FP32 B8, and verification operation 4 to FP32 B4. Operation 1 remains FP64 attention. B8 uses four accumulation lanes; B1/B4 use eight. Thus changing prefill chunk boundaries can change floating-point results. Runtime advertises canonical prefix policy `none`; existing FP64-prefix cache policy must not be reused for this profile.

Initial artifact SHA256: `c87074825ab522ead128c121312d8c799205230d6f9235081ce261c1ad7a8449`. Transport tests cover counts 1–8, partition counts 1/3/17, grouped descriptors, CV/hybrid waits and failure lifecycle. Real full-vocabulary gates confirm B4 tails match sequential B1, B8 same-route replay across 8/3 workers, truncation/overwrite and restart. These are routing checks, not broad quality or performance evidence. Full corpus/conversation evaluation and timing remain separate gates.

Final linked B8 `Vfp32_prefill_kernel::evalNba()` begins at `0x100026158`. Its hot loop has 16 `fmul.2s` and 16 `fadd.2s` operations per four columns, direct BF16 loads, no calls, and no fused multiply-add instructions. No forced vector width flags are used. This matches the validated unroll1 strategy; it does not claim four-lane FP32 SIMD.

Run the opt-in real route test with `SV_FP32_PREFILL_BINARY=<artifact> python -m unittest tests.test_fp32_prefill_real`. The build uses the same selected fast flags as the hybrid profile plus the two options above, with `--shortreal-compiler` pointing at the reviewed compiler fork. Model weights remain external BF16; no GPU or host neural kernels are introduced.
