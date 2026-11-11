The isolated FP32 B8 projection kernel was rebuilt with candidate-only Verilator `--unroll-count 1`. Original SV source/arithmetic/reduction order is unchanged. This prevents Verilator from unrolling the short token loops and scalarizing the partial-sum arrays before Clang sees them. The stock FP64 baseline retains its original compiler/flags; no full-model/default precision profile changes.

Generated storage changes from32 separately named partial floats to a fixed two-dimensional float array. It does **not** restore four-lane SIMD: the hot arithmetic remains16 `fmul.2s`/`fadd.2s` pairs per four columns, without FMA. The inner hot loop instead shrinks from97 static instructions (original addresses0x100008fb4–0x100009134 inclusive) to65 (candidate0x100009004–0x100009104), largely by reducing address/control work and using paired loads. These are static instruction counts, not dynamic profiling or proof of which cost dominates.

Three exact-rational tiny B8 fixtures passed. A three-way comparison measured full Q/gate/down matrices with8workers,1000repetitions and three alternating engine orders. All full-matrix FP32 outputs were bit-identical to the original FP32 kernel across every round.

| Matrix | Candidate / original FP32 median | Candidate / stock FP64 median | Candidate / FP64 range |
|---|---:|---:|---:|
| Q | 1.166 | 1.124 | 1.077–1.126 |
| Gate | 1.190 | 1.123 | 1.079–1.132 |
| Down | 1.266 | 1.186 | 1.150–1.227 |

Every measured round favored this candidate over both references. The complete report and manifests are in `reports/fp32-b8-unroll1.json`. These warmed synthetic-activation projection results exclude full-model prefill overhead and do not establish model-quality equivalence to FP64. FP32 prefill would require its own explicit canonical routing/cache policy and full-model quality validation.

Reproduce with the existing probe builder's optional `--candidate-unroll-count 1`, batch8 and a separate directory, then `tools/compare_prefill_projection.py` with independent stock FP64, frozen original FP32 and candidate binary paths. Keep builds/correctness separate from the exclusive scored window. No generated C++ edits or relaxed floating-point flags are used.
