# Hybrid full-model performance

The opt-in hybrid executable `d27a2ffc51aacad610dbef0f6b358e66118d486504bcb98128345e28f50b19d9` completed full-model CPU inference with FP32 scalar projections and FP64 prefill, attention and KV. This is the experimental binary, not the live default. Runtime precision capabilities were verified before timing. [Raw measurements](../reports/fp32-hybrid-matched-current.json) include binary hashes, generated-source hashes, prompts and per-request outputs.

Eight workers/threads; identical 85/452-token prompts; 32 greedy outputs; one initial request and two warm repetitions; empty logical KV each request. No GPU or PyTorch inference. No prefix reuse or speculative verification.

| Engine | Prompt | Warm median decode tokens/s | Warm median first-token seconds |
|---|---|---:|---:|
| rtl | short | 54.69 | 0.873 |
| rtl | long | 42.01 | 5.707 |
| llama-f16 | short | 139.12 | 0.229 |
| llama-f16 | long | 126.95 | 0.575 |
| llama-f32 | short | 84.69 | 0.129 |
| llama-f32 | long | 80.28 | 0.442 |

All initial and warm output token sequences matched across engines within each prompt. The separate [hybrid quality gates](fp32_hybrid_quality.md) also preserve all 292 teacher-forced top predictions and all nine conversation outputs, with deterministic same-route sampled replay. The conversation suite remains only 4/9 strict passes for both variants; these checks do not establish broad model quality.

The hybrid remains slower than both native references. Prefill is a major gap: the longer prompt takes 5.707 seconds to first output versus 0.575 for native FP16. The measured kernel gains therefore do not close the end-to-end gap.

This exclusive agent CPU window used sequential engines and only two warm samples. It is not an alternating paired ablation against our current FP64 backend. Compare against earlier measurements cautiously because host performance varies. Native and SV arithmetic/KV precision and transports differ as documented in the report. Do not infer a precise hybrid/FP64 speedup by dividing results from separate runs.

Reproduce with the locked environment: `python tools/benchmark_matched.py --engines rtl,llama-f16,llama-f32 --threads 8 --tokens 32 --repetitions 2 --profiles short,long --sv-binary HYBRID_BINARY --sv-lanes 8 --sv-block-size 1 --sv-prefill-size 8 --sv-precision-profile fp32_b1_fp64_prefill --output build/fp32-hybrid-matched-current.json`.
