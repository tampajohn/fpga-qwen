This is an isolated **projection microkernel experiment**, not a model backend. `rtl/fp32_projection_kernel.sv` implements original SV arithmetic using real ports and the unchanged BF16-to-real DPI transport, with internal `shortreal` activations, weights, products and sums. All FP32 conversion happens in SV. B1 and B4 use eight independent partial sums; B8 uses four, matching the deployed worker's reduction layout. Widths must be divisible by eight and between 8 and 3,072; rows are bounded at 3,072. The harness validates the entire mapped weight span before starting workers.

The primary FP64 reference is the untouched production `projection_worker.sv`, independently built with the installed compiler, original O3/thin-LTO flags and original contraction behavior. It is never embedded in the patched-compiler executable or compiled with the candidate's strict rounding flags. The candidate uses the reviewed compiler's internal-shortreal subset and `-fno-fast-math -ffp-contract=off`. Generated C++ is inspected but never edited.

Real checkpoint matrices are layer-zero Q (slot4,2048×1024), gate (slot11,3072×1024) and down (slot13,1024×3072); tensor names, shapes and offsets are checked against the safetensors header. Vocabulary projection is not covered. Activations are deterministic synthetic doubles `(j*7+t*11)%61-30`, divided by17, identically supplied to both kernels. They are not captured model activations and supply no full-model quality evidence.

Each subprocess initializes independent worker models and inputs, performs an untimed warmup, then releases all workers for a fixed repetition count. Measured wall time includes one initial wakeup and worker eval loops; initialization, input copies and final output copies are outside the interval. FP32 candidate input conversion occurs inside every SV tick. This is warmed arithmetic throughput, excluding full-model dispatch barriers, attention, activations, KV and sampling. Different compiler versions, strict contraction, and the candidate's smaller output port (3072×B versus the production worker's151936 scalar outputs plus batch ports) can affect eval/cache behavior; a timing difference cannot be attributed solely to precision.

`tools/fp32_rounding_oracle.py` is a bounded test-only exact-rational IEEE binary32 oracle, not an inference kernel. It rounds each product/addition to nearest/ties-to-even independently of the compiler, with the same per-lane column order. Tiny checkpoint-slice fixtures use rows1/3/7 and widths8/16/24 for each batch1/4/8. Known halfway, subnormal, cancellation and multiplication cases have standalone tests. Negative-zero arithmetic and overflow are excluded from these bounded finite fixtures. Bad widths/rows must fail before worker execution. Actual kernels must match these oracle output bits; full-matrix comparisons against FP64 report descriptive error statistics without inventing a model-quality threshold.

Build and scoring are separate commands so they can be coordinated into different CPU windows:

```sh
python -m unittest tests.test_fp32_rounding_oracle
python tools/probe_fp32_projection.py --build-only --compiler-root REVIEWED_COMPILER_ROOT
python tools/probe_fp32_projection.py --run-only --workers 8 --repetitions 20 --pairs 3
```

The builder records executable/compiler/source hashes, the candidate compiler Git base and tracked patch hash, exact command/environment and disassembly; source/compiler changes during builds invalidate the artifact. Run-only rechecks source and executable hashes. The report records checkpoint/tokenizer provenance. The compiler binary hash identifies the actual compiler even when development files outside its tracked patch exist. No global installs, production defaults, or generated-code replacements are involved. Evidence from the subsequently approved bounded run is summarized below.

The bounded experiment is now complete with the reviewed compiler at `c607db4` (compiler binary SHA256 `70df52bf26b8852b199d5c8d4e94dd0514e0f499ae6d7ff786bfb297de05989f`). All nine exact-rational fixture combinations matched every output bit; fifteen bad-shape commands were rejected. Generated candidate storage is native `float`. Independent hot-function inspection found B1 `fmul.4s`/`fadd.4s`, B4/B8 `fmul.2s`/`fadd.2s`, and no FMA-family instructions in candidate `evalNba`; original FP64 baseline hot paths retain `fmla.2d`/`fmadd`. Activation narrowing appears before row loops, output widening after reduction, with no stack accesses inside inspected hot loops. B4/B8 use only two FP32 SIMD lanes and retain substantial integer address work. Assembly inventory counts are whole-binary evidence, not dynamic instruction counts; build scripts also retain full disassembly for reproduction.

The initial20-repetition report is explicitly exploratory: intervals were too short and showed scheduling outliers. A separate exclusive-window1000-repetition rerun used the same frozen binaries and produced these median candidate/FP64 throughput ratios:

| Batch | Q | Gate | Down |
|---|---:|---:|---:|
| 1 | 1.404 | 1.502 | 1.413 |
| 4 | 1.269 | 1.256 | 1.218 |
| 8 | 0.904 | 0.896 | 0.971 |

All measured B1/B4 pairs favored the candidate, while every B8 pair favored FP64. Timed intervals were at least19.9ms in the longer run. Maximum absolute projection errors across tested matrices were approximately1.2e-6–3.0e-6, relative L2 errors2.1e-7–4.8e-7; outputs were deterministic across repeats. No model-quality acceptance threshold is inferred from those projection errors. The complete longer report, exploratory report, oracle gates and SIMD inventory are in `reports/fp32-projection-*.json`.

These results support investigating isolated B1 FP32 integration while retaining deployed FP64 B8 and attention. They do not establish an end-to-end speedup or preservation of model quality. The unmeasured151936-row vocabulary projection and actual model activations must be addressed before a full-model candidate can be assessed.
