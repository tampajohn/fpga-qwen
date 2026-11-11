Full-vocabulary FP32 projection and linking isolation were tested before full-model integration. The B1 SV kernel now supports an explicit151936-row real output port; the host validates rows, aligned width and the complete checkpoint span. The vocabulary tensor is slot2,151936×1024, checked against the checkpoint header. All conversions and neural arithmetic remain in SV.

The initial non-LTO candidate archive retained strict FP32 but could not inline the BF16 transport function. The actual hot loop contained a `bl _weight_value_unchecked` for every weight. Three paired100-repeat probes measured median throughput ratios0.118 for Q and0.109 for full vocabulary versus untouched production FP64. This linking choice is rejected; the result is archived in `reports/fp32-vocabulary-no-lto.json`.

The actual mixed-link candidate compiles only the FP32 model archive using the reviewed compiler and strict `-fno-fast-math -ffp-contract=off -flto=thin`. The C++ transport and installed Verilator runtime are compiled/linked with original default-contraction O3/thin-LTO flags. This preserves per-operation rounding while allowing transport inlining. Stock and patched `verilated.h`, `verilated_types.h` and `verilated_threads.h` hashes match exactly. Only one stock runtime is linked; the FP32 archive contains model objects, not a second runtime.

Independent final-artifact inspection confirms BF16 loads/expansion directly in the FP32 hot loop, `fmul.4s` followed by `fadd.4s`, and no branch-and-link or FMA instructions anywhere in that FP32 eval function. Final stores widen FP32 results to the real output ABI. The output array contains151936 doubles and partitioned writes remain row-disjoint. The separate stock FP64 reference retains its original FMA instructions. Three tiny exact-rational FP32 fixtures passed through this actual mixed link, not merely an isolated compiler test.

Three exclusive-window1000-repeat pairs measured median Q throughput1.509× and full-vocabulary1.427× versus production FP64. Full-vocabulary maximum absolute error versus FP64 was1.29e-6; numerical outputs were repeatable. These are projection-only results on synthetic activations, not model-quality acceptance or predicted end-to-end gains. Complete pairs/manifests are in `reports/fp32-vocabulary-mixed-lto.json`.

Reproduction:

```sh
python tools/build_fp32_mixed_probe.py --compiler-root REVIEWED_COMPILER_ROOT
python tools/probe_fp32_projection.py --run-only --batches 1 --vocabulary --matrices q,vocabulary --workers 8 --repetitions 1000 --pairs 3 --directory "$PWD/build/fp32-vocab-mixed"
```

The rejected variant uses the general builder with `--build-only --batches 1 --vocabulary --candidate-no-lto` and a separate directory. Build and scoring windows must remain separate. Generated C++ is never edited and production defaults are untouched.
