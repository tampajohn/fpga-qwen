# Isolated SV W8A8 microkernel

This candidate is rejected for promotion: it is correct against an independent integer oracle, but slower than the deployed FP64 projection worker on the tested resident tiles. It does not change model execution, defaults, or checkpoint files. It does not establish that all INT8 implementations are slower.

## Reproduction and boundaries

Starting from commit `418a305`, run from this worktree:

```sh
python3 -m tools.w8a8_probe prepare
python3 -m tools.w8a8_probe build
python3 -m tools.w8a8_probe check
python3 -m tools.w8a8_probe bench
```

The cached Qwen3-0.6B BF16 checkpoint is required. Generated artifacts are under ignored `build/w8a8`. The fixture manifest fingerprints every exact input. Build uses Verilator 5.054, Apple Clang 21, O3, native CPU and ThinLTO. The production `rtl/projection_worker.sv` is instantiated unchanged with unsigned indices, masked activations, eight reduction lanes, batched prefill and block unroll. Its original BF16-bit-expansion callback is implemented as transport in the fixture driver. No Python/C++ quantization, normalization, dot product, rescaling or error computation is used.

SV prepares activations from four checkpoint embedding rows and layer-zero input RMSNorm for the Q fixtures. The 3072-wide down-projection fixtures use real weights but synthetic activations; they are not captured model states. Dimensions 7/8/63/64/65/1024/3072, B1/B4 and zero groups exercise tails and rounding. Only the first 16 output rows are tested. The zero fixture uses four rows. Host timing includes eval dispatch, just as the comparator does; neither path includes file loading, fixture activation construction or worker-thread scheduling.

## Arithmetic and correctness

Weights and activations use symmetric signed INT8 in [-127,127], with a separate FP64 absmax/127 scale per 64-feature group. Zero groups have scale one and integer zeros. SV implements round-to-nearest, ties-to-even, with explicit saturation and signed extension before multiplication. Tail groups are zero-padded. Group sums use signed INT32; the maximum magnitude is 64×127² = 1,032,256, safely below overflow. Rescaling and accumulation over groups remain SV FP64, with explicit `(real(sum) * weight_scale) * activation_scale` order.

An independent SV oracle recomputes scales and quantization using signed truncation/remainder rather than magnitude/floor, accumulates in signed INT64, and checks every quantized value, group sum and rescaled result. Known-answer tests cover positive/negative ties, saturation and zero. All 17 fixtures passed. The unchanged production worker also matched the probe's eight-lane FP64 outputs exactly on all fixtures; this is a comparator check, not a claim that quantization preserves FP64 outputs.

First-layer Q slice error relative to FP64 was 0.5646% relative RMSE at B1 and 0.8586% at B4; maximum absolute errors were 0.00411 and 0.01262. These are local projection errors only. No perplexity, held-out text, full-model logits or chat quality have been measured for W8A8.

## Code generation and bounded timings

The original 3072-byte arrays left per-access generated bounds checks. Padding quantized rows to 4096 and using explicit 12-bit feature indices removed these guards without changing integer results. Neither variant produced SDOT/UDOT or vector widening integer multiply-accumulate instructions. The native target includes dotprod and i8mm; feature availability alone did not make Verilator's generated signed-extension/multiply expressions vectorize. The remaining integer dots are scalar. The recorded `smull` occurrences are scalar register arithmetic, not vector MACs.

Three warm repeats of 1000 invocations each gave:

| Resident tile | W8A8 quantize + dot + rescale | Production FP64 worker | FP64 time / W8A8 time |
| --- | --- | --- | --- |
| 16×1024, B1, real Q inputs | 8.1–10.3 µs | 5.2–5.9 µs | 0.51–0.73 |
| 16×1024, B4, real Q inputs | 25.0–30.8 µs | 9.0–9.8 µs | 0.32–0.36 |
| 16×3072, B1, synthetic inputs | 17.5–20.7 µs | 8.2–8.6 µs | 0.40–0.49 |
| 16×3072, B4, synthetic inputs | 59.2–61.7 µs | 19.6–21.3 µs | 0.32–0.36 |

These are bounded exploratory timings in fixed phase order, not randomized full-model benchmarks. Weight preparation is separately reported and excluded from warm execution because reusable weight bytes/scales are prepared once. Activation preparation is included, not hidden behind a dot-only number. The tiny output-row count overweights activation quantization: a real 2048/3072-row projection would amortize it over many more rows. Conversely, these resident tiles cannot demonstrate reduced DRAM traffic for full weights. Even dot-only cost and the scalar code generation give no basis to promote this implementation.

A useful next experiment would need an SV representation that actually emits packed integer SIMD, followed by realistic row counts and identical transport boundaries. Full-model adoption would additionally require held-out likelihood/logit and behavioral quality gates. The existing nine-case chat suite is insufficient to validate quantization. No full-vocabulary preparation/cache or full-model integration was attempted.
