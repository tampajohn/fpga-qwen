# Projection optimization experiments

All variants retain BF16 checkpoint weights and SystemVerilog `real` (FP64)
arithmetic. Neither the new compiler flags nor eight accumulators make this FP16
arithmetic. A local Verilator 5.054 probe of `shortreal` emitted `SHORTREAL` warnings
and generated `double` ports; simply substituting that type would mislabel precision.

## Reproduce a candidate

```sh
uv run --locked tools/build_projection_variant.py --storage scalar --lto --unchecked
uv run --locked tools/projection_probe.py \
  build/variants/scalar-O3-lto-unchecked/full/Vqwen_behavioral
```

Build outputs are isolated under `build/variants/`, leaving the exact-order
reference executable in `build/full/` untouched. The chat fast profile uses the
selected variant; restart chat after rebuilding and validating that variant. Supported storage variants are `scalar` and `row`. `--opt O3|Os` chooses
an explicit `OPT_FAST`, avoiding Verilator's default `-Os` overriding the intended
hot-code optimization. `--lto` enables Clang thin LTO in all generated model,
harness, and link steps. Build logs record the actual compile commands.

`lanes 1` retains the original dot-product accumulation order. `lanes 8` uses eight
independent SystemVerilog accumulators, combines their results in fixed order,
and has a scalar fallback for widths not divisible by eight. This exposes
instruction-level/vector parallelism, but changes floating-point association.
The simulator acknowledges either command with `configured`. Changing the lane
count reconstructs worker models and leaves main-model KV state intact.

`--unchecked` only removes the **per-weight** bounds check from projection worker
reads. `ProjectionPool.run` first validates the entire matrix span against the
mapped checkpoint; this covers regular projection dispatch and the benchmark
command. General model/norm/embedding reads retain their checked accessor. Shape
checks bound row and column arithmetic before dispatch. Workers cannot index
outside that validated span.

## Short probe results

Five teacher-forced tokens `[9707, 11, 1917, 0, 9707]`, eight workers, same Apple
M5 Max, first forward separated from four steady forward timings. Every logit is
compared with the original serial executable, including prefix positions.
These are exploratory timings, **not** the matched llama.cpp benchmark or a broad
quality evaluation.

| Variant | Accumulators | Median steady tok/s | Max absolute logit difference |
| --- | ---: | ---: | ---: |
| Scalar O3, no LTO | 1 | 6.58 | 0, bit-identical |
| Scalar O3, no LTO | 8 | 7.01 | 2.10e-13 |
| Row transfer O3 + LTO | 1 | 8.20 | 0, bit-identical |
| Row transfer O3 + LTO | 8 | 15.10 | 2.10e-13 |
| Scalar O3 + LTO | 1 | 12.01 | 0, bit-identical |
| Scalar O3 + LTO | 8 | 16.68 | 2.10e-13 |
| Scalar O3 + LTO, checked once | 1 | 12.72 | 0, bit-identical |
| Scalar O3 + LTO, checked once | 8 | 28.89 | 2.10e-13 |

All five greedy choices matched for every variant. The final candidate used about
2,966 MiB RSS after the probe; no resident expanded-weight cache was introduced.
The final candidate SHA256 was
`59b1a6939c1b60aad08a55f742418909f8c4c5738d3d889fb4aa0e5f02ca8b29`.
Raw records: `build/projection-probe-final.json`,
`build/projection-probe-comparison.json`, and `build/projection-probe.json`.

Row transfer amortizes DPI calls but Verilator emits temporary-array copies per
row; it was not the best candidate. A separate resident-dynamic-array prototype
hit unsupported associative-array/ref-task constructs in this Verilator version.
That unbuildable prototype was removed rather than advertised as a supported mode.

Further claims must use the matched throughput/latency and quality harness, with
benchmarks scheduled without competing model jobs. The 28.89 figure does not
establish a win over llama.cpp, and these remain behavioral simulator timings,
not predicted FPGA performance.

## Attention, buffer, and bounds-check iteration

The next candidate is built independently with:

```sh
uv run --locked tools/build_projection_variant.py --lto --unchecked \
  --masked-activation --fast-attention --compact-projection
```

`--masked-activation` pads the worker activation port to 4096 entries and masks
indices to 12 bits. Existing validated projection widths are at most 3072, so
this does not alter a legal index or consume padding. Inspection of generated
C++ confirms direct `activation[index & 4095]` accesses, eliminating the prior
per-lane `index <= 3071 ? activation[index] : 0` guards. Neural arithmetic stays
in SV. `--compact-projection` separates the 3072-element intermediate output
buffer from the vocabulary-sized final output, avoiding large temporary clears
and copies for intermediate projections. `--fast-attention` normalizes attention
scores once and traverses value accumulation in contiguous channel order.

Five-position exploratory probes, eight workers and eight accumulators:

| Additional options over scalar O3 + LTO + checked-once | Steady tok/s |
| --- | ---: |
| Fast attention + compact projection | 32.09 |
| Above + masked activation, CV scheduler | 42.50 |
| Above + masked activation, hybrid scheduler | 40.95 |
| Above + masked activation, `-ffp-contract=fast`, CV scheduler | 37.91 |

These are separate short probes, not repeated matched comparisons. All candidates
matched the five reference greedy choices, with maximum absolute logit error
2.096e-13 for eight accumulators. Lane one was bit-identical for the first three;
explicit fast contraction changed lane-one rounding (max error 1.075e-13).
The selected candidate keeps the CV scheduler and default contraction. Hybrid
waiting remains an opt-in experiment (`--pool hybrid`). No fast-math or unroll
experiment was measured. Raw records are local ignored artifacts
`build/projection-probe-round2.json` and `build/projection-probe-round3.json`.

The historical second-iteration candidate was subsequently rebuilt to include the final sampler
optimization; that rebuilt binary SHA256 is
`59139c1ff40c0df262387cd3379ab518a1262fa68b3a893f9dc15291185b062d`.
The builder writes `manifest.json` with source hashes, commands, toolchain
versions, and binary hash, and rejects sources changing during a build.
That artifact was evaluated in the [second-iteration report](performance_iteration_2.md);
the unsigned-index successor is covered by the [paired third-iteration report](performance_iteration_3.md). The short
probe does not establish performance relative to llama.cpp or real FPGA hardware.

### Assembly inspection of the second-iteration binary

Read-only disassembly of the second-iteration binary confirms four ARM NEON `fmla.2d`
instructions per eight-column step, with BF16 expansion through `shll.4s` and
`fcvtl`/`fcvtl2`. Thin LTO eliminated inner-loop DPI calls, and partial sums stay
in SIMD registers. It is already executing vector FP64 arithmetic; simply
turning on contraction is not an unexplored route to SIMD.

The inner loop still performs repeated signed-index extension and masked address
calculations. The first four BF16 values also load through several fragments and
a `zip1` instruction, whereas the next four use a single 64-bit load. These are
observations, not measured percentages of runtime. The implemented
`--unsigned-indices` option addresses that index overhead. The separate
`--cpu native` compiler option remains untested. Unsigned variables rely on
existing host validation of nonnegative dimensions and matrix spans.

## Unsigned indices and fixed KV arrays

The implemented build options preserve the original reference path. `--unsigned-indices`
uses unsigned row/column loop variables; `--static-kv` replaces deque-backed SV
dynamic arrays with fixed SV arrays of exactly 117,440,512 doubles each. The
combined KV storage remains 1,792 MiB. Both compile successfully. Existing live
binaries were preserved, and all candidates have separate build paths.

One warm request per profile, 32 greedy output tokens, eight workers/accumulators:

| Candidate | Short (85 prompt tokens), tok/s | Long (452 prompt tokens), tok/s |
| --- | ---: | ---: |
| Current selected baseline, same session | 30.396 | 13.704 |
| Fixed KV only | 32.086 | 14.117 |
| Unsigned projection indices only | 35.377 | 15.443 |
| Unsigned indices + fixed KV | 29.193 | 14.651 |

These runs were sequential, not interleaved, and noticeable host/run drift was
observed versus earlier measurements. Fixed KV is not established as a win.
Unsigned-only is the candidate for an interleaved AB/BA evaluation, not a confirmed
speedup. No native-CPU experiment was run. Its binary SHA256 is
`becad933e07b45cc027717ae8f04b842c2db46d32b5a5b3e0bc736a6e85cb07e`.

The unsigned-only candidate had bit-identical full-vocabulary logits against the
current eight-accumulator baseline over five continuations after the 85-token
prompt. The combined candidate was also bit-identical after the 452-token prompt.
The initial fixed-KV test against a one-accumulator reference had max error
1.226e-13 and matched all five greedy IDs. Subsequent paired performance, exact-logit, quality and KV replacement gates
passed for unsigned-only; see the [third-iteration report](performance_iteration_3.md).
Unsigned-only is now the chat default; fixed KV remains experimental.

Read-only disassembly explains a plausible unsigned-index benefit: the hot
8-column loop shrank from approximately 40 instructions to 19, removing repeated
signed extension/masked address arithmetic and using contiguous paired BF16
loads. FP64 SIMD and arithmetic order remain unchanged. This is a compiler result,
not proof that the entire model speeds up by the same ratio.

Raw ignored artifacts: `build/static-kv-exploratory.json`,
`build/static-kv-baseline-exploratory.json`,
`build/unsigned-indices-exploratory.json`, and
`build/unsigned-static-kv-exploratory.json`.
