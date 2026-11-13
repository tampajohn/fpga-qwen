# Four-hour CPU optimization pass — November 13

User-authorized timebox: **17:47:31–21:47:31 UTC, November 13, 2026**
(12:47–16:47 America/New_York). Baseline is live build19 at commit
`f1b30beec15ff484c81d524d20728b7532a5f701`.

## Deliverables

1. Evaluate finer canonical prefix reuse (64/128 versus 512 tokens) on real
   multi-turn conversations, including cached/fresh correctness and full-request
   latency. Smaller prefill batches must not inherit numerical qualification
   from build19 without evidence.
2. Investigate the remaining long-context prefill/first-token latency deficit
   against tuned llama with an independently developed CPU implementation.
3. Deliver any validated improvements through the localhost browser, preserving
   rollback, latest/default selection, benchmark locking, and quality. Publish
   complete evidence and negative results to GitHub.

Parallel agents prepare independent source candidates, evaluation tools, and
reviews. Root exclusively runs builds, model execution, and benchmarks, serialized
under the browser-visible benchmark guard. Shared runtime sources stay frozen
while measurements that bind them are running.

## Time allocation and stop rule

- First 30 minutes: inspect and prepare isolated candidates and targeted gates.
- Until 20:32 UTC: iterate only where evidence supports the next experiment.
- After 20:32 UTC: start no new optimization designs; select or reject existing
  candidates and finish matched measurements.
- After 21:17 UTC: reserve the remaining time for browser validation, deployment,
  documentation, publication, and clearing owned benchmark locks.
- By 21:47:31 UTC: stop this pass and report what was delivered and what remains
  unresolved. Never promote an incompletely validated candidate to meet the clock.

CPU-only Qwen3-0.6B remains the target. No GPU/PyTorch inference or copied llama
kernels. Preserve the fixed independent NumPy limits (max absolute error 0.01,
RMS 0.002), finite full logits and greedy/quality gates. Timing claims require
matched complete work, ten alternating pairs, uncertainty intervals, and all
samples retained. Do not infer complete-request or llama gains from microbenchmarks
or by multiplying separately measured ratios. Build19 remains the fallback.

## Initial work allocation

- Prefix source design: isolated 64/128 candidates and arithmetic implications.
- Evaluation: isolated adapter support and real-history multi-turn fixtures.
- Attention source review: one evidence-driven candidate for long prefill.
- Root: numerical gates, serialized measurements, selection, merge and release.

Status: active; no post-build19 candidate has been accepted or deployed.


## First numerical results: new boundary failure also affects baseline19

Isolated canonical 64/128 candidates change only the shared prefix/prefill chunk
constant; they retain full causal history and FP16 KV. Source preparation flags
in the metadata are historical, while the build manifests record executed builds:

| Candidate | Source and provenance | Completed near-4K numerical evidence |
|---|---|---|
| 64 | [Patch](../tools/experiments/native-prefix64-v1.patch), [metadata](../reports/native-cpu-prefix64-v1-experiment.json), [build](../reports/native-cpu-prefix64-v1-manifest.json) | [3968/3969 gate](../reports/native-cpu-prefix64-v1-context3968-oracle.json): pass |
| 128 | [Patch](../tools/experiments/native-prefix128-v1.patch), [metadata](../reports/native-cpu-prefix128-v1-experiment.json), [build](../reports/native-cpu-prefix128-v1-manifest.json) | [3968/3969 gate](../reports/native-cpu-prefix128-v1-context3968-oracle.json): pass |

Both near-4K reports have maximum absolute errors 0.007285118 / 0.007415295 and
RMS errors 0.001512432 / 0.001223014 at the two positions, with greedy identity.
Those passes do not qualify other prompts or establish general numerical safety.

The new [63-token boundary fixture](../reports/native-cpu-prefix-granularity-boundary63-oracle-v1.json)
**failed** at position 63 for both candidates under unchanged max-absolute 0.01 /
RMS 0.002 limits. The [independent baseline19 control](../reports/native-cpu-build19-newboundary63-control-oracle.json)
produced the same errors on the same prompt and continuation:

| Position | Maximum absolute error (both candidates and baseline19) | RMS error | Fixed gate |
|---|---:|---:|---|
| 63 | 0.012255669 | 0.002722482 | Fail: both limits |
| 64 | 0.009222031 | 0.001818819 | Pass |

All greedy predictions matched. Baseline agreement does not override the
failed full-logit contract, and the later passing position does not erase the
failure. The control establishes that this fixture also exposes a baseline19
gap; it does not establish the cause or make either candidate acceptable.

Performance claims and new optimization designs are held while numerical
investigation proceeds. No tolerance change, candidate promotion or broad gate
pass is claimed. Remaining boundary-suite outputs are not aggregated or published
as complete. The prior 28-position baseline evidence remains valid within its
own fixtures; this newly tested prompt extends coverage and reveals its limit.

## Isolated QK key-pair candidate: exactness only, performance held

The [original source patch](../tools/experiments/native-prefill-qk-keypair.patch)
processes two causal key rows together for the existing pair of query heads in
ordinary batched prefill. Each of the four dot products keeps its own two FP32
vector accumulators, ascending channel FMA order and final reduction. Query
loads are shared; odd causal tails retain the original helper. B1 decode, B2
verification, softmax, PV reduction and normalization are unchanged. See
[source metadata](../reports/native-cpu-prefill-qk-keypair-experiment.json) and
[executed build manifest](../reports/native-cpu-prefill-qk-keypair-manifest.json).
The metadata's unexecuted flags describe source preparation, not the subsequent
completed build and checks.

The [56-array repeated trajectory gate](../reports/native-cpu-prefill-qk-keypair-exact-gate.json)
and [12-array causal-boundary gate](../reports/native-cpu-prefill-qk-keypair-timeboundary-exact-gate.json)
passed with full-logit byte identity against the prior combined candidate whose
executable is byte-identical to released build19. These cover the existing tiny,
85/452, 511/512/513, 1024/2048/3968 trajectories and causal lengths
31/32/33/63/64/65. This is fixture-scoped equivalence, **not an independent NumPy
pass**: the newly discovered baseline19 63-token numerical failure above remains
unresolved. Identical causal lengths do not imply identical prompt fixtures.

[Static assembly inspection](../reports/native-cpu-prefill-qk-keypair-assembly-review.json)
confirms the intended inner channel loop: four FP32 query-vector loads, two
FP16 key-vector loads, four widening conversions and eight independent vector
FMAs per iteration. That loop has no stores, stack references or calls. The
surrounding function does store pointer/state values on the stack, so this is
not a whole-function spill-free claim. It establishes neither a measured load
traffic reduction against baseline nor a performance gain. Timing and promotion
remain held while the independent numerical gap is investigated; live build19
is unchanged.

## Direct division diagnostics: four selected positions pass only in combined variant

Two isolated arithmetic diagnostics retain FP64 accumulation of FP32-rounded
squares and unchanged tolerances. Neither is deployed or performance-qualified.
Both patches below are relative to build19, not cumulative patches.

- **Norm only:** [patch](../tools/experiments/native-norm-direct-div-v1.patch),
  [source metadata](../reports/native-cpu-norm-direct-div-v1-experiment.json),
  [build](../reports/native-cpu-norm-direct-div-v1-manifest.json). RMSNorm divides
  each FP32 activation by the rounded FP32 square-root denominator before the
  separate weight multiply, replacing multiplication by its rounded reciprocal.
- **Norm plus attention:** [patch](../tools/experiments/native-norm-attention-direct-div-v1.patch),
  [source metadata](../reports/native-cpu-norm-attention-direct-div-v1-experiment.json),
  [build](../reports/native-cpu-norm-attention-direct-div-v1-manifest.json).
  Additionally, all three QK score paths—ordinary paired-head prefill, B1 decode,
  and B2 verification—divide directly by FP32 `sqrt(D)`. Dot-product order,
  causal bounds, softmax, PV and history semantics are unchanged. Source review
  confirmed all three paths and the recorded source hashes; this review does
  not independently certify emitted machine instructions.

| Diagnostic | Position | Maximum absolute error | RMS error | Fixed gate |
|---|---:|---:|---:|---|
| Norm only | 63 | 0.009491801 | 0.002047516 | **Fail: RMS** |
| Norm only | 64 | 0.009421587 | 0.001981484 | Pass |
| Norm + attention | 63 | 0.008892775 | 0.001824594 | Pass |
| Norm + attention | 64 | 0.009775162 | 0.001842675 | Pass |
| Norm + attention | 3968 | 0.008239746 | 0.001451200 | Pass |
| Norm + attention | 3969 | 0.005151749 | 0.001036625 | Pass |

All listed greedy predictions match. The completed
[norm-only boundary report](../reports/native-cpu-norm-direct-div-v1-boundary63-oracle.json)
preserves its failure. The combined variant passes the same
[boundary fixture](../reports/native-cpu-norm-attention-direct-div-v1-boundary63-oracle.json)
and two [saved near-4K positions](../reports/native-cpu-norm-attention-direct-div-v1-context3968-oracle.json).
These finite observations do not prove a general root cause, broad numerical
qualification, better answers or speed. Broad numerical, B2, prefix/lifecycle,
quality and performance acceptance remain required. Earlier build19 and
norm-only failures remain part of the record; no tolerance was relaxed.

## Direct-division composition: short boundary matrix completed

The direct norm/QK division arithmetic was composed with canonical chunks 64 and
128, alongside its original chunk512 variant. The 64/128 source metadata binds
both the arithmetic and granularity components; no QK key-pair optimization was
included. Prepared-only flags remain historical; executed builds and terminal
gates below supersede them for these specific checks.

| Chunk | Composed source provenance | Executed build |
|---|---|---|
| 64 | [Metadata](../reports/native-cpu-norm-attention-direct-div-prefix64-v2-experiment.json) | [Manifest](../reports/native-cpu-norm-attention-direct-div-prefix64-v2-manifest.json) |
| 128 | [Metadata](../reports/native-cpu-norm-attention-direct-div-prefix128-v2-experiment.json) | [Manifest](../reports/native-cpu-norm-attention-direct-div-prefix128-v2-manifest.json) |
| 512 | [Arithmetic metadata](../reports/native-cpu-norm-attention-direct-div-v1-experiment.json) | [Manifest](../reports/native-cpu-norm-attention-direct-div-v1-manifest.json) |

The completed [boundary matrix](../reports/native-cpu-div-prefix-v2-boundary-gate.json)
passes all **36 comparisons**: six prompt lengths (63, 64, 65, 127, 128, 129),
two teacher-forced positions each, across three chunk configurations. Native
captures repeat twice with exact repeat identity. The fixed limits remain
max-absolute 0.01, RMS 0.002, relative tolerance zero, plus greedy identity.

All eight constituent numerical reports are retained:

- Prompt63: [512](../reports/native-cpu-norm-attention-direct-div-v1-boundary63-oracle.json),
  [128](../reports/native-cpu-div-prefix-v2-128-boundary63-oracle.json),
  [64](../reports/native-cpu-div-prefix-v2-64-boundary63-oracle.json).
- All three chunks at prompt lengths
  [64](../reports/native-cpu-div-prefix-v2-boundary64-oracle.json),
  [65](../reports/native-cpu-div-prefix-v2-boundary65-oracle.json),
  [127](../reports/native-cpu-div-prefix-v2-boundary127-oracle.json),
  [128](../reports/native-cpu-div-prefix-v2-boundary128-oracle.json), and
  [129](../reports/native-cpu-div-prefix-v2-boundary129-oracle.json).

This qualifies only the listed boundary fixtures. The original v1 chunk
candidates, baseline19 and norm-only failures remain unchanged above. Broad
numerical and quality checks, cached/fresh lifecycle equivalence and performance
remain separate requirements; no partial broad-suite outputs are archived here.
No speed claim or promotion follows from this matrix. Live build19 is unchanged.

## Baseline19 sequential-prompt diagnostic

The unchanged build19 binary also consumed the same 63 prompt IDs one at a time
through `step`, ignoring intermediate predictions, followed by the same fixed
teacher-forced continuation. The completed [capture](../reports/native-cpu-build19-sequential63-diagnostic.json)
contains two bit-identical repeats. The [saved-oracle comparison](../reports/native-cpu-build19-sequential63-diagnostic-oracle.json)
passes unchanged limits at positions63/64: maximum absolute errors
0.007441878 / 0.008217812, RMS 0.001454748 / 0.001817284, and greedy identity.
The corresponding ordinary batched-prefill control still fails at position63.

This narrows the observed difference to evaluation paths on this fixture; it
**does not prove an SGEMM bug**. Sequential processing changes both projection
execution and attention execution, along with intermediate rounding propagated
through layers. It is not canonical prefill, a cache-conformance result, a
proposed replacement, or performance evidence. Stagewise comparison is needed
to isolate where the paths diverge.

The exact diagnostic [helper snapshot](../tools/experiments/native-sequential-prompt-capture.py.txt),
[patch](../tools/experiments/native-sequential-prompt-capture.patch), and
[provenance](../reports/native-cpu-build19-sequential63-helper-provenance.json)
are retained. The snapshot is archived as text: its import-root calculation
expects the original `build/native-cpu/numerical-diagnostics/` location, which
must be restored when reproducing it. Source-only flags in helper provenance
refer to preparation; the terminal reports establish execution. No engine or
shared capture helper was changed by this diagnostic publication.


## Matched-input projection roundoff diagnostic

The completed [local projection study](../reports/native-cpu-matched-projection-roundoff-v1.json)
evaluates every layer0 Q/K/V output row (4096 total), using the same captured
FP32 normalized input at zero-based position62 and the same stored FP16 weights.
Its diagnostic reference forms each product in FP64 and reduces all1024 products
with `math.fsum`. It is not a replacement network oracle, a changed acceptance
limit or a model inference run. The [exact helper snapshot](../tools/experiments/analyze-matched-projection-roundoff.py.txt)
retains its original location-dependent imports and is archived as text.

| Tensor | Rows | SGEMM RMS error | B1 dot RMS error | NumPy matvec RMS error |
|---|---:|---:|---:|---:|
| q_projection | 2048 | 7.57166603e-07 | 1.76109722e-07 | 8.03470344e-08 |
| k_projection | 1024 | 7.96927272e-07 | 2.0275783e-07 | 7.9777722e-08 |
| v_projection | 1024 | 1.26186197e-07 | 5.92250517e-08 | 2.55833279e-08 |

For these three complete tensors, captured native SGEMM has larger RMS error
against this higher-precision local reference than either captured B1 dot or
NumPy matvec on the identical input. This isolates projection roundoff for this
input, but does not establish a library defect or explain all later network
error. Different FP32 reduction orders legitimately differ; NumPy may select a
different BLAS algorithm. Large ULP counts on near-zero outputs must be read
alongside absolute and RMS errors, not treated as catastrophic relative error
or a new pass/fail threshold. The report preserves full ULP histograms, rows,
input identity and source/artifact bindings. No timing or promotion claim is made.


## Original-arithmetic QK key-pair shape comparison

The [completed eight-shape comparison](../reports/native-cpu-prefill-qk-keypair-original-shapes.json)
retains all ten alternating pairs per shape, warmups, outliers and provenance.
Ratios below are baseline19/candidate command wall latency; greater than1 means
lower candidate latency. Confidence intervals are paired percentile bootstrap
intervals for each shape separately, not pooled evidence.

| Shape (prefix/new tokens) | Median paired speedup | 95% interval |
|---|---:|---|
| P0-B2 | 0.99561 | [0.98366, 1.00396] |
| P0-B16 | 0.99815 | [0.98398, 1.00944] |
| P0-B85 | 1.00277 | [0.98999, 1.01980] |
| P0-B452 | 1.02832 | [0.98723, 1.03765] |
| P3584-B2 | 1.01495 | [0.98924, 1.04534] |
| P3584-B16 | 1.04074 | [1.01764, 1.08904] |
| P3584-B64 | 1.06509 | [1.05145, 1.09517] |
| P3584-B384 | 1.04929 | [1.03829, 1.08457] |

The long-prefix B16/B64/B384 intervals exceed1; other shapes do not establish a
gain in this run. This measures the original-arithmetic QK candidate with its
68-array byte-identity evidence, not the later direct-division composition.
It is neither a whole-request nor a llama comparison. Both baseline19 and this
arithmetic-preserving candidate retain the known independent 63-token numerical
gap; relative timing and baseline identity do not confer absolute numerical
qualification. No candidate was promoted from this result.
