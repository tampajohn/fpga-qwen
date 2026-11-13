# Four-hour CPU optimization pass — November 13

User-authorized timebox: **17:47:31–21:47:31 UTC, November 13, 2026**
(12:47–16:47 America/New_York). Baseline is live build19 at commit
`f1b30beec15ff484c81d524d20728b7532a5f701`.

## Current status

**The pass is active; no new release has been accepted or deployed. Live build19
remains unchanged.** The numerical contract still requires maximum absolute
error ≤0.01, RMS ≤0.002, relative tolerance zero and greedy identity. A newly
covered 63-token prompt fails those limits in build19 and the original 64/128
chunk candidates. That absolute numerical gap remains unresolved.

| Workstream | Current evidence and disposition |
|---|---|
| Original 64/128 canonical chunks | Completed 80-position inventory fails four comparisons: prompts 63 and 64 in both variants. Matched baseline19 controls have identical full vectors on both failed trajectories. Old 28 positions pass per variant; cached/fresh qualification remains pending. Absolute failures remain unresolved. |
| Direct norm/QK division | Short boundaries pass, but 512/128/64 each independently fail the 2048 absolute gate. Rejected; no performance qualification. |
| Split-K128 projections | Both 63/64 positions fail unchanged limits. Rejected before broader gates or timing. |
| B1-order batched projections | Pass 63/64 with sequential-control byte identity, then fail 2048 absolute error. Broader run stopped; 3968 not executed. No timing or release qualification. |
| Original-arithmetic QK key pairing | 68 repeated full-logit arrays match build19 exactly. Completed shape timing shows selected long-prefix gains, with the inherited absolute gap explicitly retained. No release or whole-request/llama claim. |
| Adjacent query-row pairing | Exact68 and API/prefix/suffix/9+18 quality gates passed relatively. [First complete ten-pair request comparisons](native_query_row_pair.md#first-completed-whole-request-comparisons-against-build19) favor long-context TTFT and the3968 request in this run; short-request and all decode intervals include1. Every noisy sample retained; separate quiet repeat planned. Actual-boundary/seeded controls pending; no promotion or absolute numerical pass. |

Arithmetic-changing experiments are not advanced to timing after failed gates.
Separately authorized shape measurements of candidates with sampled byte identity
to build19 are **conditional diagnostics**, not absolute numerical qualification.
They may characterize performance without claiming to fix the inherited failure
or creating a release exception. All negative results remain in the record.

The sections below preserve the evidence in investigation order. Statements
about an earlier hold or partial suite describe that stage, not current blanket
prohibitions. Detailed numerical traces and rejected experiments are retained in
[the numerical diagnostic record](native_prefix_numerical_diagnostics.md).

## Explicit release-policy revision: baseline nonregression

The coordinator has explicitly revised the earlier self-imposed requirement
that every newly declared absolute NumPy gate pass before any release. The
user's objective is faster inference while maintaining quality. For strictly
arithmetic-preserving optimizations only, a release may instead qualify through
complete, binary-bound baseline-nonregression evidence, with the known absolute
accuracy failures disclosed and tracked separately. This decision is made within
the authorized optimization work; it is not an inference that failed oracle
checks passed.

**The absolute limits remain unchanged:** maximum error 0.01, RMS 0.002, rtol 0,
and greedy identity. The 63/64 failures remain failures for both baseline19 and
any byte-identical candidate. No release under this policy may claim to repair
them or be labelled independently numerically qualified. Arithmetic-changing
canonical 64/128 and B1-projection experiments remain subject to absolute gates;
they cannot use this narrower baseline-equivalence path.

Eligibility requires a source-level account of each output's unchanged operands,
FMA/reduction order, causal bounds and state ownership, plus compiled inspection
and measured full-vector identity. Source reasoning alone cannot establish
compiled equivalence. Finite regression coverage also does not prove equivalence
for every possible prompt; this residual risk and the independent oracle gap
must remain visible in release notes.

### Release-readiness checklist — regression scope only

These are required evidence categories, **not a claim that all are complete**.
At this revision, the original 68-array identity and source/assembly reviews are
complete for query-row pairing; the coordinator is collecting the remaining
final-candidate evidence. Any subsequent composition requires evidence tied to
that composed executable rather than transitive assumptions.

- [ ] Freeze the exact release executable, build manifest, sources, weight pack,
  runtime helper configuration and declared defaults; verify hashes before and
  after checks. Bind every report to that final binary.
- [ ] Preserve independent source/compiled review of per-output arithmetic,
  causal extra-row bounds, odd-row fallback and worker scratch ownership.
- [ ] Pass original 68-array exact regression coverage and the **actual newly
  failing 63/64 fixtures**, both teacher positions and repeats. Equal context
  lengths on other prompts do not substitute for these fixtures.
- [ ] Complete API, prefix cache and suffix verification checks, including reset,
  reuse, edits, rewind, cancellation, session changes, odd tails, multiple chunks
  and near-capacity contexts. Pass the frozen 9-case and extended 18-case quality
  gates; retain every failure or partial result.
- [ ] Compare complete sampled streams against build19 with identical seeds at
  multiple contexts, including 20-token streams. Require IDs, termination and
  all captured full logits to agree; a single sampled prediction is insufficient.
- [ ] Complete ten alternating pairs of matched **whole requests**, retain all
  samples and report uncertainty plus short/long-context tradeoffs. Shape-only
  gains do not satisfy this requirement or establish a llama comparison.
- [ ] Validate localhost browser selection/defaults and benchmark locking against
  the final release, preserve selectable build19 rollback, and publish precise
  release notes distinguishing regression qualification from unresolved absolute
  accuracy failures.

The additional seeded-stream checker is prepared locally at
`build/native-cpu/check-seeded-stream-regression-v1.py`; no result is asserted
here. It compares baseline and candidate through the actual chat sampler at
three fixed contexts, two seeds and two repeats, with 20 requested output tokens,
full-logit recording and identical EOS behavior. Separate lifecycle and quality
gates remain necessary. A regression, missing prerequisite or incomplete final
binary binding blocks release under this policy.

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

This discovery initially paused new optimization designs and performance work
while numerical investigation proceeded. Later, separately authorized shape
measurements resumed only for original-arithmetic candidates with sampled
byte identity to build19, as recorded below. No tolerance change, candidate
promotion or broad numerical pass followed. The original 64/128 remaining
boundary suite is still incomplete. The prior 28-position baseline evidence
remains valid within its own fixtures; the new prompt reveals its coverage limit.

## Isolated QK key-pair candidate: exactness and assembly evidence

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
traffic reduction against baseline nor a performance gain by itself. Timing
was initially held; the later completed conditional shape comparison is recorded
below. Promotion remains unapproved and live build19 is unchanged.

## Direct division diagnostics: initial selected-position results

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
These finite observations did not prove a general root cause, broad numerical
qualification, better answers or speed. Subsequent broad testing rejected the
combined variant at 2048, as summarized below and detailed in the numerical
diagnostic record. Earlier build19 and norm-only failures remain part of the
record; no tolerance was relaxed.

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

- Prompt 63: [512](../reports/native-cpu-norm-attention-direct-div-v1-boundary63-oracle.json),
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
numerical testing subsequently exposed a 2048 regression in all three division
variants. The completed broad 512 reports and independent 64/128 controls are
archived in the numerical diagnostic record; their failures are not superseded
by this short-boundary matrix. No speed claim or promotion follows. Live build19
is unchanged.

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
proposed replacement, or performance evidence. The subsequent
[stagewise analysis](../reports/native-cpu-native63-stage-analysis-v1.json)
verified that instrumentation preserved original final bytes and localized the
first native batch/sequential difference to layer 0 Q/K/V projection outputs
after identical normalized inputs. This is an observation of differing rounding,
not proof of an implementation defect; the full trace and qualifications are
in the numerical diagnostic record.

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
evaluates every layer 0 Q/K/V output row (4096 total), using the same captured
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

## Adjacent query-row pairing: source and exactness evidence

An isolated [original patch](../tools/experiments/native-prefill-query-row-pair.patch)
pairs two adjacent token rows and their two GQA heads during ordinary batched
prefill. Four independent queries share canonical FP16 K/V loads and conversions
across the common causal prefix; only the second row receives its extra final
key/value. Each query keeps its original channel/time reduction order and its
own softmax call at its own causal length. An odd final row uses the original
paired-head path. B1 decode and B2 verification are unchanged. This candidate
starts from original-arithmetic build19, without the two-key QK or direct-division
experiments.

PV uses fixed time32/channel16 tiles with sixteen independent vector accumulators.
Six additional per-worker/context FP32 scratch arrays cost 1.125 MiB at default
12-worker/4096-context capacity, or 6 MiB at64 workers. The weighted task map
covers `ceil(batch/2)*8` items. [Source metadata](../reports/native-cpu-prefill-query-row-pair-experiment.json),
[independent source/integer-boundary review](../reports/native-cpu-prefill-query-row-pair-source-review.json)
and [executed build](../reports/native-cpu-prefill-query-row-pair-manifest.json)
preserve provenance.

The completed [56-array gate](../reports/native-cpu-prefill-query-row-pair-exact-gate.json)
and [12-array boundary gate](../reports/native-cpu-prefill-query-row-pair-timeboundary-exact-gate.json)
pass byte identity against captures bound to the released build19 executable.
[Assembly review](../reports/native-cpu-prefill-query-row-pair-assembly-review.json)
confirms the four-query QK loop shares one Half key vector across eight independent
FMA accumulators; the common-prefix PV loop shares two Half value vectors across
sixteen accumulators. These inner loops have no stores, stack accesses or calls.
Surrounding tile boundaries store outputs and the function uses stack state;
no whole-function spill-free or performance claim follows.

The baseline19 independent 63-token NumPy failure remains inherited and unresolved
by byte identity. This is fixture-scoped arithmetic preservation, not absolute
numerical qualification or promotion. The completed conditional shape comparison follows; live build19 is unchanged.


## Rejected arithmetic changes and the remaining numerical gap

The combined direct-division 512 variant completed the original 28-position
matrix with a failure at 2048: maximum absolute error 0.012651443 exceeds 0.01,
although RMS 0.001877199 and greedy identity pass. Separate 128 and 64 controls
reproduce that failure; their full broad matrices were not completed. Their
passing short-boundary matrix does not qualify them. All eight additional 512
reports and both finer controls are linked in the
[numerical diagnostic record](native_prefix_numerical_diagnostics.md).

The matched-input projection study motivated one fixed split-K128 experiment,
retaining original norm and attention arithmetic. It also failed: at 63, maximum
error 0.012241602 / RMS 0.002866313; at 64, maximum 0.010024786 / RMS 0.002008769.
The [failed gate](../reports/native-cpu-prefill-splitk128-v1-boundary63-oracle.json)
is preserved. Local projection accuracy did not predict a passing network gate.
No broader qualification or timing followed, and no further arithmetic changes
are justified solely by these measurements.

## Original 64/128 numerical inventory: complete with inherited failures

The [completed inventory](../reports/native-cpu-original-prefix-qualification-v2.json)
contains 30 trajectories and 80 prediction positions across the original 128/64
variants. Both variants pass their original 28-position matrix and the eight
positions from new prompts 65/127/128/129. Each fails the first prediction on
new prompt 63 and new prompt 64; their teacher-forced next predictions pass.
That is 76 passing and **four failing comparisons**, not an aggregate pass.
All greedy predictions agree with the independent oracle.

| Fixture, both variants | First prediction max absolute / RMS | Teacher step max absolute / RMS | Absolute result |
|---|---:|---:|---|
| Prompt 63 | 0.012255669 / 0.002722482 | 0.009222031 / 0.001818819 | First position fails both limits |
| Prompt 64 | 0.009551048 / 0.002240883 | 0.004672050 / 0.001060937 | First position fails RMS |

Prompt 64 is a fresh 64-token prefill, distinct from the teacher step after a
63-token prefill. Their equal final context length does not imply identical
prefill arithmetic. The new [baseline19 control](../reports/native-cpu-build19-original-prefix-control-boundary64-oracle.json)
reproduces the prompt 64 failure. Its [capture](../reports/native-cpu-build19-original-prefix-control-boundary64-capture.json)
and [control summary](../reports/native-cpu-original-prefix-baseline-controls-v1.json)
show both original chunk variants' full trajectories are byte-identical to
baseline19. Together with the earlier 63 control, every failing trajectory in
this inventory has a matched inherited-failure control. This establishes the
observed failures are shared; it does not waive them or prove equivalence on
other inputs.

The [archive manifest](../reports/native-cpu-original-prefix-qualification-v2-archive.json)
indexes all 30 numerical reports, all 30 two-repeat captures, the summary and
baseline controls. The [inventory runner snapshot](../tools/experiments/run-original-prefix-qualification-v2.py.txt)
and [dynamic baseline-control runner snapshot](../tools/experiments/run-original-prefix-baseline-controls-v1.py.txt)
are byte-identical text archives; restore their original `build/native-cpu/`
locations before use. The inventory resumed only missing outputs, preserved
completed failures, and returned `completed=true, passed=false`. The dynamic
control selected newly failed trajectories except the already-controlled 63
fixture. JSON archives anonymize local paths while preserving original hashes;
raw full-vocabulary arrays remain local.

Only the original canonical-chunk constant changes are covered here; the rejected
direct-division candidates are separate. Saved NumPy vectors from division
boundary experiments remain unchanged independent references, not candidate
logits. No threshold was relaxed.

The isolated real-history harness can next evaluate original 512/128/64
cached/fresh full-logit identity and cross-variant IDs/text/EOS. Quality, cache
lifecycle and complete-request measurements remain additional requirements.
Completing this numerical inventory does not authorize a release, backend default
change or performance claim; the absolute numerical gap is still unresolved.

## B1-order batched projection diagnostic: initial pass, broader failure

The [isolated patch](../tools/experiments/native-prefill-b1-arithmetic-projection-diagnostic.patch)
replaces only batched layer SGEMM projections with the existing two-input,
two-output-row verifier dot kernels. Worker ownership is bounded to each input
pair; an odd input uses original dot4/dot. Attention, normalization, canonical512,
B1 decode, B2 verification and output-head computation retain build19 arithmetic.
The existing FP32 weight cache is intentionally retained to avoid a storage change.
See [preparation metadata](../reports/native-cpu-prefill-b1-arithmetic-projection-diagnostic-experiment.json),
[independent source/integer-boundary review](../reports/native-cpu-prefill-b1-arithmetic-projection-diagnostic-source-review.json),
and [executed build](../reports/native-cpu-prefill-b1-arithmetic-projection-diagnostic-manifest.json).

The completed [63-token capture](../reports/native-cpu-prefill-b1-arithmetic-projection-diagnostic-boundary63-capture.json)
has two exact repeats. Both full-vocabulary outputs at positions63/64 are
byte-identical to the earlier sequential-step diagnostic, despite retaining the
batched attention path. Its [fixed-oracle gate](../reports/native-cpu-prefill-b1-arithmetic-projection-diagnostic-boundary63-oracle.json)
passes: maximum absolute errors0.007441878 /0.008217812, RMS0.001454748 /0.001817284,
and greedy identity. This isolates the projection-path change on this fixture
more narrowly than the earlier all-sequential control; it does not prove a
library defect, a general cause of network error or equivalence for every chunk.

The subsequent [2048/2049 capture](../reports/native-cpu-prefill-b1-arithmetic-projection-diagnostic-context2048-capture.json)
and [fixed oracle gate](../reports/native-cpu-prefill-b1-arithmetic-projection-diagnostic-context2048-oracle.json)
completed with a failure at 2048: maximum absolute error 0.011130333 exceeds 0.01,
while RMS 0.001872640 passes. The teacher step 2049 passes at maximum 0.005331993,
RMS 0.001022315. Greedy IDs match at both positions. The earlier 63/64 success
does not override this failure, and the arithmetic-changing diagnostic is not
qualified for timing or release. No limit changed.

The broad runner stopped at this first numerical failure. **The 3968 trajectory
and subsequent broad fixtures were not executed.** Before this run, the initial
script had stopped during preflight because it named a nonexistent 3968 oracle;
no model execution began in that attempt. The
[preflight correction record](../reports/native-cpu-b1-projection-broad-preflight-correction.json)
binds the original and corrected runner hashes. The correction selects the
existing independent `context3968-native-llama-oracle-control.json` vectors;
it changes no numerical limit. Both the
[original runner](../tools/experiments/run-b1-projection-diagnostic-broad.py.txt)
and [corrected runner](../tools/experiments/run-b1-projection-diagnostic-broad-v2.py.txt)
are archived byte-for-byte as text and require restoration to their original
`build/native-cpu/` paths for reproduction. A corrected preflight is not a
completed 3968 result.

This narrows one fixture's projection-path diagnosis but supplies no general
arithmetic fix. No further arithmetic design or performance claim follows from
this partial broad matrix; original baseline failures remain in the record.


## Prepared regression and prefix-diagnostic runners

The [prepared-only source manifest](../reports/native-cpu-prepared-seed-prefix-gates-v1.json)
archives the seeded-stream checker, hardened original-prefix evaluator, offline
analyzer and three-config selection. Each snapshot is byte-identical to its
local source and has passed syntax checking and independent read-only review.
**No execution result is asserted by this archive.** Restore snapshots to the
manifest's original paths before running, since repository-root resolution
assumes their original directory depth; use fresh output directories.

For the final selected arithmetic-preserving executable, the seeded check runs
three contexts × two seeds × two repeats × baseline/candidate. It requests 20
output tokens, respects EOS, requires at least two full 20-token contexts per
seed, and compares full logits, IDs and termination. It freezes sources,
executables, pack and fixtures, then rechecks inputs and generated dump hashes.
Run from the repository root, replacing `CANDIDATE_BINARY` with the selected path:

```sh
.venv/bin/python tools/benchmark_guard.py --name 'Selected candidate seeded streams' -- .venv/bin/python build/native-cpu/check-seeded-stream-regression-v1.py --candidate CANDIDATE_BINARY --directory build/native-cpu/selected-seeded-streams-v1
```

Original 512/128/64 prefix **diagnostic gates** perform 54 requests across 18 process
visits. They compare cached/fresh full logits including EOS, cross-variant
IDs/text/termination, reuse counts and a fixed 9-turn rubric. Completed turns
and arms survive a later failure; helpers, build sources and generated dumps
are hash-bound. These are reuse/regression observations, not absolute numerical
qualification. The offline summary deliberately excludes elapsed-time and
throughput fields and retains `absolute_numerical_passed=false` and
`release_eligible=false`.

```sh
.venv/bin/python tools/benchmark_guard.py --name 'Original prefix cache diagnostic gates' -- .venv/bin/python build/native-cpu/prefix-granularity-eval/evaluate.py --configs build/native-cpu/prefix-granularity-eval/configs.json --mode gates --directory build/native-cpu/original-prefix-cache-gates-v1
.venv/bin/python build/native-cpu/prefix-granularity-eval/analyze-gates.py --report build/native-cpu/original-prefix-cache-gates-v1/report.json --output build/native-cpu/original-prefix-cache-gates-v1-summary.json
```

The evaluator retains its generic confirmation mode, but timing confirmation
and reported speed for the arithmetic-changing 64/128 candidates are **not
eligible under current numerical evidence**. The approved CPU-engine brief
requires their fixed absolute gates before reported timing; inherited failures
do not satisfy that prerequisite. This archive provides no confirmation-timing
command or performance claim. The separate arithmetic-preserving release policy
above does not extend to these candidates.
