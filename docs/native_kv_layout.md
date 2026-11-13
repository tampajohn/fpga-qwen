# Head-major KV layout experiment

This isolated candidate changes the original native engine's physical KV
layout from `[layer][position][head][channel]` to
`[layer][head][position][channel]`. Live build18 remains unchanged. There is no
release decision; the completed shape timings below are scoped evidence.

For one attention head, consecutive positions move 256 bytes instead of 2048
bytes. This changes address strides; it is not a relevance index or sparse
attention scheme. Every causal position is still read. The engine retains full
FP16 K and V history: 28 layers × 4096 positions × 8 KV heads × 128 channels × 2 bytes
× 2 arrays = 469,762,048 bytes, **448 MiB**. There is no eviction, extra staging
copy, compressed history or altered context limit.

Only the central offset formula and both ordinary/B2 write loops change. Both
writes now explicitly scatter each head into its own position sequence. All
attention readers use the central offset. Canonical FP16 rounding, query/key
math, softmax lengths and sums, ordered value FMAs, FP64 norm accumulation and
logical reset/rewind/commit behavior remain unchanged. Future speculative KV
is still excluded by causal lengths and overwritten before subsequent use.

The hypothesis is improved locality or hardware prefetching when scanning one
head's history. Less contiguous writes across heads and concurrent workers'
access patterns can offset that benefit. These are hypotheses, not measured
causes or speed claims. This candidate does not combine the separate time32
probability–V tiling experiment.

## Reproducibility and completed gates

- [Source patch](../tools/experiments/native-kv-head-major.patch), to apply only
  to a separate copy of the baseline identified by the metadata.
- [Experiment metadata](../reports/native-cpu-kv-head-major-experiment.json)
  with baseline/candidate hashes and full-capacity calculation. Its initial
  source-preparation flags are historical; subsequent execution is recorded
  in the build and gate reports.
- [Immutable build manifest](../reports/native-cpu-kv-head-major-manifest.json).
- [56-array exact gate](../reports/native-cpu-kv-head-major-exact-gate.json): 28
  evaluated positions, two repeats, matching build18 full logits exactly.
- [Full 36-case B2 API matrix](../reports/native-cpu-kv-head-major-verify-api.json):
  worker counts 1/12/64, both commit lengths, rollback, boundaries and state/capacity
  checks passed.

The address mapping covers exactly the same allocation without overlap; the
last valid element remains `layers*heads*context*channels-1`. Source review and
sampled compiled exactness support the implementation, but are not universal
correctness proofs. The [completed acceptance record](../reports/native-cpu-kv-head-major-acceptance.json)
now links a passing [28-case prefix gate](../reports/native-cpu-kv-head-major-prefix-check.json),
[suffix/lifecycle gate](../reports/native-cpu-kv-head-major-suffix-trajectories.json)
with 35 initial and seven cached cases, and [nine-case quality comparison](../reports/native-cpu-kv-head-major-quality.json)
matching build18 exactly. These are no-new-regression checks, not improved
answer quality. The completed shape timings are recorded below. Broader request timing and
browser acceptance remain separate; the candidate is not promoted.


## Completed original-shape comparison

The [full eight-shape report](../reports/native-cpu-kv-head-major-original-shapes.json)
contains ten alternating pairs per shape against immutable build18, with exact
preflight checks before timing. Ratios below are median paired wall-time
speedups, reference/candidate, with 95% paired bootstrap intervals. Retained-
prefix priming is outside the measured append. All rounds and outliers remain
archived; workloads are not pooled.

| Retained prefix | Appended batch | Wall-time speedup (95% interval) |
|---|---:|---:|
| 0 | 2 | 0.99679 (0.98511–1.02146) |
| 0 | 16 | 0.99933 (0.98211–1.02380) |
| 0 | 85 | 1.07628 (1.07346–1.08078) |
| 0 | 452 | 1.08030 (1.05339–1.09260) |
| 3584 | 2 | 1.07048 (1.03628–1.09260) |
| 3584 | 16 | 1.08415 (0.96151–1.22117) |
| 3584 | 64 | 1.20635 (1.16923–1.24903) |
| 3584 | 384 | 1.06313 (1.03397–1.08234) |

These are per-shape observations on this host, not a general engine-throughput
claim or a direct comparison with llama. They do not isolate the contribution
of cache lines, hardware prefetch or TLB behavior. Whole-request acceptance and
release decisions require their own evidence; live build18 remains unchanged.

A separately proposed combination with fixed time32/channel32 PV blocking must
be checked and measured as a new candidate. Multiplying this table's ratios by
the time32 results would not measure that composition and is not permitted as
a performance conclusion.


## Combined fixed head-major/time32 candidate: exactness evidence

A separately built candidate combines the reviewed head-major offset and both
KV write changes with the fixed time32/channel32 paired-PV loop. Source review
confirmed exact composition without additional arithmetic, eviction or context
thresholds. The [combined patch](../tools/experiments/native-kv-head-major-time32.patch),
[experiment metadata](../reports/native-cpu-kv-head-major-time32-experiment.json), and
[build manifest](../reports/native-cpu-kv-head-major-time32-manifest.json) bind the
baseline, both component sources and new candidate. Their hashes were checked
before publication; source-preparation flags are historical, not execution status.

The combined executable passed [56 full-logit arrays](../reports/native-cpu-kv-head-major-time32-exact-gate.json)
plus [12 time-boundary arrays](../reports/native-cpu-kv-head-major-time32-timeboundary-exact-gate.json)
against build18: 68 arrays total, including repeated causal 31/32/33 and 63/64/65
checks. This preserves the tested per-output arithmetic and canonical KV
behavior; it is sampled exactness, not universal correctness.

Combined lifecycle/quality acceptance subsequently completed as recorded below.
No partial timing output is published here. Component speedups are not multiplied or
inherited, and no performance conclusion follows from these gates. The combined
candidate is not live; build18 remains the selected browser release.


### Combined candidate lifecycle and quality acceptance

The terminal prerequisite run completed successfully. Individual reports retain
their original binary/input/source bindings and passed statuses:

- [Full 36-case B2 API matrix](../reports/native-cpu-kv-head-major-time32-verify-api.json).
- [28-case canonical-prefix gate](../reports/native-cpu-kv-head-major-time32-prefix-check.json).
- [Suffix trajectories and cached lifecycle](../reports/native-cpu-kv-head-major-time32-suffix-trajectories.json), including cancellation/restart.
- [Direct suffix CLI checks](../reports/native-cpu-kv-head-major-time32-suffix-cli.json).
- [Nine-case quality](../reports/native-cpu-kv-head-major-time32-quality.json) and [18-case expanded quality](../reports/native-cpu-kv-head-major-time32-extended-quality.json), preserving reference trajectories and grades.

The checker printed successful prerequisite completion but does not produce an
aggregate acceptance JSON; these terminal individual reports are the evidence.
Exact baseline agreement does not improve model answer accuracy. The completed shape timing is recorded below as a separate measurement. No timing result or browser deployment is
inferred from acceptance, and live build18 remains unchanged.


## Combined candidate: completed original-shape comparison

The [full combined shape report](../reports/native-cpu-kv-head-major-time32-original-shapes.json) contains all eight predeclared shapes and all ten alternating pairs per shape against immutable build18. Exact full-array preflight preceded every timed shape; two warmups were outside scoring. Values below are median paired wall-time speedups (reference/candidate) with 95% paired bootstrap intervals. These are append-command timings, including the final head and sampling; retained-prefix priming is outside the measurement.

| Retained prefix | Appended batch | Wall-time speedup (95% interval) |
|---|---:|---:|
| 0 | 2 | 0.99989 (0.99509–1.01032) |
| 0 | 16 | 1.00255 (0.99357–1.00933) |
| 0 | 85 | 1.05722 (1.04433–1.08535) |
| 0 | 452 | 1.11124 (1.09194–1.12775) |
| 3584 | 2 | 1.05255 (1.04438–1.06784) |
| 3584 | 16 | 1.22053 (1.21235–1.24372) |
| 3584 | 64 | 1.24673 (1.18322–1.26967) |
| 3584 | 384 | 1.21741 (1.18976–1.26581) |

All 80 pairs and outliers remain archived. The two smallest fresh batches have intervals including1; the other six sampled shapes favor the combined candidate. These observations do not establish general throughput, a direct llama advantage, or isolated causality for either component. No component ratios were multiplied. Whole-request results and release selection remain separate; no partial extended timing is published here.

A [host-activity record](../reports/native-cpu-browser-build18-restart-nov13.json) documents a later build18 browser-server restart overlapping the ongoing whole-request comparison. The server accepted connections but returned empty responses; the cause was not diagnosed. Only the owned server was restarted, its stable build18 link stayed unchanged, and no browser model inference was requested. Router/tokenizer initialization still constitutes concurrent host activity. Preserve every timing sample and carry this caveat into whole-request reporting; do not attribute any outlier to the restart without evidence. This event is not retroactively attributed to the completed shape measurements.


## Combined candidate: complete whole-request comparison against build18

Both independent ten-pair suites are terminal: [standard32](../reports/native-cpu-kv-head-major-time32-standard-confirmation.json), [standard analysis](../reports/native-cpu-kv-head-major-time32-standard-analysis.json), [extended128](../reports/native-cpu-kv-head-major-time32-extended-confirmation.json) and [extended analysis](../reports/native-cpu-kv-head-major-time32-extended-analysis.json). Each used one warmup, alternating configuration order, 12 decode / 6 prefill threads, fresh logical prompts and natural EOS handling. Every scored visit preserved the required exact output trajectory. All20 visits per suite and every outlier remain included.

Ratios above1 favor the combined candidate: decode is candidate/reference throughput; TTFT and request speedups are reference/candidate time. Intervals are95% paired bootstrap intervals, unadjusted across these multiple metrics and contexts. These are direct measured ratios, not products of component or previous llama comparisons.

| Prompt tokens | Output cap | Decode ratio (95% interval) | TTFT speedup (95% interval) | Request speedup (95% interval) |
|---|---:|---:|---:|---:|
| 85 | 32 | 1.04555 (0.93644–1.08688) | 1.02452 (0.99744–1.05272) | 1.04422 (0.96676–1.06933) |
| 452 | 32 | 1.05204 (0.96415–1.07854) | 1.04754 (1.02599–1.07566) | 1.04071 (1.00486–1.08032) |
| 1024 | 128 | 1.11244 (1.01632–1.32652) | 1.10507 (1.06000–1.13554) | 1.10407 (1.04454–1.22201) |
| 2048 | 128 | 1.13528 (1.04793–1.23997) | 1.14151 (1.09643–1.16479) | 1.14752 (1.08451–1.17413) |
| 3968 | 128 | 1.26812 (1.13693–1.35084) | 1.16738 (1.15219–1.18233) | 1.18172 (1.16332–1.20352) |

At85 tokens, all three intervals include1; neither benefit nor equivalence is established. At452, decode remains uncertain while TTFT and whole-request intervals favor the candidate. At1024/2048/3968, all three intervals favor the candidate in this run. These context-specific improvements support selecting the combined candidate for release19 evaluation; they are not a universal engine-speed claim or a direct llama result.

Manual evidence review binds the timed executable SHA to the68-array exact matrix, API/prefix/suffix/CLI lifecycle gates and both unchanged quality suites. Build18's fixed-tolerance independent oracle qualification transfers at the sampled byte-identical positions; timing trajectory equality does not establish independent full-logit accuracy at every generated position. Quality remains4/9 and10/18, with historical wrong answers still wrong.

The [concurrent browser restart](../reports/native-cpu-browser-build18-restart-nov13.json) is retained as a host-activity caveat: router/tokenizer initialization overlapped the whole-request comparison, though no browser inference was requested. No samples are removed or outliers causally assigned to it. Shared-host scheduling/background work remains uncontrolled.

Root selected this candidate for release19 based on the complete direct-build18 evidence. It is not promoted by this publication: source/binary staging, browser gates and release operations remain separate. A fresh direct comparison with tuned llama is pending; no multiplication of historical ratios substitutes for it.
