# Native prefill register tiling — November 13 experiment

The reopened CPU optimization pass targets long-prompt prefill. Live build18
remains unchanged. This isolated candidate changes only the paired GQA
probability–V reduction for ordinary prefill batches larger than one; it has
been rejected for general prefill after the shape comparison below; live code is unchanged.

The baseline traverses causal time first and repeatedly loads/stores both
128-channel output vectors. The candidate first divides each probability by
its already computed sum, then processes 32 channels at a time. Sixteen NEON
FP32 accumulators hold both heads' channel tile across the ascending-time loop;
outputs are stored once after that loop. Each output retains its original
positive-zero initialization, canonical FP16 V widening and ordered FP32 FMA
chain. Query dots, causal lengths, separate vector-exponential calls and scalar
sums remain unchanged, as do B1 decode, dedicated B2 verification and the
validated FP64 norm accumulator.

The hypothesis is fewer destination load/store instructions. Four strided KV
passes, probability rereads, cache-line use and register pressure can offset
that benefit. Tile32 was selected before timing; this is not a threshold fitted
to favorable workloads. Earlier BLAS attention failed numerical gates and
full-width FP32 KV scratch/threshold dispatch had mixed or negative timings;
those experiments remain rejected rather than silently reintroduced here.

## Reproducible source and correctness

The [isolated patch](../tools/experiments/native-prefill-v-register-tile.patch)
applies to the baseline core hash in the [experiment metadata](../reports/native-cpu-prefill-v-register-tile-experiment.json).
Use a separate source/build directory; applying it to production is not part of
this publication. The metadata's `build_run: false` and `inference_run: false`
describe source preparation at creation time. Subsequent execution is identified
by the [immutable build manifest](../reports/native-cpu-prefill-v-register-tile-manifest.json)
and [exact gate](../reports/native-cpu-prefill-v-register-tile-exact-gate.json).

All 56 complete vocabulary arrays matched build18 exactly: 28 evaluated positions,
two capture repeats, covering tiny/short/452 prompts, 511/512/513 boundaries and
1024/2048/3968 contexts. This is sampled exactness and transitive linkage to
build18's independent numerical evidence, not an exhaustive proof. Prefix,
suffix/lifecycle and browser acceptance remain separate requirements before a
release change.

## Compiled loop review

The [static assembly review](../reports/native-cpu-prefill-v-register-tile-assembly-review.json)
is bound to the candidate binary hash. All four causal tile-loop excerpts were
read and checked: each contains 16 FMLA instructions with distinct accumulator
registers, no output stores, no stack accesses and no calls inside its backedge.
Stores occur after the loops. The probability prepass uses FP32 division,
including vector FDIV and scalar tails, rather than reciprocal estimates.

The surrounding function still has stack bookkeeping. This review establishes
the intended register schedule in these loops, not zero stack traffic throughout
the function or measured speed. The completed eight-shape comparison is recorded below; no partial-run
selection was used.


## Completed shape comparison: reject general prefill use

The [full ten-pair eight-shape report](../reports/native-cpu-prefill-v-register-tile-original-shapes.json)
compares the immutable candidate against build18. Each shape passed exactness
before timing. The table reports median paired wall-time speedup, reference /
candidate, with 95% paired bootstrap intervals. Prefix priming is outside the
measured tail append; this is not full fresh-request timing for retained-prefix
shapes. All raw rounds, warmups and outliers remain in the archive.

| Retained prefix | Appended batch | Wall-time speedup (95% interval) |
|---|---:|---:|
| 0 | 2 | 1.01343 (1.00033–1.01545) |
| 0 | 16 | 1.00419 (0.98341–1.01066) |
| 0 | 85 | 1.01636 (0.99636–1.03413) |
| 0 | 452 | 1.03586 (1.00958–1.06462) |
| 3584 | 2 | 1.04936 (0.97191–1.11396) |
| 3584 | 16 | 1.23446 (1.06784–1.25202) |
| 3584 | 64 | 0.93035 (0.86238–1.01881) |
| 3584 | 384 | 0.85104 (0.83773–0.92053) |

The target long tail, 3584 retained tokens plus 384 appended tokens, regressed:
0.85104× means approximately 17.5% longer candidate latency, with its entire
interval below 1. The 64-token tail was uncertain and also had a slower median.
The 16-token tail and fresh 452-token append improved on these runs; fresh 2
barely excludes 1. Other intervals include 1. These unadjusted shape-specific
findings do not establish an overall benefit.

The candidate is rejected for general prefill use. Its exact 56-array result
remains valid, but exact arithmetic and spill-free inner loops did not establish
speed. No threshold is selected post hoc from the favorable shapes, and no
causal claim attributes the regression specifically to cache locality or
register pressure without further evidence. Live build18 remains unchanged.
The next investigation is baseline phase attribution, not promotion of this
candidate.


## Separate fixed time32/channel32 candidate

After the full-time channel tile's long-tail regression, a distinct design
bounds the active time span to 32 tokens before advancing through four 32-channel
tiles. It initializes accumulators to positive zero in the first time block;
later blocks reload the preceding FP32 output stores, preserving each output's
ascending-time FMA chain. The final block uses its actual causal remainder.
This fixed design was chosen before its measurement. It is neither a threshold
that selects favorable earlier shapes nor a claim that cache locality has been
proved to cause the first candidate's regression.

The [source patch](../tools/experiments/native-prefill-v-time32.patch),
[preparation metadata](../reports/native-cpu-prefill-v-time32-experiment.json), and
[immutable build](../reports/native-cpu-prefill-v-time32-manifest.json) identify the
isolated candidate. [56 full-logit arrays](../reports/native-cpu-prefill-v-time32-exact-gate.json)
matched build18 exactly, and [12 additional boundary arrays](../reports/native-cpu-prefill-v-time32-timeboundary-exact-gate.json)
matched at causal lengths 31/32/33 and 63/64/65 with two repeats. These are scoped
compiled exactness checks, not exhaustive correctness proof.

The [completed original eight-shape comparison](../reports/native-cpu-prefill-v-time32-original-shapes.json)
retains ten alternating pairs, warmups and all outliers. Median paired wall-time
speedup is reference/candidate, above 1 favoring the candidate; parentheses are
95% paired bootstrap intervals. Retained-prefix priming is outside the clock.

| Retained prefix | Appended batch | Wall-time speedup (95% interval) |
|---|---:|---:|
| 0 | 2 | 1.00646 (0.99781–1.01605) |
| 0 | 16 | 1.00226 (0.98083–1.00892) |
| 0 | 85 | 0.97780 (0.96799–0.98212) |
| 0 | 452 | 1.05276 (1.02219–1.08384) |
| 3584 | 2 | 1.03047 (0.98690–1.08794) |
| 3584 | 16 | 1.16202 (1.13195–1.21547) |
| 3584 | 64 | 1.22128 (1.17663–1.25062) |
| 3584 | 384 | 1.22362 (1.16487–1.32969) |

Long-prefix 16/64/384-token tails and fresh 452 improve with intervals above 1.
Fresh 85 regresses, with its entire interval below 1. This is a measured tradeoff,
not a general prefill win. Do not pool shapes or choose a new cutoff from this
table. Whole-request and held-out evidence, prefix/suffix lifecycle acceptance
and browser checks remain separate before any release decision. Candidate
acceptance is ongoing; live build18 remains unchanged.
