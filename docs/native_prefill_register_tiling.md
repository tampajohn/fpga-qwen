# Native prefill register tiling — November 13 experiment

The reopened CPU optimization pass targets long-prompt prefill. Live build18
remains unchanged. This isolated candidate changes only the paired GQA
probability–V reduction for ordinary prefill batches larger than one; it has
not been promoted and has no published timing result yet.

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
the function or measured speed. The first guarded eight-shape comparison remains
in progress at publication; partial samples are not archived or interpreted.
