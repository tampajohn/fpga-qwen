# Head-major KV layout experiment

This isolated candidate changes the original native engine's physical KV
layout from `[layer][position][head][channel]` to
`[layer][head][position][channel]`. Live build18 remains unchanged. There is no
published timing result or release decision yet.

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
correctness proofs. Prefix, suffix/lifecycle and conversation-quality checks
are ongoing. Timing and browser acceptance remain separate; no partial report
is published and the candidate is not promoted.
