# B8 lane-major projection probe: rejected layout

The isolated B8 projection layout preserves output bits but regresses the wider hidden projections. It is rejected before any model protocol or browser changes. **This rejects the tested lane-major implementation, not B8 batching in general.** B4 remains the model/default; B16 was not tested because the smaller experiment did not justify expansion.

## Existing work and constraints

Ordinary prefill already sends no-logits blocks and skips the vocabulary projection entirely. The final pending token gets one vocabulary projection for the first response token. There is no repeated vocabulary projection to remove from normal no-logits prefill. Speculative verification needs predictions at each tested position; skipping predictions after rejection would require a distinct early-stop verification protocol.

The current B4 implementation hardcodes four-token ports, activation/result strides, masks, block state, wire validation, and host limits. A full B8 change would touch each of those and enlarge working buffers. It must retain causal KV visibility and preserve B4 sampled verification. This experiment deliberately does none of that: `WIDE_BATCH` adds probe-only worker ports and an operation used only by a standalone driver, with no new full-model wire command.

The candidate computes one reduction lane at a time, visiting columns lane, lane+8, and so on, then adding completed lane sums in the original lane 0→7 order. Each weight is reused across four or eight input positions. The source has B lane accumulators and B totals rather than 8×B simultaneous partials. Every output's ordered arithmetic remains identical to the baseline. However, source-level variable scope does not guarantee the compiler keeps only that many physical registers live.

## Exactness and projection-only measurements

The driver uses real checkpoint matrices and deterministic synthetic activation fixtures, eight worker threads, 100 repeated evaluations per timed sample, and three alternating-order pairs per shape. Initialization, per-thread warmup, worker creation, and output copies are excluded. This measures warmed projection arithmetic throughput without the full model's dispatch barriers, attention, KV, or prefill behavior. Repeated matrices may remain cache-resident; these numbers are not end-to-end speed predictions.

The B8 ratio is normalized for twice as many output positions: baseline B4 time ×2 divided by candidate B8 time. Full output bit patterns match the baseline, including a seven-row, width-nine fallback fixture. The fallback's tiny workload is not representative of Qwen's hidden matrices.

| Projection shape | B4 lane-major / existing B4 | B8 lane-major throughput / existing B4 |
|---|---:|---:|
| Q: 2048 × 1024 | 0.638× | 1.041× |
| O: 1024 × 2048 | 0.554× | 0.730× |
| Gate: 3072 × 1024 | 0.649× | 0.993× |
| Down: 1024 × 3072 | 0.497× | 0.642× |

Reports retain individual pair timings, hashes, and build options: [B4 probe](../reports/wide-projection-b4.json), [B8 probe](../reports/wide-projection-b8.json).

## Assembly findings

The generated B8 path uses SIMD FP64 multiply-adds, so it did not accidentally become a wholly scalar kernel. But the compiler unrolled the lane sequence and delayed much of the final addition chain until after the individual lane loops. In the archived B8 `evalNba`, the candidate begins at address `0x1000098f4`; examples `str q0,[sp,#0x70]` and `stp q3,q4,[sp,#0x50]`, followed by reloads near `0x100009d40`, show retained partial values spilling to the stack. Thus the attempted source-level reduction in live accumulators did not fully translate into the intended register allocation.

The layout also makes eight strided passes rather than one contiguous pass. The candidate activation pointer advances by `0x200` bytes between columns within a lane. These are plausible contributors to the measured loss; the measurements do not isolate a precise fraction of time attributable to each.

Archived generated code: [B4 evalNba](../reports/wide-projection-b4-evalNba.s), [B8 evalNba](../reports/wide-projection-b8-evalNba.s). These excerpts contain both the unchanged baseline worker and probe branch, so stack references elsewhere in the function must not all be attributed to the candidate.

## Read-only prioritization for the existing FP64 kernel

The current contiguous B4 hot loop already uses paired activation loads and vector `fmla.2d`, reusing each widened weight across four positions. The observed conversion sequence is a BF16 halfword load, shift, integer-to-FP register move, and FP32-to-FP64 conversion before those FMAs. That conversion is required by the chosen storage/arithmetic formats, but instruction counts alone cannot establish its wall-time cost.

Activation addressing in the hot loop is already largely pointer-based after the earlier masked-index optimization. Replacing this with more lookup metadata is unlikely to remove substantial work. Similarly, row-base multiplication and width checks are largely outside the most frequently repeated operations; broad shape-specialization is not justified by these excerpts alone.

The next promising question is whether exact small-tile weight widening can be amortized without the previously measured cost of a full expanded weight cache, or whether the independent lower-precision SIMD experiment provides a larger gain with acceptable quality. Either requires measured evidence. No additional FP64 kernel variant is introduced here, and no relaxed arithmetic or dense-weight skipping is proposed.

## Reproduction

The rejected worker branch and standalone helper are archived, not installed in
the active tree. First follow the [exact-base patch instructions](../tools/experiments/wide-projection-reproduction.md).


`uv run --locked python tools/probe_wide_projection.py --batch 4 --repetitions 100` and the same command with `--batch 8` build the isolated probe, check all output bits, time alternating pairs, and archive disassembly. The driver is a benchmark fixture, not a runtime implementation or a hardware estimate.
