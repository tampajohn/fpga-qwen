# Contiguous B8 with four reduction lanes

This isolated probe is promising enough for a full-model experiment, not a deployed speed claim. It reuses each BF16 weight across eight positions using four FP64 reduction lanes (32 accumulators). Changing eight lanes to four changes floating-point association.

Three alternating-order pairs per shape, 100 evaluations per sample, eight workers, real checkpoint matrices and deterministic full-mantissa synthetic activations produced these median normalized throughput ratios against the existing eight-lane B4 kernel: Q 1.357×, O 1.272×, gate 1.262×, down 1.306×. All values were finite; maximum absolute projection error was 4.885e-15, below the predeclared 1e-10 threshold. Outputs are **not bit-identical**. The width-nine fallback remained exact but its tiny timing is not representative.

[Raw evidence](../reports/wide-projection-b8-four-lanes.json) preserves every pair, build flags and hashes. The first Q sample has a large outlier; the reported median does not erase it. Timing excludes worker creation, warmup, copies, barriers and all nonprojection model work. Both branches share one probe executable, so shared generated setup may differ from the standalone production worker. Warmed kernel ratios cannot predict full-model gains.

[Generated assembly](../reports/wide-projection-b8-four-lanes-evalNba.s): the candidate hot loop at 0x100009d5c–0x100009e14 loads four contiguous BF16 weights, widens each, and executes sixteen vector FP64 multiply-add instructions. The loop has no stack accesses; paired activation loads advance 0x100 bytes and weights advance eight bytes. Other branches in the same function have stack traffic and must not be confused with this loop.

Reproduce with `uv run --locked python tools/probe_wide_projection.py --batch 8 --four-lanes --repetitions 100`. The model protocol and default remain unchanged. Full vocabulary, causal-cache, held-out quality and end-to-end gates are required before promotion.
