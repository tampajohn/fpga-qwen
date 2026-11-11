# Bounded W8A8 sign-interpretation variant

This isolated followup replaces SV signed-byte extension in the dot with the equivalent expression `unsigned_byte - ((unsigned_byte & 128) << 1)`. A 256-value exhaustive SV gate checks it against both the mathematical signed-byte mapping and the original signed cast. The independent INT64 quantization/dot oracle and unchanged production FP64 comparator still pass all 17 fixtures.

It does not produce SDOT or signed widening instructions. The scored baseline's unsigned widening/mask/negation/OR sequence becomes unsigned widening/shifted widening/mask/subtract. Both emit INT32 `mla.4s` reductions. This experimentally rules out the simple claim that replacing only the sign expression is sufficient to unlock packed byte dots; it does not identify a single definitive compiler cause.

A first spelling through a small SV function caused Verilator to unroll the dot loop completely. The final variant uses a direct SV macro expression so the loop remains comparable to the baseline. Generated host arithmetic is never edited.

Reproduce from this branch with the original probe commands. The preserved baseline is commit `c6bd0a4`; build it in a separate worktree to compare. Inputs, quantization/scales, and deployed-worker comparator are unchanged. No model defaults or full-model precision change is proposed.

Three alternating-order pairs of 2000 warm invocations gave baseline/variant end-to-end time ratios of 1.111/0.844/0.973 for Q B4 and 0.979/0.999/0.994 for down B4. There is no consistent improvement; the candidate is rejected. Weight preparation remains separately reported; activation quantization and rescaling remain included. These are the same tiny resident 16-row tiles, not full-model timing or quality evidence.

The rejected variant is archived as `tools/experiments/w8a8-sign-form.patch`, based on `c6bd0a4`. Apply it in an isolated worktree before using the original probe commands. The checked-in standalone probe remains the measured baseline. Candidate binary SHA256 and correctness evidence are in `reports/w8a8-sign-codegen.json` and `reports/w8a8-sign-correctness.json`; raw paired samples are in `reports/w8a8-sign-paired.json`.
