# Query-row time128 experiment: rejected

The candidate changed only the shared-prefix PV time block from 32 to 128 in the existing query-row kernel. Channel tiling, per-output ascending FMA order, B1/B2 paths, and model/storage formats were unchanged. [Source metadata](../reports/native-cpu-prefill-query-row-pair-time128-experiment.json), [independent review](../reports/native-cpu-prefill-query-row-pair-time128-source-review.json), [build](../reports/native-cpu-prefill-query-row-pair-time128-build.json), [assembly review](../reports/native-cpu-prefill-query-row-pair-time128-assembly-review.json), and [exact patch](../tools/experiments/native-prefill-query-row-pair-time128.patch) are preserved.

All 92 full-logit array checks passed: the existing 56+12 against immutable build19 captures, eight on actual new 63/64 fixtures, and 16 on fresh 127/128/129/130 fixtures against tested query-row time32. Each trajectory has two repeats; explicit 130 reaches common causal length 129 across the 128 boundary. These are relative checks, not a repair of the inherited absolute 63/64 oracle failures. The [decision inventory](../reports/native-cpu-prefill-query-row-pair-time128-decision.json) links every gate.

The [complete shape report](../reports/native-cpu-prefill-query-row-pair-time128-vs-time32-original-shapes.json) retains all 80 pairs, warmups and exact preflights. Reference is tested query-row **time32**, not build19. Two persistent processes run serial commands, with two warmups and ten alternating pairs per shape. Retained-prefix priming is outside the timer; append timing includes the final head and sampling. This is neither fresh browser TTFT nor whole-request throughput.

| Retained prefix | Tail | Median paired wall speedup [95% interval] |
|---:|---:|---:|
| 0 | 2 | 1.0069 [1.0027, 1.0146] |
| 0 | 16 | 0.9977 [0.9842, 1.0068] |
| 0 | 85 | 0.9462 [0.9390, 0.9740] |
| 0 | 452 | 1.0104 [0.9914, 1.0387] |
| 3584 | 2 | 0.9120 [0.8850, 0.9343] |
| 3584 | 16 | 1.0123 [0.9844, 1.0344] |
| 3584 | 64 | 0.9339 [0.9066, 0.9729] |
| 3584 | 384 | 1.0015 [0.9618, 1.0367] |

The 85-token fresh shape and retained-prefix tails 2 and 64 regress in this screen. The largest retained tail 384 has no established benefit. Time128 is **rejected**, with no full-request experiment or promotion. Small-shape favorable results are retained alongside regressions; no selective exclusions or universal faster/slower claim.

Query-row time32 remains the kernel candidate. The packed-head candidate was separately rejected; the tentative composition remains unbuilt and untimed. Reproduction scripts are byte-preserved under `tools/experiments/*.txt`; restore their original `build/native-cpu/` paths before use, with fresh output destinations and root-serialized guards.
