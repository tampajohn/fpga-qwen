This isolated opt-in experiment replaces only projection accumulation with `$verilator_fma32(activation, weight, partial)`. It intentionally rounds the product-plus-partial once instead of separately rounding multiply and add. Column reduction lanes, token layout, BF16 weight conversion, FP32 activation conversion and final separately rounded lane sums remain unchanged. The default SV macro uses the original arithmetic. No full-model runtime or UI profile is introduced.

Separate and fused B1/B4/B8 artifacts use the same experimental compiler SHA256 `3a123151340a420d36edd1766585218ad882c343bf8b03c7db8ffdfb89de8c87`, strict `-fno-fast-math -ffp-contract=off`, and ThinLTO. Both B8 artifacts retain `--unroll-count 1`. No global vector-width override or implicit contraction change is involved.

All36 correctness cases passed at workers1 and8. Eighteen tiny fused outputs matched the exact-rational oracle bit for bit. The separate outputs of the new compiler matched the previous strict artifacts across all full Q/gate/down matrices. Fused full-matrix differences are descriptive, not an exactness gate: maximum absolute difference was1.430511474609375e−6. This does not establish full-model quality or justify a tolerance for future model gates. See `reports/fma-projection-correctness.json`.

Independent final-linked assembly review confirmed B1's inner loop at0x100008eb4–0x100008ee0 contains two `fmla.4s` operations across its eight column lanes. B4's loop at0x100008ed4–0x100008ff0 and B8's at0x100009004–0x1000090c4 each contain sixteen `fmla.2s` operations across independent token pairs. There are no inner-loop calls or stack spills; BF16 weight expansion stays inline. Final sums retain serial lane order, with no horizontal reduction. B8 still does not use four-token SIMD. Its loop has49 static instructions versus65 in the separate artifact; instruction count is not a performance measurement.

The optional builder argument is `--candidate-explicit-fma`; build-only use avoids accidentally applying the old separate-rounding oracle to fused results. `tools/check_fma_projection.py` takes the new build directory and three explicit legacy artifact paths. `tools/compare_fma_projection.py` requires the completed correctness report and uses1000 repeated projections and three alternating pairs by default. It checks stable outputs separately within each rounding mode and reports cross-rounding differences descriptively.

The probe times SV conversion and repeated eval work after initial input setup. A worker signals completion before output copy/teardown, so an earlier worker's copy may overlap a still-timed straggler. Output traffic is not fully isolated. There is one launch/completion barrier around repetitions, not one per model-layer dispatch. Any timing result is therefore a warmed synthetic-activation projection microbenchmark, not model prefill latency or inference throughput. Full-model quality, cache/suffix semantics, dispatch overhead and representative vocabulary work require later independent gates.

The authorized exclusive projection timing window completed with1000 repeated evals, eight workers, and three alternating pairs for each batch/matrix. All output checks within each rounding mode remained exact.

| Batch | Matrix | Median fused speedup | Range |
|---:|---|---:|---:|
| 1 | q | 0.855× | 0.851–0.867× |
| 1 | gate | 0.876× | 0.833–0.892× |
| 1 | down | 0.793× | 0.785–0.811× |
| 4 | q | 1.204× | 1.195–1.220× |
| 4 | gate | 1.212× | 1.189–1.215× |
| 4 | down | 1.194× | 1.187–1.206× |
| 8 | q | 1.421× | 1.384–1.424× |
| 8 | gate | 1.397× | 1.377–1.472× |
| 8 | down | 1.388× | 1.388–1.395× |

Every B4 and B8 pair favored fused arithmetic; every B1 pair favored separate arithmetic. The bounded result supports exploring fused prefill without changing scalar decode. B4 verification cannot simply switch to FMA while B1 stays separate: speculative verification needs target-equivalent arithmetic, so preserve existing B4 verification until explicitly proved otherwise. A prefill-only integration must separate prefill tails from verification routing and establish its own numerical/cache policy.

The raw paired report is `reports/fma-projection-paired.json`. No full-model speed, quality improvement, or general FMA advantage is claimed. Existing live defaults are unchanged.
