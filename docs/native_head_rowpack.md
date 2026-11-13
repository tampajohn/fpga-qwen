# Experimental FP16 vocabulary-head row packing

This candidate has been prepared, independently source-reviewed, and compiled. The later runtime gates and full-request measurements are now complete, as recorded below. It was rejected for release because benefit was not established; build19 remains live.

The [source metadata](../reports/native-cpu-head-rowpack4-v1-experiment.json), [build manifest](../reports/native-cpu-head-rowpack4-v1-build.json), and [preparation/reproduction inventory](../reports/native-cpu-head-rowpack4-v1-preparation.json) retain the baseline and candidate hashes. The source metadata's `built:false` describes its original preparation snapshot; the separate completed manifest records the subsequent compilation. Neither is a runtime correctness result.

The auxiliary head stores the same FP16 bits in `[row block of 4][column block of 8][row][8 columns]` order. Each four-row dot retains the original per-row FP32 accumulator, increasing-column FMA order, and final reduction. Workers peel to a global four-row boundary, process complete groups within their assigned rows, and finish with original scalar row kernels. The tied embedding lookup and B2 verification path retain the original layout.

This applies to the final vocabulary projection after both ordinary prefill and ordinary decode, because both invoke the single-input head. It does not change other projections or attention. The extra allocation is **311,164,928 bytes (296.75 MiB)**; the original mapped weights, widened prefill matrices, and KV storage remain. Weight traffic is not claimed to decrease. Synchronous packing finishes before worker creation and the ready handshake; allocation or size failures abort initialization. Startup packing time and actual cache bytes are exposed separately.

The [exact patch](../tools/experiments/native-head-rowpack4-v1.patch) and byte-preserved helper copies listed in the inventory reproduce the isolated experiment. Copy the `.txt` helpers back to their recorded `build/native-cpu/` locations before invoking them; their relative paths intentionally retain the tested repository layout. Fresh destinations are mandatory. The generator requires source hashes matching immutable build19; shared source is not changed.

The prepared serialized runner checks 68 existing full-logit arrays, then 40 arrays over decode worker counts 1, 2, 6, 12, and 64 with prefill workers fixed at 6. The existing 12-worker tiny capture is reused. Its isolated comparator permits only the declared decode-worker configuration difference; complete logits, IDs, EOS positions, and all other configuration fields must match. API36, prefix28, suffix35 plus seven lifecycle cases, CLI5, and the existing nine/eighteen-case quality suites precede any standard32/extended128 ten-pair whole-request comparisons against build19. Startup is outside request timing; the memory and packing costs still require disclosure.

The standalone `benchmark_native_head.py` measures a separate SDOT routing prototype and does not exercise this candidate's actual Model head. It is therefore not used as evidence for this packing change. No new diagnostic API or duplicate benchmark kernel is introduced.

Released build19's newly observed 63-token absolute NumPy failure remains unresolved. Even successful relative byte-identity gates would establish only sampled nonregression against that baseline. Any eventual timings are diagnostic evidence, not an absolute numerical qualification or promotion decision.

## Completed gates and rejection for release

The [decision and evidence inventory](../reports/native-cpu-head-rowpack4-v1-decision.json) records completed exact56+12 checks,40 worker-partition arrays (workers1/2/6/12/64), eight arrays on the actual new63/64 fixtures, API36, prefix28, suffix35 plus seven lifecycle cases, and CLI5. The source/assembly review and all relative runtime gates passed. Quality trajectories and grades exactly match tuned llama: **4/9 and 10/18**, with all baseline failures retained. These are sampled relative checks; the independent baseline63/64 absolute failures remain unresolved.

Both whole-request suites completed ten alternating pairs against build19 with one warmup, decode12/prefill6, standard32 and extended128 output caps. Actual candidate startup metadata confirms **311,164,928 additional bytes**. All raw pairs/outliers are archived. The table gives median paired speedups and 95% paired bootstrap intervals; values above 1 favor the candidate.

| Prompt tokens | Decode | TTFT | Complete request |
|---:|---:|---:|---:|
| 85 | 0.9611 [0.7562, 1.1023] | 1.0096 [0.9677, 1.0667] | 0.9749 [0.8099, 1.0875] |
| 452 | 0.9490 [0.7878, 1.1404] | 0.9977 [0.9657, 1.0669] | 0.9752 [0.9108, 1.0776] |
| 1024 | 1.1459 [0.9881, 1.4327] | 1.0156 [0.9956, 1.0519] | 1.0856 [0.9832, 1.2145] |
| 2048 | 1.0111 [0.8194, 1.1758] | 1.0056 [0.9500, 1.0785] | 1.0102 [0.9377, 1.0903] |
| 3968 | 1.0793 [0.9602, 1.2312] | 1.0062 [0.9641, 1.0393] | 1.0077 [0.9597, 1.0602] |

Every interval includes 1. No established benefit justifies retaining the extra 296.75 MiB, so this candidate is **rejected for release**. The results do not prove it universally faster or slower, and no outliers were excluded. Startup packing was outside request timing; its memory/startup cost remains part of the tradeoff.

The tentative query-row plus packed-head composition remains source preparation only: **not built, not timed, not selected**. Its runtime qualification cannot be inherited from the separate components. Live build19 remains unchanged.
