# Vocabulary routing: correct component prototype, negative speed result

The first standalone routing prototype passed its exact-output checks but was
slower than the dense FP16 head in all six measured comparisons. Keep the dense
head in the engine. This prototype has not been integrated into inference or
browser chat.

The [compact evidence](../reports/native-head-probe-short-v1.json) preserves all
paired timings, correctness outcomes, input/build hashes and limitations. The
complete local report is `build/native-cpu/head-probe-short-v1/report.json`.

## What was tested

The probe uses an original persistent 12-worker pool and the repository's
canonical FP16-weight/FP32-accumulation dot topology. An auxiliary whole-row
int8 encoding supplies approximate logits and conservative upper bounds. Hidden
states are quantized only for this auxiliary pass; exact rescoring uses the
saved FP32 hidden state and original FP16 weights.

The runtime exactly rescores its approximate top-1 or top-20 seed, uses that
seed's exact kth score as a threshold, and exactly rescores every other row whose
upper score can reach the threshold. Equality remains eligible. Saved dense
logits are used by the correctness harness, never to choose the runtime
threshold. Top-20 applies the actual consumed-token presence mask and existing
FP32 penalty/temperature operations. The probe does not execute top-p sampling
or RNG.

The auxiliary encoding adds **157,405,696 bytes** for int8 weights and three
FP32 metadata fields per vocabulary row. The canonical **311,164,928-byte** FP16
head remains necessary. These are encoding payloads, not measured process RSS;
scratch allocations are additional. Cache construction and process startup are
excluded from the paired timings, and both configurations retain the same
allocations.

All eight saved states from one short-prompt trajectory, contexts 85–92, passed
before any state was timed:

- All 151,936 dense logits matched the saved canonical arrays bit for bit.
- Routed greedy and history-adjusted top-20 IDs, exact logits and scores matched
  the dense head.
- Every dense logit was within its computed upper bound.
- All 12 worker FPCR values were zero, SDOT was available, and no dense fallback
  or unsafe-bound row occurred.

The routing path requires the entire FPCR to be zero; unsupported control states
or unavailable SDOT select dense fallback without changing FPCR. Source review
and these observations supplement the stated arithmetic assumptions; they do
not constitute formal verification or a full-model quality evaluation.

## Measured component latency

Steps 0, 3 and 7 were selected before results were observed. Each comparison used
two warmups and ten alternating pairs on the Apple M5 Max, with CPU-only
execution. Latency includes projection/routing and exact greedy/top-20 selection.

| Context | Selection | Dense median | Routed median | Median paired dense/routed ratio, 95% interval |
| --- | --- | ---: | ---: | ---: |
| 85 | Greedy | 1.339 ms | 1.498 ms | 0.892 (0.868–0.919) |
| 85 | Top-20 | 1.459 ms | 1.552 ms | 0.952 (0.901–0.974) |
| 88 | Greedy | 1.334 ms | 1.501 ms | 0.883 (0.872–0.901) |
| 88 | Top-20 | 1.465 ms | 1.656 ms | 0.893 (0.871–0.932) |
| 92 | Greedy | 1.376 ms | 1.500 ms | 0.934 (0.859–0.949) |
| 92 | Top-20 | 1.478 ms | 1.559 ms | 0.953 (0.915–0.992) |

Ratios below one favor the dense head. These are medians of paired ratios,
not quotients of independent latency medians. All six unadjusted paired
bootstrap intervals are below one in this run. The fixed saved vectors,
uncontrolled OS activity and multiple comparisons limit generalization. No
full-model token rate or browser latency improvement follows from this test.

## What the result identifies

The timed greedy cases rescore 3, 1 and 1 rows respectively. Thus the last two
already eliminate all exact rescoring beyond the single seed, yet routing is
slower. Tighter bounds cannot reduce that work further. Across all eight
validation states, greedy rescoring ranges from 1–10 rows; top-20 ranges from
101–11,351 rows. Candidate counts alone did not establish a speed gain.

No stage profile was recorded, so the result does not attribute a percentage of
time to any implementation detail. Source inspection identifies several
testable costs in [the probe](../tools/native_head_probe.cpp):

- Every row uses one dependent SDOT accumulator chain, followed by scalar FP64
  bound construction.
- Approximate top-k selection and the subsequent vocabulary-wide threshold scan
  run on the coordinator.
- Routing adds a second pool dispatch, final subset selection, and a diagnostic
  scan counting unsafe bounds.

One bounded followup is justified before abandoning the approach: separately
profile activation preparation, the parallel approximate/bound pass, seed
selection, threshold filtering, exact rescoring and final selection. Measure
instrumentation overhead with profiling disabled/enabled before interpreting
small stage costs.

If the approximate dot pass is material, test four independent integer
accumulators or row interleaving as one isolated change. Their reduction can
preserve the exact integer dot because the absolute full-row sum is bounded by
`1024 * 127² = 16,516,096`, safely within int32. If coordinator selection is
material, test worker-local top-k and parallel threshold filtering instead.
Both are hypotheses, not measured gains. Further tightening the candidate bound
alone is a lower priority given the one-row greedy results.

Any followup must retain the current negative evidence, pass all saved-state
checks, and repeat the paired component comparison before considering engine
integration. The current implementation is rejected for integration.

## Reproduction

The saved coupled capture must already exist. Under the main benchmark guard:

```sh
.venv/bin/python tools/benchmark_guard.py --name 'Standalone head routing comparison' -- \
  .venv/bin/python tools/benchmark_native_head.py \
  --build-directory build/native-cpu/head-probe-new \
  --capture-manifest build/native-cpu/build-13-head-short/report.json \
  --directory build/native-cpu/head-probe-new-results
```

Use new artifact directories. The driver validates every capture step before
timing its preselected 0/3/7 subset, freezes inputs and source hashes, and records
partial failures. It does not alter the engine, weights, or live browser backend.
