# B1 head pairing: not selected after incremental confirmation

The final head-pair candidate is **not selected**. At 12 decode / 6 prefill workers,
all 15 confidence intervals for full requests, TTFT and decode across five
contexts include 1. The coordinator retains plain prefill query-row pairing.
This does not establish a universal slowdown; it establishes no reliable
incremental benefit in these measurements.

## Complete-request measurements

Ratios above 1 favor head pairing over plain query-row pairing. Standard contexts
use 20 alternating pairs and 32 output tokens; extended contexts use 10 pairs and 128
tokens. Paired percentile bootstrap intervals retain all samples and outliers.
These are incremental comparisons, not build19 or llama comparisons.

| Prompt tokens | Median request speedup | 95% interval |
|---|---:|---|
| 85 | 1.01125 | [0.97485, 1.03798] |
| 452 | 1.00781 | [0.98448, 1.04403] |
| 1024 | 1.02176 | [0.91453, 1.11059] |
| 2048 | 1.03561 | [0.98521, 1.06515] |
| 3968 | 1.03894 | [0.97596, 1.08211] |

The 3968-token decode estimate is 1.17500 with interval [0.99359,1.39982]: suggestive,
but not an established gain. No pooling, cherry-picked metric or product of
separately measured ratios is used to claim a winner. Complete
[standard observations](../reports/native-cpu-queryrow-b1-headpair-v1-vs-queryrow-standard-confirmation.json),
[standard analysis](../reports/native-cpu-queryrow-b1-headpair-v1-vs-queryrow-standard-analysis.json),
[extended observations](../reports/native-cpu-queryrow-b1-headpair-v1-vs-queryrow-extended-confirmation.json)
and [extended analysis](../reports/native-cpu-queryrow-b1-headpair-v1-vs-queryrow-extended-analysis.json)
are preserved.

## Functional nonregression and its limits

The candidate passed 100 exact-array comparisons, the full API matrix,
[prefix lifecycle](../reports/native-cpu-queryrow-b1-headpair-v1-prefix-check.json),
[suffix trajectories](../reports/native-cpu-queryrow-b1-headpair-v1-suffix-trajectories.json),
[CLI suffix checks](../reports/native-cpu-queryrow-b1-headpair-v1-suffix-cli.json), and
[complete seeded streams](../reports/native-cpu-queryrow-b1-headpair-v1-seeded-stream.json).
The [API report](../reports/native-cpu-queryrow-b1-headpair-v1-verify-api.json)
binds the tested wrapper to the candidate sources. These checks establish their
stated finite regression scope, not a speed benefit.

Strict answer accuracy remains **4/9** on the
[frozen suite](../reports/native-cpu-queryrow-b1-headpair-v1-quality.json) and
**10/18** on the [extended suite](../reports/native-cpu-queryrow-b1-headpair-v1-extended-quality.json).
A passing trajectory/nonregression gate is not an all-answers-correct claim.
Wrong answers and the inherited63/ 64 absolute numerical failures remain in the
record; the numerical limits are unchanged.

The previously local [plain-query-row seeded-stream report](../reports/native-cpu-prefill-query-row-pair-seeded-stream.json)
is also archived. The [archive manifest](../reports/native-cpu-headpair-functional-timing-archive-v1.json)
binds originals and sanitized copies. Raw float dumps remain local. Process
identifiers and command names are omitted; per-process CPU/memory/elapsed
observations are retained without identifying processes. Source snapshots and
initial 100-array evidence are linked in the
[four-hour pass record](native_cpu_four_hour_pass.md).
