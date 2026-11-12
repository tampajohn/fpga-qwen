The final fused-prefill, phase-pool, canonical-capable artifact shows a modest fresh-request gain on the longer prompt versus the live phase profile. Short-request totals are mixed, so this does not establish a universal latency improvement or justify changing the default by itself.

Candidate `6c9ab44bf49b9f35ebf1a86897c284640041fb2a1a811f38184bff306f9e1b89` was compared directly with live phase `162bed6577d91b19a57ea0f83d244d3fbb2666e329e90733f56f347267e8580d`, then isolated FMA-only `013a56b2271120f6ca39cf38f74398b2d918e7d8c5e7dfe0b37679bd8317ec32`. Driver62676 exited0. All12 scored pairs passed fixed32-output IDs/positions and each profile reproduced its own full-vocabulary logit bits. The FMA-only comparison also required exact cross-model bits; the live FP64-prefill comparison reports rounding differences descriptively. Runtime policy and phase counters were checked outside timed intervals.

| Reference | Prompt tokens | Prefill ratio | TTFT ratio | Decode ratio | Total ratio |
|---|---:|---:|---:|---:|---:|
| Live phase FP64 prefill | 85 | 1.103 | 1.103 | 0.963 | 1.036 |
| Live phase FP64 prefill | 452 | 1.118 | 1.117 | 1.012 | 1.100 |
| FMA-only eight-worker pool | 85 | 1.086 | 1.090 | 0.888 | 0.981 |
| FMA-only eight-worker pool | 452 | 1.076 | 1.075 | 1.036 | 1.071 |

Ratios are medians of three alternating per-pair reference/candidate timings; values above one favor the candidate. Live-phase short totals were1.036/0.898/1.058, while long totals were1.100/1.089/1.123. FMA-only short totals were0.981/0.924/1.080 and long totals1.071/0.994/1.076. No historical gains are multiplied together.

Against the live phase profile, absolute TTFT medians were0.617→0.559 seconds for85tokens and3.837→3.399 seconds for452tokens. Total medians were1.089→1.067 and4.398→4.020 seconds. Medians of independently aggregated times do not equal the median paired ratios above. Decode here measures the31 tokens after the first output, unlike some chat metrics that also include final-prompt processing. Scheduling variation remains visible, particularly in short decode.

Both models were resident and warm, with logical KV reset before every request, B8 prefill/B4 tails, separate-rounded FP32 final-prompt/decode/vocabulary and FP64 attention/KV. The candidate and live phase reference each have20 resident worker threads across12-prefill/8-decode pools; isolated FMA has eight. Only one synchronous pool is active per model. Startup, telemetry and closed-file dumps were outside request timing. No OS cache eviction, CPU placement guarantee or hard host isolation was attempted.

This is strictly a fresh-request comparison: canonical capability is validated but no prefix reuse or suffix drafting occurs. Cached behavior and cache/suffix correctness have separate evidence. Live localhost status was sampled every0.5seconds; no busy request was observed, though sampling cannot rule out subinterval activity. The monitored live model remained resident and idle. Reports preserve both sessions separately, plus monitoring evidence. The helper's source hash and frozen binary hashes are included in each report; default/backend settings were not changed.
