# Eight versus twelve decode workers: retain twelve

The completed same-binary experiment compares decode 8 with decode 12 on the tested plain query-row time32 kernel; prefill 6 is fixed. Both suites use ten alternating pairs and one warmup: standard 32 outputs at 85/452 input tokens, extended 128 at 1024/2048/3968. The full-vocabulary preflight checked every output position in both configurations: 448 paired vectors (896 dumped arrays), all bit-identical. Warm/scored output IDs and natural-EOS contracts also matched.

The [archive inventory](../reports/native-cpu-worker-count-v2-d8-archive-v1.json) links both full reports and analyses, preserving every pair, outlier and full-logit hash. This is a worker comparison on one kernel, not a direct build19 or llama comparison.

| Prompt tokens | Decode speedup [95% CI] | TTFT speedup [95% CI] | Request speedup [95% CI] |
|---:|---:|---:|---:|
| 85 | 0.9368 [0.8726, 1.1814] | 0.9993 [0.9753, 1.0284] | 0.9548 [0.9052, 1.1422] |
| 452 | 0.9093 [0.8309, 1.0071] | 0.9813 [0.9405, 1.0305] | 0.9611 [0.9086, 0.9963] |
| 1024 | 0.9204 [0.8277, 1.2552] | 1.0136 [0.9733, 1.0559] | 0.9537 [0.9131, 1.1512] |
| 2048 | 0.9795 [0.8476, 1.1106] | 0.9818 [0.9484, 1.0417] | 0.9871 [0.9244, 1.0567] |
| 3968 | 1.1395 [1.0368, 1.2556] | 0.9991 [0.9586, 1.0506] | 1.0339 [0.9833, 1.0673] |

Values above 1 favor eight workers. The 3968-token decode interval favors eight, but the 452-token complete-request interval shows a regression. Every other interval includes 1. The decision is **retain 12 decode / 6 prefill workers**; no context-dependent threshold was fitted to these results. Workers 6/18 were deferred before worker measurements, not discarded after unfavorable results.

No optional serving-factory patch or selected-worker lifecycle gates were selected. The conditional source preparation remains unapplied. Kernel selection and fresh matched 12/6 comparisons remain separate. Known absolute 63/64 oracle failures are preserved; exact worker equivalence does not clear them.

Byte-preserved worker harness and archive-controller snapshots are under `tools/experiments/`; restore their original local paths for reproduction. No CPU affinity guarantee or universal worker-count advantage is inferred.
