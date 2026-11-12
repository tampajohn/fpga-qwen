This sweep changes only the requested worker count on the frozen combined-attention executable `f17ae654f0c64ac47b63225a3406edc8bcd114cc86e67d56610090bf5d7c075d`. It does not change arithmetic flags, source, precision profile, defaults, or native-library settings. The host reports 18 physical cores: six named Super and twelve named Performance. The experiment neither pins threads nor establishes which class runs each worker. The coordinator is additional to the requested worker count.

Worker count also changes kernel eligibility: four and eight use paired-head QK/V operation6 for scalar decode; six and twelve use scalar-head operation1. Batched prefill attention remains operation5. These are complete worker-setting comparisons, not a pure scheduling or core-class experiment. Exact tested model outputs do not imply identical FP64 intermediate rounding between attention routes.

Four models remain resident, one for each of4/6/8/12 workers. Their KV payload alone totals7GiB; this is not measured RSS. Each profile is warmed once per model. Every measured request resets logical KV, prefills the same prompt in B8 chunks with B4 tails, and generates32 greedy tokens. Startup is separate. The85- and452-token profiles each use four orders:4/6/12/8,6/8/4/12,8/12/6/4,12/4/8/6. Every count occupies each order position once. Ratios compare each setting with the eight-worker result from the same round.

Every warmup and measured request must match the eight-worker reference's generated IDs and all151936 final logit bits, consume exactly prompt length+31 positions, and avoid EOS. Direct acknowledged dumps preserve warm pools. Source and binary hashes are recorded; both runtime capabilities and combined-attention flags are checked. The raw report retains all orders, individual latencies, throughput, warmups, per-round ratios and startup durations.

Reproduce with the project Python environment: `python tools/sweep_combined_workers.py --binary <frozen-f17> --pool-policy resident-per-count --output <report>`. Reserve an exclusive scoring window; the script does not isolate the operating system or evict caches. It also offers an explicitly labeled single-model/reconfigure/rewarm mode, which was not used for this report. No native CPU comparison is performed here; earlier matched native runs used eight threads, independently of the browser's six-thread native default.

All 32 scored requests and eight warmups passed full-logit-bit, output-ID and consumed-position checks. No defaults were changed.

| Context | Workers | Median prefill (s) | Median decode (tokens/s) | Paired prefill speedup vs8 | Paired decode speedup vs8 | Paired total speedup vs8 |
|---|---:|---:|---:|---:|---:|---:|
| 85 | 4 | 0.955 | 55.23 | 0.754 | 0.919 | 0.811 |
| 85 | 6 | 0.823 | 55.45 | 0.872 | 0.920 | 0.887 |
| 85 | 8 | 0.714 | 60.78 | 1.000 | 1.000 | 1.000 |
| 85 | 12 | 0.636 | 57.78 | 1.128 | 0.962 | 1.054 |
| 452 | 4 | 6.171 | 42.54 | 0.685 | 0.857 | 0.699 |
| 452 | 6 | 4.979 | 43.51 | 0.852 | 0.841 | 0.859 |
| 452 | 8 | 4.250 | 49.64 | 1.000 | 1.000 | 1.000 |
| 452 | 12 | 3.863 | 45.12 | 1.100 | 0.946 | 1.077 |

Ratios above are medians of within-round speedups, not ratios of separate medians. Twelve workers improved prefill in every round:1.115–1.140× short and1.093–1.107× long. Total request speed also improved every round:1.026–1.062× short and1.041–1.088× long. However, twelve-worker decode was slower in every round, with speedup ranges0.924–0.984× and0.777–0.988×. Eight workers remain the faster tested decode setting; twelve trades some decode throughput for faster prefill on these32-output requests. Four and six workers were slower than eight in both phases across this sweep.

These results do not support a claim that six Super cores alone are the optimal worker target, nor do they identify thread placement. Separate tuning of prefill and decode worker counts might be useful, but reconfiguration cost and changed kernel routing would require their own measured experiment. The full raw evidence is `reports/combined-worker-sweep.json`.
