# Final query-row kernel comparison against build19

This fresh quiet-window comparison uses the selected plain query-row time32 kernel with **12 decode / 6 prefill workers** on both sides. Standard 85/452 contexts use 32 output tokens and 20 alternating pairs; extended 1024/2048/3968 contexts use 128 tokens and 10 pairs. Each visit has one warmup outside scoring. Complete output trajectories match.

The [archive inventory](../reports/native-cpu-prefill-query-row-pair-final-vs19-archive-v1.json) links both full reports and analyses with every sample/outlier preserved. These final results remain separate from the first noisy query-row comparison; none were discarded or pooled. The archive does not infer a quality pass or release.

Each cell is median paired speedup [95% paired bootstrap interval]. Higher than 1 favors query-row; latency ratios are reference/candidate and decode ratios are candidate/reference.

| Prompt tokens | Decode | TTFT | Complete request |
|---:|---:|---:|---:|
| 85 | 0.9705 [0.8902, 1.0437] | 0.9965 [0.9657, 1.0455] | 0.9754 [0.9125, 1.0308] |
| 452 | 0.9709 [0.9046, 1.0749] | 1.0122 [0.9762, 1.0301] | 0.9965 [0.9479, 1.0355] |
| 1024 | 1.1024 [0.8201, 1.4918] | 1.0610 [1.0227, 1.1828] | 1.0837 [0.9231, 1.2703] |
| 2048 | 1.1332 [0.8690, 1.3352] | 1.1028 [1.0533, 1.1608] | 1.1354 [0.9834, 1.2185] |
| 3968 | 0.9895 [0.8083, 1.3241] | 1.0997 [0.9053, 1.1636] | 1.0729 [0.9118, 1.1520] |

The1024- and 2048-token TTFT intervals favor the candidate. The 3968-token TTFT interval includes 1, so a universal near-4K benefit is not established. All decode and complete-request intervals include 1, as do all short-context intervals. This does **not** establish a decode win, noninferiority, or an overall latency win. The limited positive evidence is 1K/2K TTFT.

The selected kernel remains a sampled baseline-nonregression candidate: completed numerical/lifecycle/quality checks preserve the known baseline63/ 64 absolute oracle failures. Matching trajectories do not convert those failures into passes. That snapshot preceded the fresh tuned-llama comparison below. Actual release/browser verification remains separate evidence.

## Fresh matched comparison with tuned llama FP16

Both engines use 12 decode / 6 prefill workers. Standard 85/452 contexts use 32 outputs; extended 1024/2048/3968 use 128. Each suite completed ten alternating pairs and one warmup per profile/visit. The [complete archive](../reports/native-cpu-prefill-query-row-pair-final-vsllama-archive-v1.json) links full reports and analyses, retaining every pair and outlier. This comparison is direct; no ratios are chained from older runs.

Each cell is median paired speedup [95% paired bootstrap interval]; higher than 1 favors the original native engine.

| Prompt tokens | Decode | TTFT | Complete request |
|---:|---:|---:|---:|
| 85 | 0.8561 [0.4897, 1.0001] | 1.6957 [1.4536, 2.1194] | 1.0277 [0.6928, 1.3344] |
| 452 | 0.9493 [0.7861, 1.1613] | 1.1972 [1.0467, 1.4381] | 1.0313 [0.9896, 1.3142] |
| 1024 | 1.0068 [0.7869, 1.2380] | 1.2260 [1.1363, 1.4069] | 1.1901 [0.9761, 1.2700] |
| 2048 | 1.1906 [1.0876, 1.3003] | 1.1411 [1.0969, 1.2506] | 1.1397 [1.0682, 1.2756] |
| 3968 | 1.2426 [1.0852, 1.4002] | 1.0433 [1.0052, 1.1635] | 1.0770 [1.0335, 1.1347] |

Decode favors native at 2048, 3968 prompt tokens; remains uncertain at 85, 452, 1024 prompt tokens.

TTFT favors native at 85, 452, 1024, 2048, 3968 prompt tokens.

Complete request favors native at 2048, 3968 prompt tokens; remains uncertain at 85, 452, 1024 prompt tokens.

These unadjusted paired intervals describe the sampled workloads on this host. They do not establish a universal win, noninferiority where an interval includes 1, or general model quality. The frozen 9/18 suites retain 4/9 and 10/18 rubric scores with matching trajectories; known absolute63/ 64 NumPy failures remain unresolved. Deployment authorization and browser evidence are recorded separately.
