# Query-row pairing: completed relative acceptance

The isolated query-row candidate has completed the following gates. The [binding archive](../reports/native-cpu-prefill-query-row-pair-relative-acceptance.json) pins the candidate binary and the complete report hashes. These results establish sampled relative behavior, not an absolute numerical qualification or release decision. Build19 remains live; the later completed timing results are recorded separately below.

| Gate | Completed evidence |
|---|---|
| B2 API | [Full36 matrix](../reports/native-cpu-prefill-query-row-pair-verify-api.json), including commit/rollback and invalid-operation checks |
| Prefix reuse | [28 cases](../reports/native-cpu-prefill-query-row-pair-prefix-check.json) |
| Suffix adapter | [35 trajectories plus seven cache/lifecycle cases](../reports/native-cpu-prefill-query-row-pair-suffix-trajectories.json), cancellation and restart |
| Suffix CLI | [Five cases](../reports/native-cpu-prefill-query-row-pair-suffix-cli.json) |
| Original quality suite | [Exact trajectories and unchanged4/9 score](../reports/native-cpu-prefill-query-row-pair-quality.json) against tuned llama |
| Extended conversations | [Exact trajectories and unchanged10/18 score](../reports/native-cpu-prefill-query-row-pair-extended-quality.json) against tuned llama |

The quality scores are **4/9 and 10/18**, not 17/18. The frozen strict rubrics retain all baseline failures. Some answers fail formatting requirements despite containing the requested value; others give a wrong answer, fail task switching, or hit the output cap. Exact baseline agreement is not semantic correctness. No rubric was relaxed and no quality gain is claimed.

These lifecycle and quality results accompany the separate 56+12 full-logit relative checks. They do not resolve released build19's new63-token independent NumPy failure or establish universal exactness. Additional actual-boundary and seeded-stream controls remain separate evidence. The gate snapshot preceded whole-request timing. Its later completed results appear below; no general speed claim or promotion is implied by these gates.

## First completed whole-request comparisons against build19

Both suites completed ten alternating pairs with one warmup per profile/visit, decode12/prefill6 and identical complete output IDs. Standard contexts use 32 output tokens; extended contexts use 128. These results compare the immutable query-row candidate directly with build19, not llama. The [standard full report](../reports/native-cpu-prefill-query-row-pair-standard-confirmation.json), [standard analysis](../reports/native-cpu-prefill-query-row-pair-standard-analysis.json), [extended full report](../reports/native-cpu-prefill-query-row-pair-extended-confirmation.json), and [extended analysis](../reports/native-cpu-prefill-query-row-pair-extended-analysis.json) retain all samples.

Each cell is the median paired speedup and 95% paired bootstrap interval. Higher than 1 favors the candidate; decode is candidate/reference throughput, while latency ratios are reference/candidate. Ratios of medians are not substituted.

| Prompt tokens | Decode | TTFT | Complete request |
|---:|---:|---:|---:|
| 85 | 0.8988 [0.7006, 1.0856] | 0.9821 [0.9321, 1.0149] | 0.9254 [0.7674, 1.0617] |
| 452 | 0.9884 [0.9139, 1.4168] | 1.0286 [0.9889, 1.1107] | 1.0144 [0.9628, 1.2118] |
| 1024 | 1.0904 [0.8980, 1.1852] | 1.0678 [1.0017, 1.1145] | 1.0807 [0.9398, 1.1392] |
| 2048 | 0.9613 [0.7679, 1.0886] | 1.0680 [1.0335, 1.0965] | 1.0402 [0.9027, 1.0831] |
| 3968 | 0.9533 [0.7831, 1.0632] | 1.1119 [1.0902, 1.1535] | 1.1058 [1.0328, 1.1246] |

TTFT intervals favor the candidate at 1024, 2048 and 3968 tokens. At 3968, complete-request latency also favors it in this run; the 1024/2048 request intervals include 1. All decode intervals include 1, and the ordinary decode path is unchanged: no overall decode gain is established. The 85-token request median is 0.9254 with interval [0.7674, 1.0617]; this neither establishes a slowdown nor proves noninferiority.

Documentation/git/artifact work occurred on the shared host during this first comparison and may have contributed noise. The measurements do not identify its causal effect. Every pair and outlier remains included; no selective exclusions or pooling with a future run are allowed. A final quiet-window comparison is planned as distinct evidence, not a replacement for these reports.

This is diagnostic performance evidence while baseline19 absolute 63/64 numerical failures remain unresolved. The supplementary actual-boundary and seeded-stream checks are pending at this snapshot. No release or general speed advantage is claimed.
