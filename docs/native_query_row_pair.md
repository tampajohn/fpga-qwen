# Query-row pairing: completed relative acceptance

The isolated query-row candidate has completed the following gates. The [binding archive](../reports/native-cpu-prefill-query-row-pair-relative-acceptance.json) pins the candidate binary and the complete report hashes. These results establish sampled relative behavior, not an absolute numerical qualification or release decision. Build19 remains live; performance results are not included here.

| Gate | Completed evidence |
|---|---|
| B2 API | [Full36 matrix](../reports/native-cpu-prefill-query-row-pair-verify-api.json), including commit/rollback and invalid-operation checks |
| Prefix reuse | [28 cases](../reports/native-cpu-prefill-query-row-pair-prefix-check.json) |
| Suffix adapter | [35 trajectories plus seven cache/lifecycle cases](../reports/native-cpu-prefill-query-row-pair-suffix-trajectories.json), cancellation and restart |
| Suffix CLI | [Five cases](../reports/native-cpu-prefill-query-row-pair-suffix-cli.json) |
| Original quality suite | [Exact trajectories and unchanged4/9 score](../reports/native-cpu-prefill-query-row-pair-quality.json) against tuned llama |
| Extended conversations | [Exact trajectories and unchanged10/18 score](../reports/native-cpu-prefill-query-row-pair-extended-quality.json) against tuned llama |

The quality scores are **4/9 and10/18**, not17/18. The frozen strict rubrics retain all baseline failures. Some answers fail formatting requirements despite containing the requested value; others give a wrong answer, fail task switching, or hit the output cap. Exact baseline agreement is not semantic correctness. No rubric was relaxed and no quality gain is claimed.

These lifecycle and quality results accompany the separate56+12 full-logit relative checks. They do not resolve released build19's new63-token independent NumPy failure or establish universal exactness. Additional actual-boundary and seeded-stream controls remain separate evidence. Whole-request timing must complete before publication; no partial timing, general speed claim, or promotion is implied by this archive.
