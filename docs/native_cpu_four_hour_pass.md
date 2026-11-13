# Four-hour CPU optimization pass — November 13

User-authorized timebox: **17:47:31–21:47:31 UTC, November 13, 2026**
(12:47–16:47 America/New_York). Baseline is live build19 at commit
`f1b30beec15ff484c81d524d20728b7532a5f701`.

## Deliverables

1. Evaluate finer canonical prefix reuse (64/128 versus 512 tokens) on real
   multi-turn conversations, including cached/fresh correctness and full-request
   latency. Smaller prefill batches must not inherit numerical qualification
   from build19 without evidence.
2. Investigate the remaining long-context prefill/first-token latency deficit
   against tuned llama with an independently developed CPU implementation.
3. Deliver any validated improvements through the localhost browser, preserving
   rollback, latest/default selection, benchmark locking, and quality. Publish
   complete evidence and negative results to GitHub.

Parallel agents prepare independent source candidates, evaluation tools, and
reviews. Root exclusively runs builds, model execution, and benchmarks, serialized
under the browser-visible benchmark guard. Shared runtime sources stay frozen
while measurements that bind them are running.

## Time allocation and stop rule

- First 30 minutes: inspect and prepare isolated candidates and targeted gates.
- Until 20:32 UTC: iterate only where evidence supports the next experiment.
- After 20:32 UTC: start no new optimization designs; select or reject existing
  candidates and finish matched measurements.
- After 21:17 UTC: reserve the remaining time for browser validation, deployment,
  documentation, publication, and clearing owned benchmark locks.
- By 21:47:31 UTC: stop this pass and report what was delivered and what remains
  unresolved. Never promote an incompletely validated candidate to meet the clock.

CPU-only Qwen3-0.6B remains the target. No GPU/PyTorch inference or copied llama
kernels. Preserve the fixed independent NumPy limits (max absolute error 0.01,
RMS 0.002), finite full logits and greedy/quality gates. Timing claims require
matched complete work, ten alternating pairs, uncertainty intervals, and all
samples retained. Do not infer complete-request or llama gains from microbenchmarks
or by multiplying separately measured ratios. Build19 remains the fallback.

## Initial work allocation

- Prefix source design: isolated 64/128 candidates and arithmetic implications.
- Evaluation: isolated adapter support and real-history multi-turn fixtures.
- Attention source review: one evidence-driven candidate for long prefill.
- Root: numerical gates, serialized measurements, selection, merge and release.

Status: active; no post-build19 candidate has been accepted or deployed.


## First numerical results: new boundary failure also affects baseline19

Isolated canonical 64/128 candidates change only the shared prefix/prefill chunk
constant; they retain full causal history and FP16 KV. Source preparation flags
in the metadata are historical, while the build manifests record executed builds:

| Candidate | Source and provenance | Completed near-4K numerical evidence |
|---|---|---|
| 64 | [Patch](../tools/experiments/native-prefix64-v1.patch), [metadata](../reports/native-cpu-prefix64-v1-experiment.json), [build](../reports/native-cpu-prefix64-v1-manifest.json) | [3968/3969 gate](../reports/native-cpu-prefix64-v1-context3968-oracle.json): pass |
| 128 | [Patch](../tools/experiments/native-prefix128-v1.patch), [metadata](../reports/native-cpu-prefix128-v1-experiment.json), [build](../reports/native-cpu-prefix128-v1-manifest.json) | [3968/3969 gate](../reports/native-cpu-prefix128-v1-context3968-oracle.json): pass |

Both near-4K reports have maximum absolute errors 0.007285118 / 0.007415295 and
RMS errors 0.001512432 / 0.001223014 at the two positions, with greedy identity.
Those passes do not qualify other prompts or establish general numerical safety.

The new [63-token boundary fixture](../reports/native-cpu-prefix-granularity-boundary63-oracle-v1.json)
**failed** at position 63 for both candidates under unchanged max-absolute 0.01 /
RMS 0.002 limits. The [independent baseline19 control](../reports/native-cpu-build19-newboundary63-control-oracle.json)
produced the same errors on the same prompt and continuation:

| Position | Maximum absolute error (both candidates and baseline19) | RMS error | Fixed gate |
|---|---:|---:|---|
| 63 | 0.012255669 | 0.002722482 | Fail: both limits |
| 64 | 0.009222031 | 0.001818819 | Pass |

All greedy predictions matched. Baseline agreement does not override the
failed full-logit contract, and the later passing position does not erase the
failure. The control establishes that this fixture also exposes a baseline19
gap; it does not establish the cause or make either candidate acceptable.

Performance claims and new optimization designs are held while numerical
investigation proceeds. No tolerance change, candidate promotion or broad gate
pass is claimed. Remaining boundary-suite outputs are not aggregated or published
as complete. The prior 28-position baseline evidence remains valid within its
own fixtures; this newly tested prompt extends coverage and reveals its limit.
