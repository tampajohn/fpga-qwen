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
