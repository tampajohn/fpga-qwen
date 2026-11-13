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

## Isolated QK key-pair candidate: exactness only, performance held

The [original source patch](../tools/experiments/native-prefill-qk-keypair.patch)
processes two causal key rows together for the existing pair of query heads in
ordinary batched prefill. Each of the four dot products keeps its own two FP32
vector accumulators, ascending channel FMA order and final reduction. Query
loads are shared; odd causal tails retain the original helper. B1 decode, B2
verification, softmax, PV reduction and normalization are unchanged. See
[source metadata](../reports/native-cpu-prefill-qk-keypair-experiment.json) and
[executed build manifest](../reports/native-cpu-prefill-qk-keypair-manifest.json).
The metadata's unexecuted flags describe source preparation, not the subsequent
completed build and checks.

The [56-array repeated trajectory gate](../reports/native-cpu-prefill-qk-keypair-exact-gate.json)
and [12-array causal-boundary gate](../reports/native-cpu-prefill-qk-keypair-timeboundary-exact-gate.json)
passed with full-logit byte identity against the prior combined candidate whose
executable is byte-identical to released build19. These cover the existing tiny,
85/452, 511/512/513, 1024/2048/3968 trajectories and causal lengths
31/32/33/63/64/65. This is fixture-scoped equivalence, **not an independent NumPy
pass**: the newly discovered baseline19 63-token numerical failure above remains
unresolved. Identical causal lengths do not imply identical prompt fixtures.

[Static assembly inspection](../reports/native-cpu-prefill-qk-keypair-assembly-review.json)
confirms the intended inner channel loop: four FP32 query-vector loads, two
FP16 key-vector loads, four widening conversions and eight independent vector
FMAs per iteration. That loop has no stores, stack references or calls. The
surrounding function does store pointer/state values on the stack, so this is
not a whole-function spill-free claim. It establishes neither a measured load
traffic reduction against baseline nor a performance gain. Timing and promotion
remain held while the independent numerical gap is investigated; live build19
is unchanged.
