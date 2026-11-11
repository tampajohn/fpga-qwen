# FP32 hybrid projection acceptance experiment

This is an opt-in experiment, not a deployed model or a quality-equivalence claim.
The candidate uses the unchanged FP64 B8 prefill wire command on prompt IDs except
for the last ID, including every tail of one through eight IDs. The last prompt ID
and subsequent decode steps use FP32 B1 projections. Remaining model operations
and KV storage stay in SystemVerilog. The integration artifact must separately
record exactly which projections use FP32 and its compiler/build provenance.

Every chat request resets KV (`reuse_prefix=False`). FP32 decode KV cannot be
reused as though it came from fresh FP64 prefill. Verification capacity is one,
so suffix drafting is rejected before execution. Defaults of existing backends
remain unchanged. Future canonical FP64 prefix checkpoints and FP32 B4 suffix
verification need separate validation; current suffix parity is not evidence
for mixed-precision suffix parity.

## Frozen checks before observing candidate outputs

- The existing held-out 16-case, 292-target corpus retains its authored targets,
  category grouping, hashes and full-vocabulary checks. Run the existing
  `tools/evaluate_projection_quality.py` with `--baseline-block-size 8
  --candidate-block-size 8`. This performs canonical FP64 prefill then token
  decode for each engine. Do not use `--require-exact` for FP32 versus FP64.
  Record per-token and category NLL delta, KL, top-1 agreement and raw logit
  error. These are descriptive measurements, not a blanket logit tolerance.
  Preserve changed predictions for review instead of tuning targets or prompts.
- Run `tools/evaluate_prefill_conversations.py --reference-prefill-size 8
  --candidate-block-size 1 --disable-prefix-reuse` on the same two artifacts.
  All nine existing expectations remain fixed. A baseline-passing case becoming
  a candidate failure is a regression even if another case improves. Reports
  explicitly list these regressions. Different valid responses need not be
  token-identical; all texts and IDs remain available for review.
- `SV_HYBRID_BINARY=... python -m unittest tests.test_suffix_real.RealHybridRouteTests`
  checks all eight tail lengths, same-session reset/replay, process restart,
  sampled token identity and full-logit bit identity within the same hybrid
  route. It also verifies suffix rejection leaves existing state untouched.
  This check does not compare hybrid output to an all-token FP32 route.
- `python -m unittest tests.test_suffix_chat` checks transport pair validation,
  no dropped prefill tokens, explicit cache resets and suffix rejection before
  execution without loading a model. Existing default prefix/suffix tests stay
  enabled. Low-level decode truncate/replay may reuse `tests/test_kv_layout.py`
  with both binary arguments set to the candidate, as a B1-only isolation gate.

Record exact checkpoint, tokenizer/template, corpus, binary and compiler hashes.
Use the unchanged deployed FP64 artifact, with its original optimization flags,
as reference. Do not compile it with the shortreal compiler's strict rounding
flags simply to match the candidate. Both engines must receive identical prompt
and teacher-forced target IDs, same worker count and explicit route settings.

## Decision and limitations

Nonfinite or incomplete logits, dropped tokens, nondeterministic same-route
replay, accidental prefix reuse, or accepted suffix requests fail integration.
Any conversation pass regression blocks a maintained-quality claim. Distribution
metrics require inspection by case/category and of changed predictions; this
small corpus supplies no calibrated universal loss threshold. Even zero observed
regressions only support this bounded evaluation, not general model equivalence.

Measure hybrid TTFT and decode separately after correctness, using exclusive
paired timing windows. Include startup cost and retained FP64 B8 prefill benefits.
All-token B1 performance is an isolation measurement, not the final candidate.
No inference results have been produced by this acceptance-plan change.
