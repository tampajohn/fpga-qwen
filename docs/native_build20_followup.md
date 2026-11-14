# Performance work after build 20

Build 20 remains the serving baseline. This follow-up keeps CPU-only execution,
FP16 weights/KV, unchanged numerical limits, and original kernels. It does not
restart the previous four-hour timebox or imply that inherited numerical failures
have been fixed.

Three investigations run in parallel at the source/review level. Builds, model
runs, numerical calculations and benchmarks run serially under the browser-visible
benchmark guard. Candidates use isolated source/build directories; live sources
and release defaults change only after validation.

1. **Operation-specific worker scheduling.** The earlier same-binary 8-versus-12
   test improved near-4K decode but regressed the 452-token complete request.
   Investigate eight useful workers for ordinary B1 attention while retaining
   twelve for projections and the vocabulary head. This is a new hypothesis,
   not an extrapolated gain or an adaptive context threshold. Inspect publication,
   completion, exception and empty-range handling before compiling. Measure a
   held-out context grid as well as historical workloads; freeze the policy before
   measurements and retain all results.
2. **Numerical diagnosis before finer prefix blocks.** Existing traces locate
   the first batch/sequential divergence at layer-zero Q/K/V projections after
   identical normalized inputs. Better local projection error did not resolve
   every full-network fixture. Analyze FP16 KV rounding and matched-input attention
   perturbations before proposing another arithmetic change. Diagnostic references
   do not replace the independent network oracle or its fixed acceptance limits.
3. **Decode overhead.** Inspect protocol round trips, sampling, and worker
   synchronization. Measure components before changing streaming or cancellation
   semantics. Any amortized protocol must preserve per-token output, EOS, seeded
   sampling, suffix verification and cancellation behavior.

Every candidate needs source/binary-bound correctness evidence before performance
claims. Arithmetic-preserving changes must retain exact baseline trajectories,
including the known failing 63/64 fixtures; arithmetic changes remain subject to
the unchanged absolute numerical gates. Comparative quality checks are not claims
that the small model answers every test correctly. Complete-request latency,
TTFT and decode throughput remain separate metrics, with alternating pairs and
uncertainty reported. A new release requires isolated and live browser checks,
rollback artifacts, current documentation and published evidence.

Status: investigations started; no new candidate qualified or promoted.
