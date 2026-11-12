# Shared FP32 KV scratch for batched prefill (candidate 16)

Candidate 16 widens canonical FP16 keys and values into shared FP32 scratch once per layer and prefill chunk, then reuses those values across causal query rows. This reduces repeated conversions without replacing attention reductions with BLAS. The [immutable build record](../reports/native-cpu-build-16-build.json) identifies the executable, compiler options and source hashes.

The scratch is head-major `[8, context, 128]` for each of K and V. Both buffers together allocate **32 MiB at context 4096**, once per model instance, including instances that only decode. They are temporary working storage reused across layers and chunks; the retained KV remains FP16. Conversion reads the already rounded canonical FP16 cache after current-chunk KV stores, and a worker-pool completion barrier precedes attention reads. Each query retains its causal length, FP32 FMA operands, accumulation order, normalization and reduction topology. Ordinary single-token decoding and the dedicated two-position verifier do not use this scratch.

This is an arithmetic-preserving design on the tested implementation. Source inspection and finite tests are not a formal proof across compilers, architectures or floating-point control modes. No speedup follows automatically from reducing conversion count: the extra dispatch, copying, memory traffic and resident allocation may cost more for small prefill tails.

## Completed correctness gates

- [56 full-vocabulary arrays](../reports/native-cpu-build-16-exact-gate.json) matched frozen build 12 bit-for-bit, including predicted and consumed IDs. The matrix contains two repeats of 28 positions: tiny 1–4, short 85–92, long 452–455, two positions each for initial lengths 511/512/513, and two each at 1024/2048/3968. This checks the ordinary path against the full previously frozen matrix.
- [28 prefix-reuse cases](../reports/native-cpu-build-16-prefix-check.json) passed cached-versus-fresh full-logit/trajectory checks across greedy and seeded chat, edits, shorter inputs and session changes. Cancellation/reaping/restart and CLI rewind checks also passed.
- [35 suffix adapter fixtures](../reports/native-cpu-build-16-suffix-trajectories.json) matched the frozen reference with suffix off/on, preserving visible IDs, EOS, text and stop/cap behavior. These fixtures exercised 63 proposals and 32 acceptances. Seven additional cached-prefix cases covered identical prompts, tail and early edits, and session changes; cancellation and restart checks also passed. These counts establish exercised paths, not general acceptance rates or speed.
- [Five real CLI lifecycle cases](../reports/native-cpu-build-16-suffix-cli.json) passed rejection/acceptance and first/bonus EOS handling, committed logits/hidden-state identity, and malformed/chat-mode reset behavior.
- The [nine-case comparison with tuned llama](../reports/native-cpu-build-16-quality.json) passed exact trajectory and no-new-regression gates. Both engines scored **4/9** under the unchanged rubric. Matching baseline errors is not an answer-quality improvement.

The compact archives retain fixtures, token IDs, array hashes, protocol counters and helper hashes, with raw-report SHA256 provenance. Host identifiers and incidental unscored timing fields are omitted. Correctness gates do not provide benchmark evidence, and the exact matrix is scoped to its recorded snapshots.

## Shape diagnostic outcome: no candidate-16 promotion

The [completed shape diagnostic](../reports/native-cpu-build-16-prefill-shapes.json) compares frozen build 15 against candidate 16 using ten alternating pairs and two symmetric warmups per shape. All eight preflight shapes matched full logits and normalized hidden vectors exactly before any timing, and every measured visit matched its expected token. The archive retains all **80 measured pairs**, not a selected subset.

Each ratio is reference 15 latency divided by candidate 16 latency; values above 1 favor candidate 16. These are tail-prefill command timings including the final vocabulary head and greedy selection. Prefix 3584 was primed outside timing and retained by rewind, so these are not fresh-request TTFT or llama comparisons. Two processes stayed resident, executing serially.

| Retained prefix P | Tail B | Paired CLI speedup (95% interval) | Paired pipe-wall speedup (95% interval) |
|---|---|---|---|
| 0 | 2 | 0.978007 (0.943269–1.006321) | 0.977996 (0.943225–1.006107) |
| 0 | 16 | 1.002736 (0.990368–1.015592) | 1.002732 (0.990085–1.015918) |
| 0 | 85 | 1.057285 (1.033078–1.079073) | 1.057186 (1.033099–1.078990) |
| 0 | 452 | 1.053737 (1.047142–1.058044) | 1.053774 (1.047139–1.058024) |
| 3584 | 2 | 0.863544 (0.846016–0.915251) | 0.863689 (0.847148–0.915423) |
| 3584 | 16 | 1.025833 (0.940283–1.036988) | 1.025852 (0.940201–1.037006) |
| 3584 | 64 | 1.052144 (1.011330–1.091992) | 1.052152 (1.011202–1.091998) |
| 3584 | 384 | 0.981335 (0.967012–0.990022) | 0.981344 (0.967018–0.990025) |

The outcome is mixed. The unadjusted intervals favor widening for P0/B85, P0/B452 and P3584/B64, but favor the prior path for P3584/B2 and P3584/B384. Other intervals include 1. In particular, the long-prefix two-token tail regressed; the unconditional candidate 16 policy is **not promoted**. These separate intervals are not adjusted for multiple shape comparisons and do not establish a universal speedup or an optimal dispatch threshold. The 32MiB allocation also remains part of the tradeoff.

A new, explicitly declared dispatch hypothesis limits widening to batches of at least 64 tokens with at most 512 total visible tokens. Candidate 17 must validate the held-out short-prefix boundary 63/64 and new 32/65/128/256/512 tails, plus the long-prefix controls, with a fresh ten-pair run against build 15. The original diagnostic suite remains the default; selecting `--suite dispatch-validation` records the new suite before timing. This is validation of a fixed proposed policy, not permission to tune a threshold to each noisy winning shape. No candidate 17 timing result is available here.

The rejected build 14 BLAS-attention experiment is distinct: it changed reduction association and failed the unchanged independent RMS gate at context 1024. Candidate 16's exact-array comparisons do not rehabilitate that experiment or relax its numerical limits.

The dispatch-validation suite also predeclares P448/B64 (visible512), P449/B63 (visible512 but below the batch threshold), and P449/B64 (visible513). These distinguish the two dispatch limits. Prefixes448/449 are deliberately noncanonical numerical API fixtures, not browser512-block reuse: each process computes its full identical prefix outside timing, then rewinds before each tail. All new shapes require full-logit and hidden-state bit equality before any measured shape.
