# Shared FP32 KV scratch for batched prefill (candidate 16)

**Decision: reject both scratch candidates 16 and 17; keep live build 12 selected.**
The initial gains did not generalize to the predeclared dispatch-validation
shapes. The scratch changes were reverted by commits ff54973 and 2b431be;
immutable binaries and all evidence remain preserved. There is no promotion or
further threshold tuning based on these results. See the final confirmation
section below for all 14 shapes and their uncertainty.

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

A new, explicitly declared dispatch hypothesis limits widening to batches of at least 64 tokens with at most 512 total visible tokens. Candidate 17 must validate the held-out short-prefix boundary 63/64 and new 32/65/128/256/512 tails, plus the long-prefix controls, with a fresh ten-pair run against build 15. The original diagnostic suite remains the default; selecting `--suite dispatch-validation` records the new suite before timing. This is validation of a fixed proposed policy, not permission to tune a threshold to each noisy winning shape. The subsequent candidate 17 confirmation is reported below.

The rejected build 14 BLAS-attention experiment is distinct: it changed reduction association and failed the unchanged independent RMS gate at context 1024. Candidate 16's exact-array comparisons do not rehabilitate that experiment or relax its numerical limits.

The dispatch-validation suite also predeclares P448/B64 (visible512), P449/B63 (visible512 but below the batch threshold), and P449/B64 (visible513). These distinguish the two dispatch limits. Prefixes448/449 are deliberately noncanonical numerical API fixtures, not browser512-block reuse: each process computes its full identical prefix outside timing, then rewinds before each tail. All new shapes require full-logit and hidden-state bit equality before any measured shape.

## Candidate 17: restricted dispatch correctness before timing

Candidate 17 fixes the proposed dispatch policy before measurement: widen only when **batch ≥64 and total visible tokens ≤512**. Other batched prefill uses the original paired FP16 attention path; ordinary B1 and transactional B2 remain unchanged. The condition is evaluated per prefill chunk. The allocated scratch still costs **32 MiB at context4096** per model instance even when a request does not use it; dispatch does not remove that memory cost. The [immutable build manifest](../reports/native-cpu-build-17-build.json) preserves source and binary hashes.

The restricted candidate passed the [complete 56-array comparison](../reports/native-cpu-build-17-exact-gate.json) against frozen build 12 and [28 prefix cases](../reports/native-cpu-build-17-prefix-check.json), including cancellation/restart and rewind checks. Its [35 suffix fixtures plus seven cache/session cases](../reports/native-cpu-build-17-suffix-trajectories.json) retained exact trajectories and stops, exercised 63 proposals with 32 accepted, and passed cancellation/restart. The [five CLI lifecycle cases](../reports/native-cpu-build-17-suffix-cli.json) passed exact committed-state and EOS/error checks. The [nine-case tuned-llama quality comparison](../reports/native-cpu-build-17-quality.json) passed exact trajectories/no-new-regressions, with both engines still scoring **4/9** under the unchanged rubric.

These are correctness results on declared fixtures and hardware, not a formal proof or a speed claim. The 14 shape diagnostics were declared before the subsequent confirmation: P0/B32,63,64,65,128,256,512; P448/B64; P449/B63,64; and P3584/B2,16,64,384. They include new boundary shapes and repeated long-prefix controls; they are not all previously unseen workloads. All shapes must match full logits and hidden state before timing. Candidate 17 is not promoted, and no universal threshold, fresh-request improvement or llama advantage is established here.


## Candidate 17 confirmation: reject restricted widening

The [complete dispatch-validation report](../reports/native-cpu-build-17-dispatch-shapes.json) preserves all **140 measured pairs**, ten per shape, plus all 14 exact full-logit/hidden-state preflights. Two symmetric warmups preceded each shape's measurements; configuration order alternated. No measured rounds or shapes were excluded. Reference is frozen build 15, candidate is 17, and ratios above 1 favor the candidate. All preflight arrays and measured predicted tokens matched.

| Prefix P / tail B | Candidate 17 path | Paired CLI speedup (95% interval) | Paired pipe-wall speedup (95% interval) |
|---|---|---|---|
| 0 / 32 | FP16 paired | 0.995855 (0.981033–1.015654) | 0.995903 (0.981085–1.015614) |
| 0 / 63 | FP16 paired | 0.999549 (0.947070–1.042567) | 0.999526 (0.946937–1.042485) |
| 0 / 64 | FP32 scratch | 0.994175 (0.975540–1.010601) | 0.994218 (0.975443–1.010580) |
| 0 / 65 | FP32 scratch | 0.984647 (0.965288–1.000102) | 0.984681 (0.965551–1.000080) |
| 0 / 128 | FP32 scratch | 0.982193 (0.958703–0.999050) | 0.982135 (0.958743–0.999036) |
| 0 / 256 | FP32 scratch | 0.972516 (0.963669–0.979012) | 0.972538 (0.963629–0.979020) |
| 0 / 512 | FP32 scratch | 0.960014 (0.949599–0.967456) | 0.960039 (0.949616–0.967462) |
| 448 / 64 | FP32 scratch | 0.922129 (0.878545–0.933761) | 0.922187 (0.878594–0.933742) |
| 449 / 63 | FP16 paired | 0.950219 (0.944698–0.958536) | 0.950275 (0.944675–0.958642) |
| 449 / 64 | FP16 paired | 0.950559 (0.924459–0.953819) | 0.950575 (0.924526–0.953820) |
| 3584 / 2 | FP16 paired | 0.997544 (0.978732–1.010174) | 0.997599 (0.978702–1.010058) |
| 3584 / 16 | FP16 paired | 0.966617 (0.873650–0.988463) | 0.966627 (0.873777–0.988486) |
| 3584 / 64 | FP16 paired | 0.918708 (0.879856–0.943755) | 0.918746 (0.879877–0.943792) |
| 3584 / 384 | FP16 paired | 0.954119 (0.907455–0.994924) | 0.954117 (0.907474–0.994913) |

New P0/B128,256,512 and P448/B64 sizes did not reproduce the earlier observed gains. Several shapes that bypass widening also ran slower, including P449/B63,64 and longer-prefix controls. This prevents attributing the entire outcome specifically to conversion scratch: the processes were persistent and not hard-pinned, so worker placement, scheduling, resident-memory differences and background activity remain possible influences. The experiment does not distinguish their contributions. The uncertainty intervals are separate, unadjusted per-shape summaries, not a proof of a universal slowdown or a causal explanation.

There is nonetheless **no evidence supporting deployment of this dispatch policy**. The earlier favorable shapes were insufficient to select a general policy, and the fixed follow-up did not generalize. Both scratch source changes are reverted; the ordinary FP16 path is restored while the independent B2 verification work remains separate. Candidate 16/17 binaries and positive, neutral and negative measurements are retained. Live build 12 remains selected; no microtuning or selective exclusion is used to turn the experiment into a speed claim.

These measurements include the tail-prefill command's final vocabulary head and greedy selection. Prefix priming and rewind are outside the timed command. They are not fresh-browser TTFT, a tuned-llama comparison, or a direct measurement of attention alone.
