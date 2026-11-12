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

## Performance decision still pending

Candidate 16 is experimental and has not been promoted to the live browser. The next diagnostic compares it directly with frozen build 15 at fixed prefix/tail shapes, using complete full-logit and hidden-state equality before any timing. Prefix 3584 is primed outside the timed command and rewound before each visit; those results will describe retained-prefix tail-prefill commands, including the final vocabulary head and token selection, **not fresh-request TTFT** or a llama comparison. A broader end-to-end comparison is separate. No threshold will be selected simply from the fastest noisy shape.

The rejected build 14 BLAS-attention experiment is distinct: it changed reduction association and failed the unchanged independent RMS gate at context 1024. Candidate 16's exact-array comparisons do not rehabilitate that experiment or relax its numerical limits.
