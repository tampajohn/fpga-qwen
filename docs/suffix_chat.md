# Target-verified suffix chat and block prefill

The opt-in block backend batches up to four prefill tokens and can verify n-gram suffix drafts in the same causal SystemVerilog block. Python proposes known tokens from the existing context, supplies random uniforms, and handles transport; all target logits, sampling, model arithmetic, and KV payload stay in SystemVerilog. The default remains token-only until a block-capable binary is explicitly selected.

A request may ask for a draft length up to seven, but the initial block size limits the effective draft to **three** tokens. The UI shows that limit and exposes acceptance counts. Native backends and token-only simulator configurations explicitly reject suffix drafting.

For each verification block, the inputs are the pending token followed by up to three proposed tokens. The engine accepts only the proposal prefix matching the target's predictions, then emits the target correction or bonus token. It truncates KV to the actually consumed inputs before exposing output. EOS, the reply limit, context capacity, cancellation, and session-prefix reuse constrain this process. Speculative future KV never survives rejection.

Sampled chat preserves sequential target sampling: the engine clones the request RNG for speculative uniforms, then advances the real RNG by only the number of emitted tokens. Predictions after a rejection do not consume future request randomness. Thus this implementation can support both sampled and greedy requests, subject to the real binary's numerical gates.

## Correctness evidence

Eight host orchestration tests cover real n-gram acceptance/bonus behavior, sampled rejection and RNG continuation, EOS rollback, output/context limits, prefix reuse, cancellation, disconnect cleanup, and wire validation. HTTP and existing input-validation tests also pass.

Against the initial B4 binary, actual greedy and sampled requests with seeds 42 and 43 produced identical text and exact cached token IDs with suffix disabled/sequential prefill versus suffix enabled/block prefill. Each repeated-sequence case proposed 12 draft tokens, accepted five, evaluated 19 target predictions, and emitted 11 tokens in seven rounds. These runs exercise both acceptance and rejection.

A real browser smoke test additionally passed the enabled suffix control, a sampled follow-up, 93 reused prefix tokens, Stop, context-limit errors, mobile layout, and no JavaScript errors. The report [suffix-chat-correctness.json](../reports/suffix-chat-correctness.json) identifies the tested binary and retains raw synthetic outputs. Its timing fields are correctness-run observations, **not speed claims**. Optimized block binaries require their own gates and performance measurements.

## Candidate server

```sh
uv run --locked python tools/chat_server.py --port 8766 \
  --sv-binary /path/to/tested/block-capable/Vqwen_behavioral --block-size 4
CHAT_URL=http://127.0.0.1:8766 uv run --locked python tests/browser_suffix.py
```

The executable override is a trusted local CLI argument, never an HTTP input. Selecting `--block-size 4` against a token-only executable is invalid. No live server was changed by these tests.

## Optimized B4 verification and suffix timing

The optimized B4 executable SHA256 `d576cde12f0fc64499b35647878fe2ddea21b9fdd3c5459f14dd96cf58a36272` passed the same real greedy and seeded sampled parity checks, now comparing exact output token IDs as well as text, finish reason, and cached IDs. The block-specific browser smoke passed again. Evidence is in [suffix-optimized-correctness.json](../reports/suffix-optimized-correctness.json). The independent block-validation work separately measures the prefill improvement; the suffix comparison below keeps **B4 prefill enabled in both modes**.

Two alternating pairs per workload compared suffix off/on, then on/off, using one resident process, eight workers, sampled decoding with seed 42, identical prompts, a 64-token reply limit, and empty logical KV for every request. Warmup/startup is excluded. All paired output IDs and finish reasons matched exactly. No other agent inference or compilation ran during scored requests; user applications remained running.

| Workload / mode | Median TTFT | Median prefill | Median decode | Median total request |
|---|---:|---:|---:|---:|
| Repeated code / off | 1.172 s | 1.150 s | 0.902 s | 2.053 s |
| Repeated code / on | 1.192 s | 1.169 s | 0.687 s | 1.856 s |
| Ordinary prose / off | 1.091 s | 1.068 s | 0.810 s | 1.878 s |
| Ordinary prose / on | 1.129 s | 1.106 s | 0.793 s | 1.899 s |

The repeated-code reply emitted 39 target tokens, accepting all 24 proposed draft tokens. Its two paired decode speed ratios were 1.311× and 1.317×; complete-request ratios were 1.105× and 1.106×. Ordinary prose emitted 34 target tokens and accepted two of three proposals: decode ratios were 1.017× and 1.026×, but complete requests were slightly slower (0.989× in both pairs). Prefill and TTFT variation must not be attributed to a different prefill algorithm: both modes use the same B4 path.

Decode time includes the first target prediction/verification round, avoiding misleading rates calculated from chunked UI arrivals. TTFT overlaps prefill and the first decode round and must not be added to the other columns. Two pairs are a small local sample, not a general performance guarantee. The evidence supports an **optional verified draft mode, unchecked by default**, especially for repetitive content; it does not support claiming suffix drafting speeds up every reply.

Raw paired evidence: [suffix-paired.json](../reports/suffix-paired.json). Reproduce with `tools/benchmark_suffix_chat.py --binary /path/to/tested/block-capable/Vqwen_behavioral --pairs 2 --tokens 64 --seed 42 --decoding chat`. Real parity is opt-in via `SV_BLOCK_BINARY=/path/to/binary uv run --locked python -m unittest tests.test_suffix_real -v`.
