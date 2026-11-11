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
