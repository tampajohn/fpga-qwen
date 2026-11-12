# Qwen3-0.6B FPGA experiment

Custom RTL and software co-simulation for Qwen3-0.6B with n-gram/suffix
speculative decoding. No FPGA or GPU is required to run these experiments.

## Local browser chat

```sh
cd /Users/jadams/workspace/sandbox/fpga-qwen
make setup
make chat
# Open http://127.0.0.1:8765
# Optional: make chat PORT=8766 MODEL=/path/to/local/checkpoint
# Exact-order reference profile: make chat-reference
```

The default backend is now **full-model behavioral SystemVerilog**, compiled by
Verilator. `make chat` builds the optimized CPU profile: thin LTO, validated
projection spans, eight independent accumulators, bounded activation indexing,
optimized attention access, and compact projection transport. It uses the sleeping
condition-variable worker scheduler. `make chat-reference` retains
the original accumulation order. Both are behavioral SystemVerilog, not FPGA
implementations. PyTorch and Transformers are neither installed by the default setup
nor imported by the chat server. Tokenization uses the local checkpoint's
`tokenizers` data and Jinja chat template; Python handles HTTP and token transport.
All model arithmetic executes in `rtl/qwen_behavioral.sv` and
`rtl/projection_worker.sv`: all 28 layers, RMSNorm,
Q/K normalization, RoPE, grouped-query causal attention, SiLU, residuals, final
normalization, the full vocabulary projection, and token selection. KV payloads
are SystemVerilog arrays. The C++ harness only maps checkpoint bytes, transports
weights/token IDs, schedules row partitions, and clocks the simulation; it never invokes BLAS or model math.

**This is a functional simulator, not synthesizable or cycle-accurate RTL.**
It uses BF16 checkpoint weights with SystemVerilog `real` (double) arithmetic,
dynamic KV arrays, and transcendental functions. One clock edge evaluates a whole
token; edge counts do not predict FPGA latency. The earlier synthesizable W4A8
primitive remains a separate experiment because its quantization error is unresolved.
Replacing functional operations with pipelined, fixed-width hardware remains future work.

The browser supports streamed replies, multi-turn context, prefix reuse, Stop,
new conversations, a backend selector, simulation worker controls, and
128/256/512-token reply limits. Eight workers are the default. Changing worker
count between replies preserves KV state; serial baseline is also selectable. The
simulator evaluates prompt tokens sequentially, so first replies take longer;
the UI shows prompt progress. Stop terminates an in-flight simulation promptly,
clears its KV state, and starts a fresh simulator for the next request. Completed
turns reuse the exact matching token prefix, including correct cache truncation
when the chat template retokenizes prior replies. Suffix drafting is disabled
until it is implemented for the full simulator.

The dropdown also retains **llama.cpp FP32**, **llama.cpp FP16**, and **llama.cpp Q4_K_M** as optional
CPU comparison backends; neither uses PyTorch or SystemVerilog. Native GPU backends
are compiled OFF and runtime offload is disabled. Changing backends preserves
conversation messages, but switching engines clears the simulation's KV cache.
Missing native builds/weights appear as unavailable choices. History lives in
browser memory, so refreshing starts a new conversation.

Only `127.0.0.1` is bound. The chat port is 8765, suitable for your own ngrok tunnel
if desired. The server is single-user, with one active reply and no built-in auth.
There is no remote inference or automatic model download. Prompt bodies are not logged.

`POST /api/chat` accepts a session ID, request ID, alternating user/assistant
messages, `backend` (`rtl`, `llama-f32`, `llama-f16`, `llama-q4`), `max_new_tokens`, and
`max_draft` (must be zero), `simulation_workers` (0, 1, 2, 4, or 8),
`decoding` (`chat` by default, or `greedy`), and an optional integer `seed`. Responses are NDJSON `loading`, `start`, `prefill`,
`update`, `done`, or `error` events. `POST /api/stop` cancels by request ID.
`GET /api/status` reports the backend catalog and CPU/framework status.
Concurrent requests receive HTTP 409. Context overflow is rejected, not truncated.

### Chat decoding and repeated replies

Chat now defaults to sampled decoding: temperature 0.7, top-k 20, top-p 0.8,
and a presence penalty of 1.5 over the most recent 256 input/generated tokens.
Control tokens are excluded from the SV presence penalty. These settings follow
[Qwen's non-thinking sampling and repetition guidance](https://huggingface.co/Qwen/Qwen3-0.6B#best-practices).
The sampler's probability calculations, filtering, penalties, and token choice
are SystemVerilog (`rtl/sampling.svh`). Python supplies a uniform random value;
it does not perform logit processing or select a token. An API `seed` makes the
random stream reproducible. The UI retains **Greedy · testing** for numerical
comparisons; low-level simulator tests remain greedy unless sampling is requested.

The reported cross-turn repeat was reproduced with an empty simulator cache and
with llama.cpp FP32 on the same conversation. It was model/decoding behavior,
not a parallel arithmetic or KV-reuse discrepancy. Sampling alone did not reliably
fix that history. The system prompt now identifies the model, and each later user
turn receives a short system reminder to answer that question. Reminders are
retained in subsequent prompts so prefix reuse remains consistent. They consume
context tokens and increase initial prompt processing. The model still has weak
handling of vague follow-ups; these changes do not guarantee that all repetition
or incorrect answers disappear.

`make conversation-test` replays the supplied repetitive history, checks that the
model recovers on identity questions, and checks prefix reuse on the next turn.
`make full-test` also checks SV sampling against independent NumPy top-k/top-p
probabilities and verifies cache truncation of sampling history. The sampled
chat path and the greedy numerical reference are separate decoding modes over
the same logits and weights.

### Parallel simulation

Dense projections, including the vocabulary head, are split by output rows
across persistent CPU threads. Every thread runs its own Verilated
`projection_worker` instance. All dot-product arithmetic remains SystemVerilog;
the C++ pool only partitions work and copies activations/results. Workers share
one read-only checkpoint mapping. The reference profile preserves serial accumulation order. The optimized profile
uses eight partial sums per row for decode/B4 verification and four for wide
prefill, changing floating-point association; it is
validated with numerical tolerances and output checks rather than claiming
bit-identical logits. There is no cross-worker floating-point reduction. Attention,
normalization, nonlinearities, residuals, and KV state remain in SV. Attention
heads run on the persistent SV workers, with the main model owning KV state.
This is shared-memory execution on one machine, not a multi-machine service.

The selected fast build is reproducible with
`uv run --locked tools/build_projection_variant.py --storage scalar --opt O3 --lto --unchecked --masked-activation --unsigned-indices --fast-attention --parallel-attention --batched-prefill --block-unroll --compact-projection --group-projections --prefill-b8 --scalar-group`.
Unsigned indices simplify generated projection address calculations without
changing accumulation order; this incremental change passed bit-exact gates
against the prior fast build. See the [paired comparison](docs/performance_iteration_3.md).
Parallel attention preserves per-head arithmetic order and passed bit-exact
long-context gates; see the [paired validation](docs/attention_paired_validation.md).
These options preserve BF16 checkpoint storage and SV FP64 arithmetic; they do
not implement FP16 inference. `make fast-test` checks full-logit tolerances,
same-uniform sampling, KV truncation/reset/restart, invalid projection spans,
worker scheduling, attention head partitions against the preceding unsigned
binary, and the conversation recovery fixture. These short checks
complement the matched benchmark's quality suite; they are not a broad quality guarantee.

The original parallel implementation measured the following on this Apple M5 Max,
with two warmup tokens and nine measured
short-context token forward passes (including Python/process IPC):

| Execution | Median tok/s | Relative to serial |
| --- | ---: | ---: |
| Original serial path | 1.49 | 1.00× |
| 1 partition worker | 1.34 | 0.90× |
| 2 workers | 2.57 | 1.72× |
| 4 workers | 4.35 | 2.91× |
| 8 workers | 6.87 | 4.60× |

Every logit was bit-identical to the serial baseline at all nine measured
positions for every worker count. A separate regression covers runtime resizing,
uneven partitions (three workers), cache truncation, and reset. One worker is
slower because dispatch adds overhead without parallel compute. These are
simulator timings, not FPGA estimates, and exclude initial loading and prompt
prefill. Longer-context performance will differ.

Reproduce with `uv run --locked tools/benchmark_parallel.py`; raw samples and
projection benchmarks are saved to `build/parallel-benchmark.json`. Start with a
specific count using `uv run --locked tools/chat_server.py --workers 4`, or
change **Simulation workers** in the browser. The default server uses the fast
profile; add `--simulator reference` for exact-order worker arithmetic. `0` selects
the serial
implementation. Resizing happens at request boundaries and does not clear KV.

The fast profile prefills uncached prompts in blocks of up to eight through all
transformer layers. Compared with the previous grouped B4 build, measured
first-token wait fell about 15–16% and total 32-token request latency fell about
11–14% on the tested 85/452-token contexts. Decode and suffix verification retain
their existing paths; this is not a claim of improved steady decode throughput or
parity with llama.cpp. See [B8 measurements and validation](docs/b8_prefill.md).

The fast profile continues to group Q/K/V and gate/up projections sharing an input.
This removes repeated worker dispatches and activation copies while preserving
the same SV arithmetic. Four matched pairs showed about 4–12% faster short
prefill and 6–11% faster long prefill, with bit-identical outputs; see
[grouped projection evidence](docs/grouped_projection_execution.md). This does
not remove weight reads or dense matrix operations. The browser displays
first-token latency separately from generation throughput.

Single-token decode also groups Q/K/V and gate/up work, removing 84 worker
synchronization rounds per token while retaining identical SV arithmetic.
Paired median decode speed ratios were 1.126× short and 1.196× long, with
substantial variation in the long runs. Observed median throughput was
40.80→46.94 and 30.35→34.08 tok/s respectively; see
[decode grouping evidence](docs/scalar_projection_groups.md).

The browser's **Suffix drafting** checkbox is available for this profile and
starts off. It proposes up to three cached n-gram continuation tokens and verifies
them with the SV block engine; sampled decoding preserves the same seeded target
sequence. Acceptance does not guarantee a speedup, so enable it for experiments.
The reference profile keeps single-token prefill and does not advertise suffix
support. See [the block runtime](docs/batched-prefill-runtime.md) and
[suffix verification](docs/suffix_chat.md) for protocol and validation limits.

### Validation without PyTorch

The [held-out distribution probe](docs/approximate_projection_quality.md) compares
candidate logits with the current SV baseline using authored reference continuations.
It reports token-weighted loss changes, KL divergence, and token agreement; its
small corpus is a regression probe, not proof of broad model quality. `make quality-test`
checks the scorer and token alignment without loading a model. The standalone
[W8A8 experiment](docs/w8a8_microkernel.md) passed integer correctness checks but
was slower than the optimized FP64 worker and is not used by chat.

```sh
make check
# Or the focused full-model and chat checks:
make full-test fast-test chat-test conversation-test
# With make chat running:
uv run --locked python tests/browser_chat.py
```

The full-model test compares all 151,936 logits against an independent NumPy FP32
implementation, including multiple causal positions, KV truncation, and reset.
Observed maximum absolute logit error was 2.44e-5, with identical greedy choices
for the tested positions. Reset reproduces the original simulator logits exactly.
The simulator uses higher precision internally than the NumPy oracle; bitwise
identity with FP32 is not expected. This is numerical validation on a small test
sequence, not a broad model-quality evaluation.

Chat tests cover framework isolation, input validation, prefix reuse, cancellation
inside a token, restart after cancellation, HTTP concurrency, and origin checks.
Browser tests cover actual replies, switching between simulator/native engines while
preserving conversation, Stop, context errors, and a 390px mobile viewport.

Historical PyTorch reference tools remain as optional experiments, isolated from
the server. They require `uv run --extra reference ...`; `make block-test` and
`make reference-suffix-test` explicitly opt in. Default `make check` does not.

The earlier experiments are retained alongside the full-model simulator:

| Experiment | Runs in Verilator RTL | Runs on CPU |
| --- | --- | --- |
| Primitive | Signed W4A8 matrix engine | Independent integer golden model |
| Complete decoder block | All seven full linear projections; KV transaction metadata | FP32 norms, RoPE, causal grouped-query attention, SiLU, scales; FP16 KV payload |
| Full 0.6B suffix decoding | KV transaction metadata | Entire FP32 language model and longest-match n-gram drafter |

**The complete 28-layer model now runs in behavioral SystemVerilog, but not
synthesizable RTL.** The older CPU suffix experiments establish acceptance and
output correctness, not FPGA performance. No synthesis, placement, resource fit, or hardware clock has been
established. Simulated matrix cycles exclude CPU operations and transfers.

## Run

Requires Python 3.12, uv, Verilator, Make, and a C++17 compiler. The optimized
profile additionally requires Clang thin LTO; use `make chat-reference` for the
portable exact-order build. The primitive and
KV cursor tests also work with a standalone Python 3 standard library.

```sh
cd /Users/jadams/workspace/sandbox/fpga-qwen
make setup           # isolated .venv, dependency versions in uv.lock
make check           # default checks, without PyTorch

# Individual experiments:
make test            # original 128 primitive scenarios
make kv-test         # capacity, rejection, reset, backpressure, randomized commands
make block-test      # one complete decoder block against HF and integer oracles
make suffix-test     # n-gram unit tests only
make full-test       # full SystemVerilog model vs independent NumPy oracle
make reference-suffix-test # optional historical PyTorch experiment
make chat-test       # full-simulator chat + HTTP protocol checks

# Or specify a local unsharded BF16 checkpoint:
make check MODEL=/path/to/Qwen3-0.6B
```

The default checkpoint is discovered in the local Hugging Face cache. Model and
tokenizer loading is local-only. No model is downloaded. Generated outputs live
under `build/`; `make clean` removes those outputs and simulator binaries.

| Output | Contents |
| --- | --- |
| `build/results.csv` | Primitive cycles, traffic, activation preload and stalls |
| `build/manifest.json` | Checkpoint provenance, dimensions, row scales, weight error |
| `build/summary.txt` | Primitive comparison; not tokens/sec |
| `build/block/report.json` | Independent full-block comparisons and RTL measurements |
| `build/suffix_report.json` | Real candidate acceptance, exact-output checks and per-round cache trace |

## Results from the initial local run

- 128 primitive scenarios pass; all signed integer results match the reference.
- Complete block: 159,744 RTL integer results match, across 28 full projection
  calls. The block's FP32 path agrees with the stock Hugging Face Qwen3 decoder
  layer within approximately 1.2e-6 maximum absolute error on the test input.
- The W4A8 path agrees with a separately quantized HF block within approximately
  4.8e-6. This checks implementation agreement, not quantization quality.
- Rejecting two of four candidates, poisoning the rejected cache tail with
  NaNs, and inserting a correction gives the same output as a fresh run.
- Prefix truncation/reuse and batched/serial execution agree on the test input.
- KV tests cover exact capacity, overflow, partially written transactions,
  zero acceptance, abort, reset, held responses, and 2,000 randomized commands.

Full-model **FP32 CPU** greedy suffix smoke tests, at most 48 new tokens:

| Workload | Output tokens | Verification rounds | Accepted/proposed | Tokens/round |
| --- | ---: | ---: | ---: | ---: |
| Repeated phrase | 48 | 16 | 32 / 38 | 3.00 |
| Copy repeated code | 36 | 8 | 28 / 35 | 4.50 |
| Open prose | 37 | 36 | 1 / 7 | 1.03 |

All outputs exactly match ordinary greedy generation. These are three short
smoke tests, **not a representative acceptance benchmark or wall-clock speedup**.
CPU experiment time includes bookkeeping and subprocess interaction and is not
a hardware estimate. Quantized-model acceptance may differ from FP32.

**Quantization remains unresolved:** experimental per-row W4A8 produces about
21% relative block-output RMSE versus the FP32 block on this one prompt. This is
an activation-error diagnostic, not a task-accuracy score. Groupwise quantization
and calibration should be investigated before making quality claims or extending
the numerical format to all 28 layers.

## Matrix engine

`rtl/q4_matvec.sv` has 16 signed int4 weight lanes per beat (64 bits) shared by
up to eight token positions, int8 activation banks, and signed int32 accumulators.
Row scales are applied in software. It is synthesizable-style RTL but currently
uses parallel activation reads and combinational multiplication/reduction.
Banking, pipelining, and device synthesis are needed before assuming BRAM use,
clock speed, or area.

Protocol: initialize every used activation index while idle; pulse `start` with
`token_count` in 1..8; supply K/16 weight beats with `w_last` on the last beat;
consume the held result using `y_ready`. K must be a positive multiple of 16,
at most 3,072. Load and start on separate cycles. No writes during a row. Reset
invalidates the transaction but does not clear activation RAM.

The primitive test uses eight actual rows from each layer-0 projection, full
input widths, deterministic synthetic activations, signed extrema, two seeds,
output stalls, and two memory delivery rates. The same eight-token hardware is
used at every width: width 1 leaves seven lanes idle. Wider execution uses more
parallel arithmetic, not free compute. The test memory source provides a 12-cycle
row startup and a 64-bit beat every 1 or 4 cycles. It is not a DDR/HBM model.

The full-block test uses **all rows and all input columns** of the seven real
projections, and actual token embeddings. Its separate linear runner supplies
one 64-bit weight beat/cycle with no simulated external-memory latency. Every
int32 output is checked against a NumPy dot product. A second oracle uses the
stock HF decoder layer, with separate PyTorch W4A8 linear wrappers, to check the
normalization, head layout, RoPE, causal mask, residuals and MLP implementation.

## KV transaction controller

`rtl/kv_cursor.sv` controls one sequence's contiguous KV ownership. Capacity is
4,096 tokens, with up to eight tentative positions per transaction. It stores
**metadata only**; the payload is presently in CPU arrays/PyTorch's cache.

| Opcode | Operation | Rule |
| --- | --- | --- |
| 0 | Reserve count slots | Idle, 1..8 slots, within capacity |
| 1 | Mark count writes complete | Active, no more than remaining reserved slots |
| 2 | Commit count prefix slots | All reserved writes complete; count may be zero |
| 3 | Abort | Active; count must be zero |
| 4 | Truncate to count | Idle; count cannot exceed committed length |

Invalid commands leave ownership unchanged and return an error. Responses stay
stable under backpressure. `committed` is the next reservation's base;
`visible_end = committed + written` bounds initialized payload. Causal attention
must impose its own per-query limit within this visible range. Rejected tail
payload is not erased; it becomes inaccessible and can be overwritten. There is
no sequence multiplexing, paging, DRAM interface, or all-layer completion
tracking yet. The caller must only mark a token written when all required KV
payload for that token is complete. The full-model test coordinates that after
the complete model call.

## Suffix verification

`tools/suffix.py` implements a simple longest-match n-gram lookup, not the full
published SuffixDecoding tree algorithm. It searches prior text for a match to
the current suffix and proposes known continuations. The model test uses only
the current prompt and already generated tokens; no baseline/future output is
fed into the drafter. Optional history documents remain separate in the lookup.

`tools/suffix_decode.py` keeps the final context token pending outside the KV
cache. Each target call processes `[pending, proposed tokens...]`, accepting
only the contiguous matching prefix and emitting the target correction/bonus.
The cache is cropped and the RTL cursor committed to the corresponding prefix.
The output is compared token-for-token against HF greedy generation. This flow
supports greedy decoding only, not temperature sampling. Prefix KV reuse is
distinct from suffix token drafting; stale suffix-KV reuse is not implemented.

## Model sizing and next steps

The config has 28 layers, hidden width 1,024, intermediate width 3,072, 16 query
heads, 8 KV heads, and head dimension 128. Query width is 2,048, not hidden width.
The checkpoint contains duplicate tied embedding/output tensors; byte equality
is verified before counting them once. Unique count is 596,049,920 parameters:
ideal all-int4 storage is 298.02 MB, excluding scales and higher-precision tensors.
FP16 KV requires 114,688 bytes per token, or 448 MiB at 4,096 tokens per sequence.

Next: improve/calibrate quantization, pipeline and bank the matrix primitive,
move attention/KV payload access into RTL, then implement the nonlinearities and
full model. A physical FPGA target is still needed for synthesis constraints and
realistic memory-controller integration. Do not turn matrix cycles or accepted
tokens/round into tokens/sec without accounting for those components.

## References

- [Qwen3-0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
- [Verilator](https://verilator.org/guide/latest/overview.html)
- [SuffixDecoding](https://arxiv.org/abs/2411.04975)

### Local llama.cpp comparison

The [combined hybrid CPU comparison](docs/attention_combined.md)
measured approximately 60/49 tokens/s for short/long prompts, versus 139/127 for
llama.cpp FP16 and 95/87 for FP32. It remains behind llama.cpp and is not the live default.
When its fixed local artifact is built, choose **SystemVerilog · FP32 hybrid** in
the browser backend selector. This option uses FP64 B8 prefill and FP32 scalar
decode projections with [validated combined FP64 attention](docs/attention_combined.md),
which improved long-context decode by 1.25× against the preceding hybrid in a
direct paired comparison; short-context total time was neutral/mixed. Runtime precision is
verified before inference. The backend reuses only complete
FP64 prefill blocks through the [validated cache policy](docs/canonical_prefix_reuse.md).
Optional suffix drafting uses verified FP32 B4 execution and stays off by default;
it helped repetitive code in the bounded tests, not ordinary prose. Serial execution
remains disabled. See [the build and routing policy](docs/fp32_hybrid_runtime.md).

The [current deployed-backend comparison](docs/current_native_comparison.md)
measures B8 prefill and grouped decode against CPU-only llama.cpp: approximately
50/33 tokens/s for short/long prompts versus 120/122 with native FP16. It includes
first-token latency, identical-token checks and measurement limitations.

On this Apple M5 Max, using the same Qwen3-0.6B checkpoint, 512 tokens of
context, 128 generated tokens and three repetitions, CPU-only llama-bench measured:

| Weights | 4 threads | Default 6 threads |
| --- | ---: | ---: |
| FP32 | 66.19 tok/s | 79.62 tok/s |
| Q4_K_M | 138.89 tok/s | 173.95 tok/s |

These are synthetic decode rates, excluding sampling, tokenizer, browser and RTL
overhead. The reported 40 tok/s in the reference app is not a matched-prompt run.
In that historical benchmark, native KV was FP16 and the PyTorch reference KV was FP32. Q4 changes weight precision and can change
answers. Raw measurements and commands are in `build/bench/comparison.json`;
rerun with `uv run tools/benchmark_llama.py`.

The native source is in `build/llama.cpp`, build in `build/llama-cpu`, and models
in `build/bench/qwen3-0.6b-{f32,q4_k_m}.gguf`. GPU backends METAL, CUDA,
VULKAN and OPENCL are compiled OFF; the adapter checks the CMake configuration
and also disables GPU layers, KV offload and operation offload at runtime.
The GGUF files were converted from the same local Hugging Face checkpoint.
`make clean` removes these optional build artifacts along with the simulators.

## Coordinated performance and quality experiments

Run scored jobs through the serving checkout's benchmark guard:

```sh
.venv/bin/python tools/benchmark_guard.py --name "CPU inference comparison" -- COMMAND ARGUMENTS
```

The browser polls `/api/status` every second and shows the run name while the
guard is held. Drafts are preserved, but new chat requests receive HTTP 409 until
the run finishes. The last run name and finish time remain visible afterward.
The guard also refuses to start during an existing reply.
Its lock is released by the OS when the wrapper and inherited child exit, so a
stale status file cannot leave chat disabled. Use the guard from the checkout
serving the browser, even when `COMMAND` benchmarks a different worktree.
Unwrapped external processes are not detected by this flag.

See [the matched benchmark protocol](docs/BENCHMARK_PROTOCOL.md) for identical
prompt/token workloads, separate TTFT and decode rates, precision disclosures,
and measured comparisons with llama.cpp. [Projection experiments](docs/projection_optimization.md)
explain the compiler and reduction changes. Tracked `reports/` contains raw
synthetic benchmark/quality evidence; local binaries and model files remain ignored.

[Conversation findings](docs/conversation-quality-findings.json) distinguish
numerical agreement from answering correctly. The frozen objective suite includes
recall, correction, quotation, task switching, extraction and arithmetic. Its low
native-model pass rate is a model/decoding limitation, not evidence that matching
those answers is high quality. No GPU or PyTorch is required for these experiments.

The optional **FP32 hybrid · 12 prefill / 8 decode** backend uses the same
validated hybrid arithmetic and canonical prefix policy, with fixed worker counts
for each phase. Both pools remain resident (20 worker threads); only one receives
work at a time. Its worker control is disabled, while the existing FP32 hybrid
keeps its configurable worker count. The option appears only when the trusted
phase-pool artifact is installed. It verifies the numerical profile and phase
policy on every request, including process restarts. Suffix drafting remains off
by default. This option does not change the default FP64 backend.

The optional **Fused FP32 prefill · 12 prefill / 8 decode** backend uses explicitly
fused FP32 prefill arithmetic, separate FP32 decode and verification, and FP64
attention/KV. It reuses complete fused B8 prefill blocks under its own cache policy.
Worker counts are fixed; 20 worker threads remain resident with one pool active.
Existing FP64 and hybrid/phase choices remain available. The browser selects the
newest available backend on initial page load and preserves manual selection
during the session. Suffix drafting
is optional and off by default. This remains CPU behavioral simulation, not FPGA
cycle timing.
