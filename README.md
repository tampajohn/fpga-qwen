# Qwen3-0.6B CPU inference experiment

An original, experimental CPU inference engine for Qwen3-0.6B, with local
browser chat and reproducible comparisons against tuned llama.cpp. The current
implementation targets Apple ARM CPUs and uses C++20, ARM SIMD, and Accelerate.
It does not use llama/ggml neural kernels, a GPU, PyTorch, or Transformers for
inference. Python handles tokenization, transport, and evaluation; offline weight
conversion and the independent numerical oracle use NumPy.

This project began as an FPGA/behavioral SystemVerilog experiment. Those sources
and results remain available, but simulator performance development is frozen.
The behavioral simulator is neither synthesizable nor cycle-accurate; its CPU
timing does not predict FPGA performance. Current work lives in [`engine/`](engine/).

## Current result

The first original-engine build, **build-02**, passed full-vocabulary numerical
gates at 16 teacher-forced positions: contexts 1–4, 85–92, and 452–455. All greedy
IDs matched the independent NumPy oracle. The predeclared limits are maximum
absolute error ≤0.01 and RMS error ≤0.002, with finite logits and greedy identity.
It also matched tuned llama.cpp's exact token trajectories, EOS behavior, text,
and grades on the existing nine-case conversation suite. Both scored **4/9**:
matching baseline mistakes is not a model-quality improvement.

**Build-10 is the selected native candidate.** It combines persistent workers,
grouped/four-row projection dispatch, larger prefill chunks, vector-library
nonlinear functions, and balanced causal-attention work during prefill. Its
full logits are bit-identical to build-08 at all 22 checked positions, including
511/512/513-token chunk boundaries; build-08 passed the unchanged NumPy gates
there. The nine-case exact-trajectory suite still matches llama.cpp (both 4/9).

Thirty alternating paired rounds used 85/452-token uncached prompts and 32
identical greedy output tokens, with three warmups before each measured profile:

| Engine, 12 decode / 6 prefill threads | 85-token prompt decode | 452-token prompt decode | First token, short / long |
| --- | ---: | ---: | ---: |
| Tuned llama.cpp FP16, KleidiAI + LTO | 134.42 tokens/s | 129.18 tokens/s | 0.216 / 0.628 s |
| Original CPU engine, build-10 | 135.32 tokens/s | 129.50 tokens/s | 0.124 / 0.532 s |

Median **paired** native/llama decode ratios were **1.028x** (95% paired-bootstrap
interval 0.954–1.079) and **0.980x** (0.905–1.022). This meets the Phase 1
within-10% competitive target on these workloads; **no decode win is established**.
Paired first-token speedups were 1.647x (1.563–1.911) and 1.143x (1.099–1.241).
Paired ratios are not quotients of the table's independent medians.

See [confirmation](reports/native-cpu-build-10-confirmation.json),
[paired uncertainty](reports/native-cpu-build-10-confirmation-analysis.json),
[conversation quality](reports/native-cpu-build-10-quality.json), and
[exact-logit comparison](reports/native-cpu-build-10-exact-gate.json).
Underlying NumPy coverage includes [511](reports/native-cpu-boundary-511-oracle-gate.json),
[512](reports/native-cpu-boundary-512-oracle-gate.json), and
[513-token boundary gates](reports/native-cpu-boundary-513-oracle-gate.json).
Earlier and negative results remain in the [current plan](docs/native_cpu_plan.md).
Candidate-09's auxiliary cache was rejected: it added 1.19 GB and regressed short
context decode without an established long-context gain.

These are single-stream measurements on this Apple M5 Max. Native subprocess
transport differs from llama.cpp HTTP streaming; native prefill retains a
1.76 GB widened FP32 layer cache; background scheduling remains uncontrolled.
The numerical suite does not establish accuracy at every supported context or
answer quality beyond the stated tests. Build-10 is deployed locally and passed
[actual live-browser checks](reports/native-cpu-build-10-browser-live.json):
latest-first/default selection, multi-turn chat, cancellation, backend switching,
and the browser-polled benchmark lock. Runtime checks cover worker counts 1/64,
idle/reset cycles, malformed packs, and context-limit handling.

The primary target remains the tuned **FP16** baseline. FP32 and constrained
KleidiAI-OFF Q4 measurements are separate precision classes, not interchangeable
quality/performance claims. See the [current plan and evidence](docs/native_cpu_plan.md)
and [matched benchmark protocol](docs/BENCHMARK_PROTOCOL.md). Historical results
and implementation notes are retained in [`docs/`](docs/) and [`reports/`](reports/).

## Local browser chat

The existing browser service supports streamed replies, conversation history,
Stop, new chats, backend selection, sampled/greedy decoding, and reply limits.
It binds only to localhost, with one active request. It performs no automatic
model download. Supply a local unsharded Qwen3-0.6B checkpoint containing
`model.safetensors`, configuration files, and tokenizer files.

Install the locked Python dependencies, prepare the native assets below, then start:

```sh
make setup
.venv/bin/python tools/chat_server.py --native-only --model /path/to/Qwen3-0.6B --port 8765
```

Open **http://127.0.0.1:8765**. The installed native release appears first and
is selected automatically. The local deployment uses **build-10**, with 12
decode / 6 prefill threads; simulator worker controls do not apply. The
`--native-only` launcher needs no simulator binary or simulator imports. Omit
that flag to include any installed legacy simulator choices. Starting the
server does not build the engine, export weights, or download a model.

Native prefix reuse and suffix drafting are unsupported: every message processes
the full conversation again. Stop terminates the active native process, and the
next request starts with fresh KV state. Benchmark jobs block generation and
show a banner through browser polling.

Installed llama.cpp choices remain available; the default launcher also offers
installed simulator choices. The browser llama.cpp entries currently use the
existing six-thread comparison build; the published contest above uses a
separate tuned 12/6-thread build. Their caching and suffix capabilities differ
and are displayed per backend. Do not infer a
native feature from a simulator control. Browser history is held in memory and
is lost on refresh. The server is single-user without built-in authentication;
exposing localhost through a tunnel requires your own access controls.

## Explicit native build and weight export

Use macOS ARM with the Xcode command-line compiler and Accelerate available.
The original engine uses FP16 matrix weights, FP32 normalization weights and
arithmetic, and FP16 current/past KV storage. The embedding is tied to the output
head. Its [weight and process contract](engine/INTERFACE.md) describes the pack,
precision, and persistent protocol.

Build and export are manual operations. Use fresh build directories; outputs
are immutable. Run CPU work through the shared benchmark guard so the browser
shows the active job and rejects overlapping generation:

```sh
.venv/bin/python tools/benchmark_guard.py --name 'Build native CPU engine' -- \
  .venv/bin/python tools/build_native_cpu.py --directory build/native-cpu/my-build

.venv/bin/python tools/benchmark_guard.py --name 'Export native CPU weights' -- \
  .venv/bin/python tools/export_native_cpu_weights.py \
  --checkpoint /path/to/Qwen3-0.6B --output build/native-cpu/weights.bin
```

The exporter verifies all 311 checkpoint tensors and the tied output head, then
writes 310 aligned records to `weights.bin`, its `.index`, and its `.manifest.json`.
It rejects nonfinite values, FP16 overflow, invalid tensor ranges, and existing
output paths. The manifest records hashes and rounding rules.

Browser availability additionally requires an explicitly installed executable
at `build/native-cpu/qwen_cpu`. Builds remain in their own directories until
correctness, performance, and browser checks justify installing that stable
path. After validating a new build, install it explicitly, for example:

```sh
ln -s my-build/qwen_cpu build/native-cpu/qwen_cpu
```

This command expects no existing stable path. On the development machine the
verified stable link already points to `build-10/qwen_cpu`; preserve immutable
build directories when changing releases.

## Validation and development

Correctness precedes timing. Reuse the same checkpoint/tokenization, exact prompt
IDs, natural EOS, declared precision, and immutable build artifacts:

- `tools/capture_native_cpu_logits.py` records two repeat-checked full-logit
  trajectories. `tools/check_native_logits.py` compares them to the independent
  NumPy oracle without adapting the declared limits to the result.
- `tools/benchmark_native_cpu.py --mode quality` reuses the existing nine-case
  suite against tuned llama.cpp, retaining wrong answers and exact EOS IDs.
- Its `discovery` and `confirmation` modes use the same fixed workload and
  preflight exact-ID gate. Confirmation requires at least ten alternating paired
  rounds; `tools/analyze_native_phase0.py` supplies paired uncertainty intervals.
- `tests/browser_native_cpu.py` prepares actual browser checks on an isolated
  localhost server at port 8766, including cancellation, switching backends,
  unsupported controls, and the polled benchmark guard. It does not launch a
  server or build assets automatically.

Reports distinguish numerical agreement, token identity, answer accuracy, and
performance. Passing one does not establish the others. Phase 1 met its
within-10% decode target before the four-hour deadline. This result permits
further specialization; it does not establish a decode win or a general result
for other models, machines, context lengths, or precision classes.
