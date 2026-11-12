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

**Build-08 is the current evaluated candidate.** It combines improved worker
scheduling, grouped/four-row projection dispatch, larger prefill chunks, and
CPU vector-library nonlinear functions. It passed the unchanged numerical gates
at contexts 1–4, 85–92, and 452–455, and matched tuned llama.cpp on the nine-case
exact-trajectory suite (both still 4/9). Additional synthetic boundary predictions
at contexts 513/514 passed for both engines against NumPy; this is chunk-carry
coverage, not an answer-quality benchmark.

Ten alternating paired rounds used 85/452-token uncached prompts and 32 identical
greedy output tokens:

| Engine, 12 decode / 6 prefill threads | 85-token prompt decode | 452-token prompt decode |
| --- | ---: | ---: |
| Tuned llama.cpp FP16, KleidiAI + LTO | 141.35 tokens/s | 130.55 tokens/s |
| Original CPU engine, build-08 | 148.35 tokens/s | 136.93 tokens/s |

Median **paired** native/llama decode ratios were **1.014x** (95% paired-bootstrap
interval 0.812–1.114) and **1.055x** (0.949–1.196). Both intervals cross 1:
point estimates are competitive, but **no decode win is established**. Paired
ratios are not quotients of the table's independent medians. Native first-token
medians were 0.100/0.603 seconds versus llama's 0.160/0.559 seconds: shorter
context improved, while longer-context TTFT remained behind.

See [confirmation](reports/native-cpu-build-08-confirmation.json),
[paired uncertainty](reports/native-cpu-build-08-confirmation-analysis.json),
[conversation quality](reports/native-cpu-build-08-quality.json), and
[513-token boundary gate](reports/native-cpu-boundary-513-oracle-gate.json).
Earlier build-02 and build-03 evidence remains in the
[current plan](docs/native_cpu_plan.md); comparisons across historical runs do
not establish isolated optimization gains. Native subprocess transport differs
from llama.cpp's HTTP streaming, native prefill retains widened FP32 layer
weights, and background scheduling remains uncontrolled.

An isolated actual-browser check was completed for build-06; it is not a
build-08 browser validation. The live backend is unchanged and no candidate has
been promoted. Candidate-09's lossless BF16 auxiliary-cache source has no result
reported here. The 511/512-token boundary work is still pending publication.

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

Install the locked Python dependencies and start the existing service:

```sh
make setup
.venv/bin/python tools/chat_server.py --model /path/to/Qwen3-0.6B --port 8765
```

Open **http://127.0.0.1:8765**. The current shared launcher still requires an
already-built historical simulator asset, even when selecting a native backend;
it fails explicitly if that asset is missing. Starting this command does not
build the original engine or export its weights. This is a transitional launcher,
not a standalone native-only installation flow.

The **Native CPU · FP16 · experimental** option is available only when the
explicitly prepared assets below are installed. It is not automatically selected
or promoted by this work. Its worker counts are 12 decode / 6 prefill; simulator
worker controls do not apply. Native prefix reuse and suffix drafting are
unsupported: every message processes the full conversation again. Stop terminates
the active native process, and the next request starts with fresh KV state.

Existing simulator and llama.cpp choices remain for comparison; their caching
and suffix capabilities differ and are displayed per backend. Do not infer a
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
path. Neither the build command nor this README installs or promotes a build.

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
performance. Passing one does not establish the others. The current Phase 1
implementation has a four-hour budget and a stop rule: if a correct plain engine
cannot reach 90% of tuned FP16 decode by the deadline, report the gap rather than
claim success or continue an open-ended optimization sweep.
