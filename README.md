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

**Build-18 is the current local browser release**, with 12 decode / 6
prefill threads. Its [live browser check](reports/native-cpu-build-18-browser-live.json)
passed, including prefix reuse, opt-in drafting, precision disclosure,
cancellation, backend switching, and benchmark-lock visibility. It fixes the sampled
near-4K numerical failure by accumulating FP32 normalization square products in
FP64, then rounding the sum to FP32 before scaling. Matrix weights and KV remain
FP16; normalization weights and stored activations remain FP32. This is an
explicit arithmetic change, not an unchanged all-FP32 implementation.

The [56-array binding gate](reports/native-cpu-build-18-exact-gate.json) matches
the tested norm candidate exactly at 28 positions. Those independent numerical
checks cover tiny/short/452-token prompts, 511/512/513 boundaries, and
1024/2048/3968-token contexts, using unchanged maximum absolute error ≤0.01
and RMS ≤0.002. See the [numerical evidence](docs/native_cpu_plan.md).
These are selected-position checks, not full-context or full-generation proof.
The [nine-case](reports/native-cpu-build-18-quality.json) and
[18-case conversation gates](reports/native-cpu-build-18-extended-quality.json)
retain scoped tuned-llama agreement; they do not establish improved answer accuracy.

Build-18 retains canonical 512-token prefix reuse and adds optional one-token
n-gram drafting in Greedy mode. Drafting is **off by default**; prior suffix
measurements established no request-speed gain. Its
[28-case prefix check](reports/native-cpu-build-18-prefix-check.json),
[B2 API checks](reports/native-cpu-build-18-verify-api.json),
[suffix trajectory/lifecycle checks](reports/native-cpu-build-18-suffix-trajectories.json),
and [isolated browser check](reports/native-cpu-build-18-browser-isolated.json)
passed. Partial tails are recomputed, and decoded/verified KV is not reused as
a canonical prefill block.

Fresh direct comparisons against tuned KleidiAI-ON/LTO-ON llama FP16 use the
same requested 12 decode / 6 prefill threads and identical prompt/output IDs.
Each workload has ten alternating pairs and one unscored warmup per visit.
The [standard run](reports/native-cpu-build-18-final-standard-confirmation.json)
uses 32 output tokens; the [extended run](reports/native-cpu-build-18-final-extended-confirmation.json)
uses 128. Natural EOS remains enabled. Ratios above 1 favor this engine:

| Prompt tokens | Decode throughput ratio (95% interval) | TTFT speedup (95% interval) | Request-time speedup (95% interval) |
| --- | ---: | ---: | ---: |
| 85 | 0.944× (0.915–0.979) | 1.395× (1.333–1.540) | 1.102× (1.058–1.132) |
| 452 | 0.949× (0.852–0.997) | 1.080× (1.008–1.179) | 1.029× (0.963–1.091) |
| 1,024 | 1.000× (0.981–1.027) | 0.987× (0.955–1.113) | 0.993× (0.966–1.068) |
| 2,048 | 1.017× (0.900–1.047) | 0.846× (0.821–1.002) | 0.888× (0.859–1.010) |
| 3,968 | 1.119× (1.030–1.143) | 0.748× (0.698–0.810) | 0.805× (0.758–0.875) |

The [standard analysis](reports/native-cpu-build-18-final-standard-analysis.json)
shows decode about 5–6% slower at both short prompts, with both intervals below
1. First-token latency is about 28% and 7% lower. The
[extended analysis](reports/native-cpu-build-18-final-extended-analysis.json)
shows no established decode difference at 1K or 2K. Near 4K, decode is about
11.9% faster, but first-token latency is about 34% higher and the whole request
about 24% slower. **There is no overall performance win over llama.cpp.**
Percentages use median paired ratios; these separate intervals are unadjusted
and must not be pooled or chained through earlier comparisons.

Historical failures remain part of the evidence. Build-12's
[3968/3969 NumPy control](reports/native-cpu-build-12-context3968-oracle-control.json)
failed the same fixed limits despite matching greedy IDs: maximum errors were
0.0148945 / 0.0106325 and RMS errors 0.00166184 / 0.00204414. Build-18 passes
those sampled positions with changed norm accumulation; it does not erase that
failure or establish correctness at every position. Build-12's
[final standard](reports/native-cpu-build-12-final-standard-analysis.json),
[final extended](reports/native-cpu-build-12-final-extended-analysis.json), and
[earlier unfavorable comparison](reports/native-cpu-build-12-standard-confirmation-analysis.json)
remain archived. Its [cache-on/off comparison](reports/native-cpu-build-12-prefix-confirmation.json)
showed scoped TTFT benefits after untimed priming, without an established decode
benefit; those measurements are not new build-18 results. Build-10's
[30-pair confirmation](reports/native-cpu-build-10-confirmation.json) also remains
historical evidence. The [vocabulary-routing prototype](docs/native_head_probe.md)
was slower than the dense head, and [suffix timing](docs/native_suffix_b2.md)
did not establish a request-speed gain.

These are single-stream Apple M5 Max measurements. Native subprocess transport
differs from llama.cpp HTTP streaming; native reset/setup is outside the clock,
and a widened FP32 prefill weight cache adds roughly 1.76 GB. Requested worker
counts do not establish identical CPU placement or internal BLAS threading.
Background activity is uncontrolled, arithmetic differs, and numerical and
conversation checks cover their stated fixtures. The primary comparison is
FP16 weight/KV storage; FP32 and Q4 results are separate precision classes.
See the [current plan](docs/native_cpu_plan.md) and
[matched benchmark protocol](docs/BENCHMARK_PROTOCOL.md).

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
is selected automatically. The local deployment uses **build-18**, with 12
decode / 6 prefill threads; simulator worker controls do not apply. The
`--native-only` launcher needs no simulator binary or simulator imports. Omit
that flag to include any installed legacy simulator choices. Starting the
server does not build the engine, export weights, or download a model.

Native prefix reuse is enabled for completed canonical 512-token prefill blocks
in the same conversation; partial tails are recomputed. Prefixes shorter than a
complete block are processed fresh. Supporting releases offer one-token n-gram
drafting in Greedy mode, off by default; Chat mode disables it. Stop
terminates the active native process, and the next request starts with fresh KV state. Benchmark jobs block generation and
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
stored activations, and FP16 current/past KV storage. Normalization square
products are FP32, their sum accumulates in FP64, and the sum rounds to FP32
before the remaining scale/output operations. The embedding is tied to the output
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
stable link currently points to `build-18/qwen_cpu`; preserve immutable
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
performance. Passing one does not establish the others. Build-10 met the Phase 1
within-10% decode target before the four-hour deadline. That historical result permits
further specialization; it does not establish a decode win or a general result
for other models, machines, context lengths, or precision classes.
