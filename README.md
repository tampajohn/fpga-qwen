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

**Build-12 is the current local browser release**, with 12 decode / 6 prefill
threads. It adds exact-safe prefix reuse of completed canonical 512-token prefill
blocks and shares K/V conversion between paired query heads during batched
prefill. Partial tails are recomputed; decoded KV is never reused as a canonical
prefill block. Suffix drafting remains unavailable.

Newer experiments remain separate from that live release. Candidate 15 adds
opt-in greedy two-token verification and has passed exact-logit, conversation,
EOS and rollback checks. Its first paired suffix benchmark established no
request-speed gain, so drafting stays off by default; see the
[suffix correctness and timing evidence](docs/native_suffix_b2.md).
The [standalone vocabulary-routing prototype](docs/native_head_probe.md)
also passed its scoped correctness checks but was slower than the dense head.
Neither experiment establishes a new speed win over llama.cpp.

Its [56-array comparison](reports/native-cpu-build-12-exact-gate.json) matches
prior canonical full logits exactly. The [nine-case quality suite](reports/native-cpu-build-12-quality.json)
retains exact tuned-llama token trajectories and stopping behavior, with both
engines scoring **4/9**. The [28-case prefix check](reports/native-cpu-build-12-prefix-check.json)
covers greedy/seeded chat, edited and shorter prompts, session changes and
cancellation. Baseline agreement is not improved answer accuracy. Earlier
independent NumPy gates use fixed maximum absolute error ≤0.01 and RMS ≤0.002,
including [511](reports/native-cpu-boundary-511-oracle-gate.json),
[512](reports/native-cpu-boundary-512-oracle-gate.json), and
[513-token chunk boundaries](reports/native-cpu-boundary-513-oracle-gate.json).

In [ten paired cache-on/off rounds](reports/native-cpu-build-12-prefix-confirmation.json),
primed prompts with an edited tail showed the following TTFT results. Priming is
outside the measured request; these are not fresh-prompt speedups.

| Prompt tokens | Median TTFT fresh → reused | Median paired speedup (95% interval) |
| --- | ---: | ---: |
| 1,024 | 1.335 → 0.803 s | 1.632× (1.575–1.730) |
| 2,048 | 3.728 → 1.397 s | 2.672× (2.557–2.819) |
| 3,968 | 11.719 → 1.974 s | 5.744× (5.512–6.512) |

All decode-speed intervals in that cache comparison include 1: no decode gain
is established from prefix reuse.

The latest [ten-pair fresh-request confirmation](reports/native-cpu-build-12-final-standard-confirmation.json)
retains build-12 against tuned KleidiAI-ON/LTO-ON llama FP16, both with 12 decode /
6 prefill threads, identical 85/452-token prompts and 32 output tokens. Every
visit has one unscored warmup per prompt; configurations alternate order.
[Paired bootstrap intervals](reports/native-cpu-build-12-final-standard-analysis.json)
retain all ten rounds. Ratios above 1 favor the original engine.

| Prompt tokens | Decode throughput ratio (95% interval) | TTFT speedup (95% interval) | Request-time speedup (95% interval) |
| --- | ---: | ---: | ---: |
| 85 | 0.974× (0.939–1.042) | 1.380× (1.313–1.715) | 1.109× (1.063–1.253) |
| 452 | 0.953× (0.917–0.978) | 1.079× (1.010–1.181) | 1.040× (1.001–1.098) |

These workloads show lower TTFT and request time, **not a decode win**. The
452-token decode interval indicates a slowdown; both decode intervals remain
within the declared 10% allowance at their lower bounds. The 452-token request
interval barely excludes 1, and these are separate, unadjusted workload/metric
intervals. They do not establish broad superiority.

**Near-4K numerical validation failed for the selected build-12.** The fixed
NumPy contract remains maximum absolute error ≤0.01 and RMS ≤0.002. The
[3968/3969-position control](reports/native-cpu-build-12-context3968-oracle-control.json)
reported maximum errors 0.0148945 / 0.0106325 and RMS errors 0.00166184 /
0.00204414. Both positions fail; the second exceeds both limits. Greedy IDs
still matched, which does not satisfy the full-logit contract. The
[1024/1025](reports/native-cpu-build-12-context1024-oracle-control.json) and
[2048/2049](reports/native-cpu-build-12-context2048-oracle-control.json)
controls passed, within those selected positions' scope. The completed
[128-output extended timing run](reports/native-cpu-build-12-final-extended-confirmation.json)
is retained, but near-4K timing is diagnostic evidence and cannot support a
qualified speed win. The numerical gap remains unresolved; the follow-on goal
is not complete, and tolerances have not been relaxed.

For the two shorter extended workloads, the [ten-pair 128-output analysis](reports/native-cpu-build-12-final-extended-analysis.json)
reports decode ratios 0.991× (0.913–1.039) at 1024 prompt tokens and 1.077×
(0.947–1.175) at 2048. Neither establishes a decode improvement; their TTFT
and whole-request intervals also include 1. Independent oracle checks cover
two positions per prompt, not all 128 outputs. These scoped results must not
be pooled with the numerically failed near-4K workload.

The earlier [unfavorable ten-pair comparison](reports/native-cpu-build-12-standard-confirmation.json)
is retained separately: paired decode ratios were 0.687× and 0.772×, with
[intervals](reports/native-cpu-build-12-standard-confirmation-analysis.json)
0.487–0.914 and 0.658–0.923. Reported concurrent machine activity remains a
possible confound, not an established explanation. The newer run does not erase
those measurements or establish why results differed.

Build-10's earlier [30-pair confirmation](reports/native-cpu-build-10-confirmation.json)
met the Phase 1 within-10% decode target, with paired native/llama ratios 1.028×
(0.954–1.079) and 0.980× (0.905–1.022). This is historical evidence for build-10,
not build-12 performance. The immutable build-10 artifact remains available.
Earlier and negative results are retained in the [current plan](docs/native_cpu_plan.md).

These are single-stream measurements on an Apple M5 Max. Native subprocess
transport differs from llama.cpp HTTP streaming; native prefill retains a
1.76 GB widened FP32 layer cache; background activity is uncontrolled. Numerical
and conversation checks cover their declared fixtures, not every context or
all model behavior. Build-12 is exposed locally at the user's request; its
[latest live browser checks](reports/native-cpu-build-12-final-browser-live.json) passed: default/reload,
dropdown interaction across status polls, cached-prefix replies to changed
questions, cancellation/new chat, backend switching and the benchmark lock.

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
is selected automatically. The local deployment uses **build-12**, with 12
decode / 6 prefill threads; simulator worker controls do not apply. The
`--native-only` launcher needs no simulator binary or simulator imports. Omit
that flag to include any installed legacy simulator choices. Starting the
server does not build the engine, export weights, or download a model.

Native prefix reuse is enabled for completed canonical 512-token prefill blocks
in the same conversation; partial tails are recomputed. Prefixes shorter than a
complete block are processed fresh. Suffix drafting is unavailable. Stop
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
stable link currently points to `build-12/qwen_cpu`; preserve immutable
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
