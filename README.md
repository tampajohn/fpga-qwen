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

**Build20 is the current localhost browser release**, listed first and selected
by default at `http://127.0.0.1:8765`. The
[live browser check](reports/native-cpu-build-20-browser-live.json),
[promotion record](reports/native-cpu-build-20-promotion.json) and
[release manifest](reports/native-cpu-build-20-manifest.json) bind the released
artifact. It uses **12 decode / 6 prefill workers**, no GPU and no PyTorch.
Build19 remains available for rollback.

Build20 adds adjacent query-row pairing to ordinary batched prefill: two token
rows and their GQA heads share FP16 key/value loads and conversions while each
output retains its original reduction order. Odd final rows use the existing
paired-head path; ordinary single-token decode and B2 verification remain
unchanged. Extra attention scratch is **1.125 MiB** at the selected worker counts
and 4096 context. Full causal history, the 448 MiB FP16 KV allocation, FP16 weights,
FP32 stored activations and FP64 normalization-square accumulation are retained.

Canonical **512-token prefix reuse** remains enabled; partial tails are
recomputed and decoded/verified KV is not canonical reusable prefill. One-token
n-gram suffix drafting remains opt-in, Greedy-only and **off by default**.

### Quality and numerical scope

The [relative acceptance record](reports/native-cpu-prefill-query-row-pair-relative-acceptance.json)
binds 76 exact full-vocabulary arrays (the original 68 plus 8 on the actual new
63/64 fixtures), API/prefix/suffix lifecycle checks, and final candidate sources.
[Complete seeded streams](reports/native-cpu-prefill-query-row-pair-seeded-stream.json)
match build19. The [nine-case](reports/native-cpu-prefill-query-row-pair-quality.json)
and [18-case](reports/native-cpu-prefill-query-row-pair-extended-quality.json)
suites retain matching trajectories and strict scores **4/9 and 10/18**. Wrong
answers remain wrong; passing nonregression checks does not imply all answers
are correct.

**Known independent numerical failures remain unresolved.** The
[63-token baseline control](reports/native-cpu-build19-newboundary63-control-oracle.json)
fails max absolute 0.01 and RMS 0.002 limits; the
[64-token control](reports/native-cpu-build19-original-prefix-control-boundary64-oracle.json)
fails RMS. Build20 preserves these arrays exactly rather than fixing or
reclassifying them. Prior 28-position oracle evidence remains fixture-scoped.
The documented [release policy](docs/native_cpu_four_hour_pass.md#explicit-release-policy-revision-baseline-nonregression)
distinguishes arithmetic-preserving baseline nonregression from absolute
accuracy remediation. Limits are unchanged; arithmetic-changing finer prefix
chunks remain held.

### Fresh performance comparisons

Against build19 at the same 12/6 settings, the fresh comparison supports TTFT
improvements at 1K (median speedup 1.0610, 95% interval [1.0227,1.1828]) and 2K
(1.1028,[1.0533,1.1608]). Near-4K TTFT is uncertain, and **all complete-request
and decode intervals include 1**. No overall latency win, decode win or
noninferiority is established against build19. Standard 85/452 contexts use 20
alternating pairs; extended 1K/2K/3968 contexts use 10. See the
[complete final-vs19 table](docs/native_query_row_final.md).

The fresh direct comparison against tuned llama FP16 uses the same requested
12 decode / 6 prefill workers, ten alternating pairs per context, one unscored
warmup per visit, 32 outputs for standard contexts and 128 for extended contexts.
All samples/outliers are retained. Ratios above 1 favor native; parentheses are
95% paired bootstrap intervals:

| Prompt tokens | Decode throughput ratio | TTFT speedup | Complete-request speedup |
|---|---:|---:|---:|
| 85 | 0.8561 (0.4897–1.0001) | 1.6957 (1.4536–2.1194) | 1.0277 (0.6928–1.3344) |
| 452 | 0.9493 (0.7861–1.1613) | 1.1972 (1.0467–1.4381) | 1.0313 (0.9896–1.3142) |
| 1,024 | 1.0068 (0.7869–1.2380) | 1.2260 (1.1363–1.4069) | 1.1901 (0.9761–1.2700) |
| 2,048 | 1.1906 (1.0876–1.3003) | 1.1411 (1.0969–1.2506) | 1.1397 (1.0682–1.2756) |
| 3,968 | 1.2426 (1.0852–1.4002) | 1.0433 (1.0052–1.1635) | 1.0770 (1.0335–1.1347) |

TTFT favors native in all five sampled contexts; decode and complete requests
favor native at 2K/3968 and remain uncertain at 85/452/1K. This is **not a universal
win or a noninferiority guarantee**. Native subprocess and llama HTTP transport,
shared-host activity and unadjusted multiple intervals limit interpretation.
The [full direct comparison and reports](docs/native_query_row_final.md#fresh-matched-comparison-with-tuned-llama-fp16)
preserve the measured scope. No historical or separate kernel/worker ratios are
multiplied or pooled.

Build19's deployment and its comparisons against build18/llama remain
[historical evidence](docs/native_kv_layout.md). Rejected experiments and the
remaining numerical work are documented in the
[four-hour pass](docs/native_cpu_four_hour_pass.md).

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
benefit; those measurements are historical, not current build20 results. Build-10's
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
is selected automatically. The local deployment uses **build-20**, with 12
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
stable link currently points to `build-20/qwen_cpu`; preserve immutable
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
