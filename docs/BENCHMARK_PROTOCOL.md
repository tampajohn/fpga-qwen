# CPU comparison protocol and results

## Selected B8 profile

The fast profile now uses B8 prefill with B4 verification. Paired model runs
reduced first-token wait by about 15–16% versus grouped B4, with unchanged timed
output IDs; see [B8 evidence](b8_prefill.md). This is a prefill gain, not a claim
of faster steady decode or parity with llama.cpp.

## Matched grouped B4 comparison before B8 promotion

The grouped B4 simulator trailed llama.cpp in this CPU-only comparison
uses the actual B4 prefill path, eight workers/threads, identical 85/452-token
prompts, empty logical KV, and 32 greedy outputs with EOS respected. Each row
has one first request followed by two warm repetitions; engines ran sequentially
in the order SV, FP16, FP32. These are current measurements, not an alternating
paired speedup experiment or a confidence interval.

| Engine | Prompt tokens | Median warm TTFT (s) | Median decode (tok/s) |
| --- | ---: | ---: | ---: |
| SV grouped B4 | 85 | 1.015 | 41.50 |
| llama.cpp FP16 | 85 | 0.214 | 139.01 |
| llama.cpp FP32 | 85 | 0.131 | 81.22 |
| SV grouped B4 | 452 | 7.201 | 29.75 |
| llama.cpp FP16 | 452 | 0.637 | 123.10 |
| llama.cpp FP32 | 452 | 0.504 | 77.45 |

All three engines generated identical token IDs on both timed prompts, and each
engine repeated its own sequence. The separate 452-token SV gate compared eight
subsequent full-vocabulary logit positions against the prior fast executable:
bit-identical, maximum error zero. This verifies the exact prefill optimization,
not broad model quality. The prior nine-case suite's 4/9 strict score remains a
limitation; it was not rerun here. Suffix drafting was disabled.

The SV path still uses BF16 weight storage and FP64 arithmetic/KV; native engines
use their named weight formats, mixed arithmetic and FP16 KV. Same worker count
does not imply identical memory formats, kernels, or CPU utilization. Background
user applications and idle localhost servers remained present; other agent heavy
work was held during scoring. Raw timings, IDs, hashes and limitations:
[current report](../reports/grouped-matched-current.json).

`tools/benchmark_matched.py --sv-block-size 4 --sv-lanes 8 --sv-binary PATH`
explicitly selects block prefill for a compatible executable. The default remains
single-token prefill for reference executable compatibility. Before this option,
the comparison tool always used single-token prefill even with a B4-capable
binary; historical timings below describe their recorded execution path.

## Historical scalar optimization comparison

The optimized SystemVerilog candidate is **3.20× faster than its original
simulator baseline** on the matched workload, but still does **not** beat
llama.cpp. All rows below use the same 85 prompt token IDs, 32 greedy decoding
steps, eight workers/threads, empty logical KV per request, and **three warm
repetitions**. Both engines respect EOS; every timed run completed 32 steps
without encountering EOS. No stop-token masking is used in these final runs.

| Engine | Startup (s) | Median warm TTFT (s) | Median decode (tok/s) |
| --- | ---: | ---: | ---: |
| Original SV bulk-row baseline | 0.122 | 9.792 | 6.34 |
| SV scalar O3 + thin LTO + checked job bounds + 8 lanes | 0.126 | 3.221 | 20.32 |
| llama.cpp FP32 weights | 0.278 | 0.155 | 78.61 |
| llama.cpp FP16 weights | 0.274 | 0.237 | 138.77 |
| llama.cpp Q4_K_M weights | 0.294 | 0.072 | 246.94 |

Startup values are process/model startup with an uncontrolled OS page cache.
The raw artifact is authoritative for exact values and every repetition. This
is one short-context workload, not proof of scaling at longer contexts or a
statistical survey of all CPU conditions. No other agent ran heavy jobs during
these scheduled benchmark windows. Warm decode ranges were 6.33–6.47 tok/s
(original SV), 20.12–20.78 (candidate), 62.69–82.61 (native FP32),
138.01–140.03 (FP16), and 239.22–249.58 (Q4). The FP32 row varied noticeably;
three repetitions do not characterize all scheduling or thermal variability.

The candidate retains BF16 weight storage and FP64 arithmetic/KV; it is **not an
FP16 arithmetic implementation**. Its numerical gate compared every vocabulary
logit at 16 teacher-forced positions after the same 85-token prompt. Maximum
absolute error was **1.43885e-13** (declared limit 1e-6), with all logits finite
and an unchanged greedy sequence. Every timed engine also reproduced its own
greedy output sequence across the first request and three warm repetitions.

On the separately declared nine-case objective suite, candidate SV, native FP32,
and native FP16 produced **identical text on all nine cases**, each passing
**4/9** strict gates. This establishes no new failures in this small comparison;
it does not establish high general answer quality. The failed answers remain
visible in the raw artifacts.

Final evidence: `reports/matched-candidate.json` and
`reports/matched-baseline-repeat.json`. Original SV binary SHA-256:
`a915e91014f0332a69bcc4c2d8f5bf5270e17aa1f0b50b41d6f0edb95c4f0b27`.
Candidate SHA-256:
`59b1a6939c1b60aad08a55f742418909f8c4c5738d3d889fb4aa0e5f02ca8b29`.
The original executable is retained locally as
`build/baselines/Vqwen_behavioral-a915e91014f0` for future comparisons.


## Initial pilot run (superseded for speedup claims)

A preliminary matched run on the
Apple M5 Max measured the following. All engines received the same **85 prompt
token IDs**, generated **32 greedy tokens**, used eight compute workers/threads,
and started each measured request with an empty logical KV cache.

| Engine | Startup (s) | Warm-process TTFT (s) | Steady decode (tok/s) |
| --- | ---: | ---: | ---: |
| SV, exact bulk-row transport | 0.123 | 10.001 | 6.22 |
| llama.cpp FP32 weights | 0.330 | 0.142 | 84.95 |
| llama.cpp FP16 weights | 0.271 | 0.234 | 124.83 |
| llama.cpp Q4_K_M weights | 0.385 | 0.072 | 239.99 |

This is **one warm repetition**, preceded by a first empty-cache request. It is
an initial comparison, not a stable performance claim. Startup is process/model
startup; the OS file cache was not evicted. TTFT excludes process startup but
includes prompt evaluation and the first output. Steady decode is the interval
between first and last generated token, divided into the remaining token count.
Both measurements include their existing transport: simulator pipes and native
HTTP/SSE. The initial fixed-count run used native EOS suppression and SV continuation,
but no EOS was selected in its recorded SV sequence. This was an implementation
mismatch identified during review; current runs instead let both engines respect
EOS and reject the comparison if either finishes before the requested count.
Fixed-count throughput is not a meaningful generated-answer quality score.

The run is stored in `build/matched-quick.json`, including raw token IDs, timings,
prompt fingerprint, and executable/generated-source hashes. The SV executable
SHA-256 was `a915e91014f0332a69bcc4c2d8f5bf5270e17aa1f0b50b41d6f0edb95c4f0b27`.
This is the exact bulk-row transport prototype, not the later optional lane
reduction experiment. Do not compare its 6.22 figure directly with the earlier
6.87 figure: that benchmark used only nine token positions at a shorter context.

## What is and is not matched

Matched:

- Local Qwen3-0.6B checkpoint lineage and exact prompt token IDs.
- Greedy selection with repetition, presence, and frequency penalties disabled.
- Generated token count, logical starting context, and numerical worker count.
- CPU-only execution, no Torch, GPU, remote inference, or speculative decoding.
- Fresh logical KV state per timing request, with resident-process repetitions.

Not matched, and therefore recorded explicitly:

- SV uses BF16 stored weights, FP64 `real` arithmetic and FP64 KV payloads.
- Native FP32/F16/Q4 models use the named storage formats, native mixed-precision
  kernels, and FP16 KV. The F16 format retains some one-dimensional tensors at
  FP32. A weight format does not specify every intermediate arithmetic format.
- SV has eight projection workers plus a coordinator; llama.cpp uses eight
  inference threads. Their scheduling and prompt batching differ.
- Native prompt evaluation is batched; current SV prompt evaluation is serial
  across token positions. That is a real implementation disadvantage, not hidden
  by reporting decode rate alone.

llama.cpp supports both FP32 and FP16 **weights**. Local evidence is
`build/llama.cpp/include/llama.h` (`LLAMA_FTYPE_ALL_F32` and
`LLAMA_FTYPE_MOSTLY_F16`) and the successfully run native F16 model. The earlier
rejected FP32 setting was the benchmark's KV-cache type, not model weight support.

The SV checkpoint already stores each weight in 16 bits (BF16). Switching that
storage to IEEE FP16 would not halve its byte count. Reducing FP64 computation or
KV precision is a separate optimization, which must be evaluated for numerical
and behavioral regressions. Verilator promoting `shortreal` to `real` does not
constitute a lower-precision implementation.

## Reproduction

Create native FP16 weights without installing or invoking Torch:

```sh
build/llama-cpu/bin/llama-quantize \
  build/bench/qwen3-0.6b-f32.gguf \
  build/bench/qwen3-0.6b-f16.gguf F16 8
```

Run the quick comparison during an exclusive CPU benchmark window:

```sh
uv run --locked tools/benchmark_matched.py \
  --profiles short --tokens 32 --repetitions 1 \
  --output build/matched-quick.json
```

Use at least three repetitions and several contexts for promotion decisions:

```sh
uv run --locked tools/benchmark_matched.py \
  --profiles short,medium,long --tokens 32 --repetitions 3 \
  --output build/matched-full.json
```

A performance prompt must naturally produce at least the requested number of
tokens. If early EOS rejects a run, shorten `--tokens` or choose a longer-answer
prompt; do not silently suppress EOS on only one engine.

Do not run model tests, chat generation, conversion, compilation, or other agents'
benchmarks concurrently with measured runs. Run candidate and baseline through
this same harness and retain both artifacts. Record code/executable hashes;
changing an executable underneath a measurement invalidates its provenance.

## Quality gates

`tools/conversation_eval.py` defines nine objective cases before inference:
history recall, corrections, quoting prior assistant text, task switching,
structured JSON, arithmetic, selective retrieval, evidence-only unknowns, and
recovery after repetitive history. These are small regression gates, not a
comprehensive model-quality assessment. Grading is deliberately strict about
requested output formats and declared token budgets.

```sh
uv run --locked tools/benchmark_matched.py \
  --engines llama-f32,llama-f16,llama-q4_k_m --quality-only \
  --output build/matched-native-quality.json
uv run --locked tools/benchmark_matched.py \
  --engines rtl --quality-only --quality-cases fact_recall,task_switch \
  --output build/matched-sv-quality.json
```

Initial greedy results:

- Native FP32 and FP16 produced identical text on all nine cases, each passing
  **4/9** strict gates. Q4 also passed 4/9, with some different output text.
- Passed: fact recall, assistant quotation, selective codename recall, and
  evidence-only unknown.
- Failed: punctuation in a city-only answer, echoing rather than sorting numbers,
  Markdown fences around JSON, an explanation violating a number-only arithmetic
  answer and exhausting the 16-token budget, and answering `7` to “which is larger,
  12 or 7?”. Do not reinterpret all format failures as factual reasoning failures.
- The two SV cases tested in this window matched native FP32/F16 exactly:
  `7319` (pass) and `9, 2, 5.` (fail). The full SV suite is not yet measured here.

A performance candidate should retain the existing independent full-logit tests,
show declared numerical tolerances when changing arithmetic, and show no new
failures on the fixed behavioral suite. Greedy tests isolate math changes; seeded
sampled chat evaluations are a separate behavior check. Passing these small gates
is necessary evidence, not proof of general quality preservation. In particular,
matching an already incorrect baseline answer does not make that answer correct.

## Candidate binaries and reassociated reductions

The harness accepts a trusted local executable path with `--sv-binary`; this is
CLI-only and is never exposed as an HTTP chat parameter. `--sv-lanes 8` enables
the candidate's eight-accumulator reduction through its simulator protocol.
`--sv-lanes 1` retains the ordered scalar reduction. Eight lanes may alter low
floating-point bits even when every operation remains FP64.

Before timing a candidate, compare every vocabulary logit over the same prompt
and teacher-forced generated sequence. The default gate is declared as maximum
absolute logit error **at most 1e-6**, all logits finite, and **identical greedy
choices** at every tested position. This tolerates reassociation without treating
arbitrary output drift as acceptable. The observed error is always reported.
This gate does not establish that a future FP16 implementation has sufficient
quality; a different arithmetic format needs a separately declared evaluation.

```sh
uv run --locked tools/benchmark_matched.py \
  --sv-binary build/variants/scalar-O3-lto/full/Vqwen_behavioral \
  --sv-lanes 8 \
  --reference-sv-binary build/full/Vqwen_behavioral \
  --gate-tokens 16 --logit-atol 1e-6 \
  --engines rtl,llama-f32,llama-f16 \
  --profiles short --tokens 32 --repetitions 3 \
  --output build/matched-candidate.json
```

The numerical gate is performed separately from timing and fails closed: a
candidate that fails it is not timed. Objective conversation gates still need
comparison; unchanged greedy IDs on one prompt do not establish general quality.

Sanitized initial raw artifacts are checked into `reports/matched-initial.json`,
`reports/quality-native-initial.json`, and `reports/quality-sv-initial.json`.
Only machine-specific directory prefixes were replaced with `$PROJECT` and
`$CHECKPOINT`; measurements, synthetic outputs, checks, and hashes are retained.
