# Second performance iteration

The existing fast simulator scales modestly beyond eight workers: an exploratory sweep reached 22.91 tokens/s with sixteen workers versus 21.35 with eight. This is not yet a new matched comparison against llama.cpp. The stronger next opportunities are unnecessary projection-buffer copies and serial attention memory access.

## Existing fast baseline

The binary is `build/variants/scalar-O3-lto-unchecked/full/Vqwen_behavioral`, SHA256 `59b1a6939c1b60aad08a55f742418909f8c4c5738d3d889fb4aa0e5f02ca8b29`. It uses BF16 weights, FP64 arithmetic/KV, eight reduction lanes, and persistent projection workers. No GPU or PyTorch is involved.

The workload is the existing matched short prompt: 85 prompt tokens and 32 greedy generated tokens. Each request resets logical KV; the first request warms the process, followed by two measured requests per worker count. The sampling requests are excluded from timings. All measured outputs have identical token IDs across worker counts and no EOS.

| Workers | Median warm TTFT | Median steady decode |
|---:|---:|---:|
| 4 | 3.439 s | 16.35 tokens/s |
| 8 | 2.957 s | 21.35 tokens/s |
| 12 | 2.881 s | 22.08 tokens/s |
| 16 | 2.884 s | 22.91 tokens/s |

Raw results: [worker-sweep.json](../reports/worker-sweep.json). The order was 8, 4, 12, 16, with only two warm repetitions. This is an exploratory sweep, not an interleaved controlled experiment. A concurrent user VM consumed approximately 82% CPU at one spotcheck, with `mds` around 19% and WindowServer around 15%. Other agents were editing or building, not running inference. User processes were left running. Host load and run order can affect the small differences between 8, 12, and 16 workers; eight remains the numeric thread/worker count for matched comparisons.

## Sampling findings

A separate ten-second macOS `sample` run used eight workers on repeated copies of the same workload. Main-thread counts, out of 7,908 samples:

| Main-thread stack category | Samples | Approximate share |
|---|---:|---:|
| Projection dispatch subtree | 5,382 | 68.1% |
| Condition waiting within that subtree | 5,082 | 64.3% |
| Memory copying | 747 | 9.4% |
| Buffer zeroing | 222 | 2.8% |

The condition-wait row is a subset of dispatch, not an additional category. Waiting includes useful projection work executed by worker threads; it does not establish that 64% of time is removable synchronization overhead. Whole-process sample totals also include idle Verilator threads and must not be interpreted as wall-time percentages. Sampling overhead and mixed prefill/decode phases limit the precision of these estimates. Full macOS stacks and host addresses are intentionally not committed.

Generated C++ shows a concrete buffer cost: every projection clears a 151,936-element FP64 result buffer and copies it into the main model, including hidden projections with only 1,024–3,072 output rows. There are 196 hidden projections and one vocabulary projection per decoded token. Clearing these oversized buffers alone writes roughly 239 MB per decoded token, before copies. Splitting hidden and vocabulary output buffers should remove most of this unnecessary traffic while preserving arithmetic.

The attention value loop visits time steps inside the dimension loop, accessing KV values with an 8 KB stride. Reversing those loops permits contiguous dimension access while preserving each output component's time-order accumulation. Normalizing each score once rather than dividing it for every dimension removes repeated division. Numerical and greedy-output gates are still required because generated code and compiler contraction can affect rounding.

There are 197 projection dispatches per decoded token, so wakeup scheduling may matter after the arithmetic improvements. A hybrid spin/sleep experiment should be measured separately: spinning can waste CPU during serial attention and worsen competing host load. The profile does not by itself prove this change will help.

## Reproduction

```sh
uv run --locked python tools/profile_simulator.py \
  --binary build/variants/scalar-O3-lto-unchecked/full/Vqwen_behavioral \
  --workers 8,4,12,16 --lanes 8 --tokens 32 --repetitions 2 \
  --sample-seconds 10 --output build/profile-workers.json \
  --host-load-note 'Record competing host work observed during this run'
```

The helper reuses the matched benchmark prompt and EOS policy. It records executable and prompt hashes, individual timings, greedy IDs, and worker-count output equality. Profiling results must not replace the existing three-repeat matched benchmark.

## Candidate validation and matched results

The selected candidate improves the short-context matched median from **21.85 to 27.71 tokens/s**, about **27% faster**. At the longer context, decode improves only from **10.48 to 10.99 tokens/s**; the difference is small relative to observed timing variation. llama.cpp FP16 remains substantially faster at **141.44 tokens/s** on the same short workload, approximately **5.1×** the candidate rate. These are CPU software simulation measurements, not FPGA performance estimates.

The candidate combines masked activation indexing, contiguous attention value accumulation with scores normalized once, and separate hidden/vocabulary projection buffers. It retains BF16 weight storage and FP64 arithmetic/KV. Its binary is `build/variants/scalar-O3-lto-unchecked-masked-activation-fast-attention-compact-projection/full/Vqwen_behavioral`, SHA256 `59139c1ff40c0df262387cd3379ab518a1262fa68b3a893f9dc15291185b062d`. Hybrid worker waiting and aggressive contraction were exploratory alternatives, not part of this measured candidate.

| Engine | Prompt tokens | Warm TTFT median | Decode median | Warm decode range |
|---|---:|---:|---:|---:|
| Previous fast SV | 85 | 2.971 s | 21.85 tokens/s | 21.59–21.86 |
| Selected candidate SV | 85 | 2.411 s | 27.71 tokens/s | 25.84–30.38 |
| Previous fast SV | 452 | 26.198 s | 10.48 tokens/s | 10.15–10.55 |
| Selected candidate SV | 452 | 20.450 s | 10.99 tokens/s | 10.29–14.00 |
| llama.cpp FP16 | 85 | 0.235 s | 141.44 tokens/s | 118.45–144.04 |

Each row has three warm requests after one first request, with 32 generated tokens, greedy decoding, eight workers/threads, and empty logical request KV. Decode excludes prompt work and the first token. Each backend reproduces its own 32-token sequence across all repetitions; old and new SV also produce identical sequences for both contexts. No timed request reaches EOS. Native FP16 uses FP16 KV and mixed native arithmetic, so this is a workload comparison with explicitly different precision. Process startup was 0.142 s for previous SV, 0.175 s for the candidate, and 0.424 s for native FP16; this is not OS-cache-cold loading, and startup is shared across a backend's context profiles.

Measured requests ran sequentially in an exclusive agent inference window, but the host was not idle. A gate-time spotcheck found the user's VM around 56% CPU, Warp around 12%, and WindowServer around 9%; these processes were not stopped. Candidates were not interleaved with the baseline, and the ranges above argue against treating the long-context difference as a robust speedup. Short warm TTFT fell about 19%, and long warm TTFT about 22% in this run.

The numerical gate compared all 151,936 logits at each of 16 teacher-forced generated positions following the **452-token** prompt. The reference used sequential lane-1 FP64 reduction; the candidate used eight reduction lanes. All logits were finite, greedy choices were unchanged, and maximum absolute error was **1.68754e-13**, below the predeclared 1e-6 tolerance. This is a finite workload gate, not proof of equivalence for every context.

All nine fixed conversation cases produced exactly the same text as the previous SV candidate and the earlier native FP16/FP32 reports. The strict score remains **4/9**. Formatting violations, sorting, arithmetic output, and choosing the larger number remain weaknesses; preserving those answers demonstrates no new measured regression, not high model quality.

Raw evidence:

- [Previous fast baseline](../reports/iteration2-baseline.json)
- [Selected candidate, long numerical gate, and nine quality cases](../reports/iteration2-candidate.json)
- [Same-session native FP16 comparator](../reports/iteration2-native-f16.json)

The report files preserve raw tokens, timing repetitions, prompt hashes, executable hashes, and generated-source provenance, with local paths sanitized. The earlier five-position candidate probe around 42.5 tokens/s uses a different workload and must not substitute for the matched 27.71 tokens/s result.

Reproduce the selected candidate gate and measurements:

```sh
uv run --locked python tools/benchmark_matched.py \
  --engines rtl --threads 8 --tokens 32 --repetitions 3 --profiles short,long \
  --sv-binary build/variants/scalar-O3-lto-unchecked-masked-activation-fast-attention-compact-projection/full/Vqwen_behavioral \
  --sv-lanes 8 \
  --reference-sv-binary build/variants/scalar-O3-lto-unchecked/full/Vqwen_behavioral \
  --gate-profile long --gate-tokens 16 --quality \
  --output build/iteration2-candidate.json
```

For the previous fast baseline, use its path as `--sv-binary`, retain `--sv-lanes 8`, and omit the reference gate and quality flags. For the native comparison use `--engines llama-f16 --profiles short` with the same thread, token, and repetition settings.
