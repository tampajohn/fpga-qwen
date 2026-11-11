# Independent paired validation of parallel attention

Parallelizing attention heads on the existing persistent workers improved short-context decode by a **19.2% median paired ratio** and long-context decode by **2.32×**. The long-context improvement addresses the serial attention bottleneck identified in earlier profiling. These are software simulation results, not FPGA latency estimates, and do not establish a win over llama.cpp.

## Workload and results

The baseline is the previous unsigned-index fast model. The candidate adds parallel attention, retaining BF16 stored weights, FP64 arithmetic/KV, eight projection reduction lanes, and eight workers. Both model processes stay resident while requests run sequentially. After one warmup each, three pairs alternate baseline/candidate, candidate/baseline, baseline/candidate. Every request starts with empty logical KV, uses identical prompt IDs, and generates 32 greedy tokens without EOS. Every warmup and timed request produces identical token IDs between engines.

| Prompt tokens | Baseline median decode | Candidate median decode | Median paired gain | Paired gain range |
|---:|---:|---:|---:|---:|
| 85 | 34.52 tokens/s | 40.84 tokens/s | 1.192× | 1.144–1.208× |
| 452 | 14.33 tokens/s | 33.18 tokens/s | 2.316× | 2.136–2.956× |

Short-context warm TTFT medians were 1.788 → 1.665 seconds. Long-context warm TTFT medians were 18.976 → 10.405 seconds. Individual long-context TTFT speed ratios were 1.905×, 1.689×, and 1.636×.

The baseline's long decode rate varied from 11.24 to 15.38 tokens/s; the candidate ranged from 32.85 to 33.22. The large gain persists across all three pairs, but baseline variation explains the broad ratio range. Median paired gains are medians of individual pair ratios, not ratios of aggregate medians.

No other agent inference or compilation ran during scored timings. User applications remained running, and the machine was not otherwise isolated. Both processes being resident differs from earlier single-engine tests, so rates should not be combined across rounds to derive extra speedups. No native backend was rerun here; earlier llama.cpp results are historical comparators only. Row-tiling compilation began after scored timings ended, during numerical/quality validation.

Raw timings, requests, settings, and hashes: [attention-paired.json](../reports/attention-paired.json).

## Correctness evidence

An independent long-context gate compared every one of 151,936 logits at each of 16 teacher-forced positions following a 452-token prompt. Both baseline and candidate used eight reduction lanes. All logits were finite and **bit-identical**, maximum absolute error was **0**, and greedy choices were unchanged.

All nine fixed quality cases produced exactly the preceding baseline's responses. The strict score remains **4/9**; no rubric changed. This shows no new measured quality regression, not that existing sorting, arithmetic, formatting, and reasoning weaknesses have been fixed. Raw evidence: [attention-quality.json](../reports/attention-quality.json).

Before this independent run, the implementation owner separately reported passing tests for worker counts 1, 3, 8, and 17, including full-logit/sample agreement, truncation, reset, and restart. The owner also ran the longer KV fixture: 133 original positions, truncation to 63, replacement of 80 tokens, and fresh replay through position 143. Those owner-run tests were not rerun during this timing window; the independent gate above uses a separate baseline executable.

Read-only review checked disjoint head/output ranges, current-token KV publication before dispatch, private worker score buffers, and joining workers before model destruction. It found no blocker in valid jobs. These checks do not prove behavior at every context or arbitrary malformed internal DPI call.

## Provenance and reproduction

Baseline binary SHA256: `becad933e07b45cc027717ae8f04b842c2db46d32b5a5b3e0bc736a6e85cb07e`.

Candidate binary SHA256: `fe7bb377e3059069ada818f8385003f1015c28bdbf9166bd3cd8828d655a185a`, built in the isolated attention worktree at source tip `83154ba`. Binary paths and generated-source hashes are retained in the sanitized reports. `$PROJECT`, `$ATTENTION_WORKTREE`, and `$CHECKPOINT` are report placeholders, not executable shell configuration.

From the primary repository, with the attention worktree adjacent:

```sh
uv run --locked python tools/compare_variants.py \
  --baseline build/variants/scalar-O3-lto-unchecked-masked-activation-unsigned-indices-fast-attention-compact-projection/full/Vqwen_behavioral \
  --candidate ../fpga-qwen-attention/build/variants/scalar-O3-lto-unchecked-masked-activation-unsigned-indices-fast-attention-parallel-attention-compact-projection/full/Vqwen_behavioral \
  --workers 8 --profiles short,long --pairs 3 --tokens 32 \
  --output build/attention-paired.json
```

For the independent numerical/quality check, use `tools/benchmark_matched.py` with these binaries as `--reference-sv-binary` and `--sv-binary`, plus `--engines rtl --threads 8 --reference-sv-lanes 8 --sv-lanes 8 --require-bit-identical --gate-profile long --gate-tokens 16 --quality-only`.

## Main-worktree integration

The attention branch was merged into main in `0f4a89f`. The default browser build
now enables unsigned indices and parallel attention; static KV and row tiling
remain excluded. `make fast-test` includes the head-partition regression against
a separate unsigned-only reference executable.

Rebuilding identical source hashes in the main worktree produced binary SHA256
`d6828bf12adbbe1338be325074ca2f92efb291683e07a39d3b8c6a863e089106`.
The scored worktree artifact above retains its own hash; its absolute rates are
not presented as a new measurement of the rebuilt artifact. All seven integration
tests passed, including attention partitions, sampling, truncation/reset/restart,
projection bounds, scheduling and conversational recovery. Browser smoke passed
across all four backends, prefix recall, worker changes, cancellation, context
errors and mobile layout, with no JavaScript errors. The idle localhost server
was restarted and its actual simulator path/hash verified; the one-word `blue`
smoke passed. Runtime logs and browser screenshots remain ignored build artifacts.
