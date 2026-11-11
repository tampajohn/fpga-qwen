# Third performance iteration: paired unsigned-index comparison

Using unsigned loop indices in the existing fast simulator improved short-context decode in all three interleaved pairs, with a **16.8% median paired gain**. Longer-context decode improved in all three pairs too, but gains varied from **3.7% to 15.9%**, with a **5.8% median**. These results support a modest optimization, not a claim that the simulator beats llama.cpp.

The selected candidate adds unsigned loop indices to the masked-activation, improved-attention, compact-projection-buffer variant. It retains dynamic KV arrays, BF16 weight storage, FP64 arithmetic/KV, and eight reduction lanes. Static KV arrays and the combined static/unsigned variant did not establish a consistent advantage in exploratory probes and were not selected.

## Paired method and results

Both model processes remain resident, but inference runs sequentially. Each profile warms both models once, then measures three pairs in **baseline/candidate, candidate/baseline, baseline/candidate** order. Each request resets logical KV and uses the same prompt IDs, 32 greedy generated tokens, and eight projection workers. No timed request reaches EOS, and every warmup and measured request produces identical IDs across both binaries. Host applications remain running. This alternating design reduces run-order effects but does not eliminate host-load variation or constitute a statistical confidence interval.

| Context | Baseline median decode | Candidate median decode | Median paired ratio | Paired ratio range |
|---|---:|---:|---:|---:|
| 85 prompt tokens | 31.03 tokens/s | 36.29 tokens/s | 1.168× | 1.146–1.179× |
| 452 prompt tokens | 13.79 tokens/s | 14.60 tokens/s | 1.058× | 1.037–1.159× |

Median warm TTFT was 2.052 → 1.750 seconds for short context and 19.247 → 18.052 seconds for long context. Paired TTFT speed ratios were 1.112, 1.176, 1.184 for short context and 0.982, 1.066, 1.208 for long context; the first long-context pair slightly favored the baseline.

The median paired ratio is calculated from individual pair ratios, not by dividing medians. Both models being resident differs from earlier single-process timing runs, so absolute rates should not be combined with earlier tables to infer an additional speedup. The prior native FP16 result remains a historical comparator; no native backend was rerun in this paired experiment. No combined-memory estimate is inferred from two resident models, because shared pages and process buffers complicate that accounting.

Raw paired requests, hashes, settings, and outputs are in [iteration3-paired.json](../reports/iteration3-paired.json). The short-context gain is consistent within this small sample. Long-context evidence supports a smaller and more variable gain.

## Provenance

Baseline: `build/variants/scalar-O3-lto-unchecked-masked-activation-fast-attention-compact-projection/full/Vqwen_behavioral`, SHA256 `59139c1ff40c0df262387cd3379ab518a1262fa68b3a893f9dc15291185b062d`.

Candidate: `build/variants/scalar-O3-lto-unchecked-masked-activation-unsigned-indices-fast-attention-compact-projection/full/Vqwen_behavioral`, SHA256 `becad933e07b45cc027717ae8f04b842c2db46d32b5a5b3e0bc736a6e85cb07e`.

## Correctness gates

The 452-token prompt followed by 16 teacher-forced positions passed an exact-bit comparison against the previous fast binary with both using eight reduction lanes. All 151,936 logits at each position were finite and had identical FP64 bit patterns; maximum absolute error was **0**, and greedy choices were unchanged.

All nine existing quality cases produced exactly the previous candidate's responses, with the strict score still **4/9**. This establishes no new measured regression; it does not establish high quality or repair the model's existing failures. The grading rubric was unchanged. Raw gate and quality evidence: [iteration3-quality.json](../reports/iteration3-quality.json).

The new opt-in `tests/test_kv_layout.py` passed in 19.435 seconds using independent candidate/reference binaries. It evaluates 133 original positions, truncates to 63, overwrites 80 suffix tokens through position 143, then resets and freshly replays the resulting 143-token sequence. Full-vocabulary checks span positions 63/64 and 127/128, and cached versus fresh logits are bit-identical within each binary. The comparison exercises every layer/head through complete model evaluation and checks that stale future KV is excluded after truncation and reset.

This fixture is intentionally not a 4,096-token capacity test. Existing position guards and exact layout dimensions remain necessary. The largest legal KV element index is 117,440,511, which fits signed 32-bit indexing; unsigned loop indices therefore do not justify removing existing capacity and token guards.

## Reproduction

```sh
uv run --locked python tools/compare_variants.py \
  --baseline build/variants/scalar-O3-lto-unchecked-masked-activation-fast-attention-compact-projection/full/Vqwen_behavioral \
  --candidate build/variants/scalar-O3-lto-unchecked-masked-activation-unsigned-indices-fast-attention-compact-projection/full/Vqwen_behavioral \
  --workers 8 --profiles short,long --pairs 3 --tokens 32 \
  --output build/iteration3-paired.json
```

`tools/benchmark_matched.py` additionally accepts `--reference-sv-lanes 8 --require-bit-identical` for layout/index changes that should preserve arithmetic exactly. Its default reference reduction remains lane 1, so earlier reassociation checks retain their original semantics. Bit identity compares the actual FP64 bit patterns, including signed zero, and finite-logit/greedy/tolerance checks remain mandatory.
