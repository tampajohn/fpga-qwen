The optional eight-lane attention dot reduction was measured on top of the deployed exact scalar-group profile, using main `1fcb965` and its deployed binary beginning `18661573` as the reference. This comparison isolates the incremental attention option; the earlier comparison against ungrouped decode is not reused as evidence of a combined gain.

The complete report is `reports/scalar-attention-paired.json`, with the isolated candidate's build manifest in `reports/scalar-attention-manifest.json`. Both engines used eight workers, B8 prefill, B4 verification capacity and the same 85/452-token prompts. Three alternating A/B, B/A, A/B pairs emitted 32 identical greedy tokens per request, with logical KV reset each time and both processes resident. The warm-request full-vocabulary checks passed the explicit 1e-6 tolerance: maximum errors were 4.62e-14 short and 4.80e-14 long. These are reassociated FP64 results, not bit-identical logits.

| Workload | Decode ratios, candidate/reference speed | Median TTFT speed ratio | Total request speed ratios |
|---|---|---:|---|
| Short | 1.096, 0.977, 1.032 | 0.966 | 1.001, 0.971, 1.041 |
| Long | 1.007, 1.078, 1.054 | 1.047 | 1.049, 1.051, 1.048 |

Ratios above one favor the candidate. The combined option yields a consistent roughly 5% total-request improvement on this long workload, while short total time is effectively neutral at the median and short TTFT is mixed/slower. Recommendation: leave attention reassociation off as the general default and retain it as an optional candidate for longer contexts. These few warm requests are not a universal latency guarantee or broad quality validation. No code or production default changes are included in this evidence commit.

To reproduce at `1fcb965`, build the existing selected profile with one extra flag:

```sh
python tools/build_projection_variant.py --lto --unchecked --masked-activation --unsigned-indices --fast-attention --parallel-attention --attention-dot-lanes8 --batched-prefill --block-unroll --compact-projection --group-projections --prefill-b8 --scalar-group
python tools/compare_attention_dot.py --reference CURRENT_SCALAR_BINARY --candidate PRINTED_COMBINED_BINARY --pairs 3 --tokens 32 --output build/scalar-attention-paired.json
```

Run during an exclusive CPU window. The recorded comparison completed after explicit orchestration handoff; no compiler or other inference jobs overlapped the scored window.
