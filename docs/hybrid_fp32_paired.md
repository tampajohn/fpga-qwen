The validated hybrid runtime was compared directly against the current deployed exact scalar-group/B8 profile in an exclusive CPU window. This paired comparison complements the native-backend measurements; it does not reuse earlier timings recorded under different host conditions.

Reference SHA256 begins `18661573`; hybrid SHA256 begins `d27a2ffc`. Both use eight workers and B8 FP64 prefill. The reference uses FP64 scalar projections with verification capacity4; the candidate uses FP32 scalar projections with verification capacity1. The candidate's precision capability JSON is verified before timing. Every request resets KV; prefix reuse and suffix verification are absent. Three pairs alternate A/B, B/A, A/B for each85/452-token prompt, emitting the same32 greedy IDs in every warm/timed request.

| Metric | Short context | Long context |
|---|---:|---:|
| Median paired decode speed ratio | 1.209 | 1.140 |
| Decode ratio range | 1.138–1.226 | 1.111–1.151 |
| Reference median decode tokens/s | 46.75 | 34.24 |
| Candidate median decode tokens/s | 56.51 | 39.04 |
| Median paired total-request speed ratio | 1.086 | 1.013 |
| Total-request ratio range | 1.051–1.093 | 0.997–1.063 |

All six pairs favored the candidate for decode. Long total-request benefit was small/mixed because unchanged prefill dominates these32-output-token requests; one long total-request pair regressed0.26%. Prefill/TTFT medians were approximately neutral. These results do not imply superiority to native backends or universal throughput.

Full-vocabulary logit differences are descriptive cross-precision measurements, not an arbitrary acceptance threshold: maxima1.45e-5 short and1.40e-5 long, same final greedy IDs. Separate held-out/conversation/canonical replay gates establish the bounded quality evidence. `tools/compare_hybrid_fp32.py` deliberately does not reuse the attention-only1e-6 gate or its FP64-only metadata.

The complete results, runtime capability response, model/checkpoint/tokenizer hashes, raw prompt/output IDs, full timings and helper source hashes are in `reports/hybrid-fp32-paired.json`. Reproduce with:

```sh
python tools/compare_hybrid_fp32.py --reference CURRENT_FP64_BINARY --candidate HYBRID_BINARY --pairs 3 --tokens 32 --output build/hybrid-fp32-paired.json
```
