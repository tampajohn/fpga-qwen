# Current deployed backend versus native CPU inference

Measured on 2026-11-11 with the deployed scalar-group/B8 executable (`186615739651cb253804374653bb8d74e816e8870873ea88a17c2b9b4c6bb682`) at repository commit `71c34c1`. Eight workers/threads, identical prompt token IDs, 32 greedy output tokens, empty logical KV per request, one initial request and two warm repetitions per profile. No GPU or PyTorch inference. Raw results and executable/generated-source hashes are in [the report](../reports/scalar-b8-matched-current.json).

| Engine | Prompt | Warm median decode tokens/s | Warm median first-token seconds |
|---|---|---:|---:|
| rtl | short (85 tokens) | 49.92 | 0.762 |
| rtl | long (452 tokens) | 32.88 | 6.096 |
| llama-f16 | short (85 tokens) | 120.32 | 0.209 |
| llama-f16 | long (452 tokens) | 121.66 | 0.615 |
| llama-f32 | short (85 tokens) | 82.25 | 0.150 |
| llama-f32 | long (452 tokens) | 79.71 | 0.469 |

All initial and warm requests produced identical token IDs within each profile across the three engines. This establishes agreement on these two prompts, not general model-quality parity. The existing separate quality corpus remains necessary for arithmetic changes.

The current backend remains behind both native references. Long-prompt first-token latency is particularly far behind: 6.096 seconds versus 0.615 for FP16 and 0.469 for FP32. Projection/prefill improvements therefore remain important alongside decode optimization.

The CPU window excluded other agents' builds and inference jobs. Existing unrelated servers were resident and observed idle. Engines ran sequentially, not in alternating pairs; there are only two warm samples. Host drift and differing transport/precision limit exact speed-ratio interpretation. SV uses BF16 stored weights with FP64 arithmetic/KV; native uses named stored weights, mixed arithmetic and FP16 KV. These results refresh the current gap; they do not measure a new optimization or supersede paired ablations.

Reproduce using `tools/benchmark_matched.py --engines rtl,llama-f16,llama-f32 --threads 8 --tokens 32 --repetitions 2 --profiles short,long --sv-binary CURRENT_DEPLOYED_BINARY --sv-lanes 8 --sv-block-size 4 --sv-prefill-size 8 --output build/scalar-b8-matched-current.json` through the locked project environment.
