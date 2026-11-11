# Opt-in B8 prefill with B4 verification

Eight-token prefill yields a measured first-token speed ratio of 1.18–1.20× against the current grouped B4 engine on this CPU (about 15–16% less waiting). This is an isolated candidate; the default remains B4 until independent full-vocabulary, quality and cache gates pass. It does not beat llama.cpp based on this comparison.

The new `prefill count tokens...` command accepts one through eight valid token IDs and returns `prefill count final_position`. Count, capacity and every token are validated before evaluating SV or mutating KV. It never computes vocabulary logits or samples. Existing `block` requests remain bounded to four positions, preserving sampled suffix verification; ordinary token decode is unchanged. `RTLModel(..., block_size=4, prefill_size=8)` explicitly declares the separate capability. No filename guessing or HTTP configuration enables it.

`PREFILL_B8` expands layer-first SV state and packed buffers to stride eight. For counts above four, the worker reuses each weight across eight positions using four FP64 reduction lanes. This changes floating-point association. Counts at most four are packed into the original stride-four, eight-lane worker and unpacked back; inactive wide inputs are zeroed. Group descriptors retain the same bounded row ranges and immutable publication lifetime; counters use the actual four/eight-position worker-copy width. All model arithmetic stays in SV. Workers-zero and reduction-lanes-one retain scalar fallback semantics.

## Paired measurements

Three alternating-order pairs per profile, eight CPU workers, identical prompts, 32 greedy output tokens, one unscored warmup per model/profile and fresh logical KV per request. Every warm/timed output sequence matched and none emitted EOS. Both processes were resident; host caches were warm, no OS cache eviction was attempted, and protocol/copies are included. Startup is reported separately. Other agents paused compilation/inference during scoring; unrelated user processes were not stopped.

| Prompt tokens | Median prefill speed ratio | Median TTFT speed ratio | Median total request speed ratio |
|---|---:|---:|---:|
| 85 | 1.189× | 1.184× | 1.128× |
| 452 | 1.198× | 1.197× | 1.159× |

Every pair improved all three measures. Short candidate TTFT ranged 0.799–0.900 seconds versus baseline 0.946–1.034; long candidate TTFT ranged 5.809–6.101 seconds versus baseline 6.953–7.385. These are local warmed measurements, not FPGA hardware predictions or guarantees for other hosts/contexts. The decode kernel did not change; decode timing variation is not attributed to a decode optimization.

[Raw paired evidence](../reports/b8-prefill-paired.json) contains every timing, prompt/output token ID, checkpoint hash and both executable hashes. Candidate SHA256 is `17b12348ced96818e18578af7fadaf88e40333ae48b29dcd4eeff520039ff184`; baseline is `64288263dd5973bf7bf6c2c60c87fa4b48104b0d66fe4a800877fbed64b88f43`. Both include explicit logit-dump close before ACK. The [microkernel evidence](wide_projection_four_lanes.md) motivated this candidate but is not substituted for these model measurements.

## Validation and pending promotion gates

Initial eight-token full-vocabulary comparison had maximum absolute error 4.80e-14 and the same greedy ID; B4 sampled output triples matched exactly. Host tests cover distinct capacities, malformed inputs/responses, sampled suffix RNG/output/cache parity, and verification staying at four while prefill uses eight. Descriptor tests cover the opt-in eight-token ceiling, wrong widths, output gaps, oversized rows and invalid spans; existing scheduler tests still pass. All 15 host/descriptor/scheduler tests pass; a separate non-B8 grouped build compiles successfully. The [candidate build manifest](../reports/b8-prefill-build-manifest.json) records exact sources and flags.

Independent validation is in progress: 85/452-context full-vocabulary tolerance, held-out NLL/KL/greedy agreement, causal prefix truncation/replacement, reset/cancel/malformed wire, and actual 4096-token capacity. This document does not claim those unfinished gates passed. Arithmetic reassociation prevents a blanket bit-exact claim even when text matches.

## Reproduction

Build with `uv run --locked python tools/build_projection_variant.py --lto --unchecked --masked-activation --unsigned-indices --fast-attention --parallel-attention --batched-prefill --block-unroll --compact-projection --group-projections --prefill-b8 --tag flushed`. The manifest records source hashes, flags, compiler and binary hash. Omitting `--prefill-b8` retains the old physical layout and wire commands.

Run `uv run --locked python tools/benchmark_prefill_b8.py --reference BASELINE --candidate CANDIDATE --pairs 3 --output build/b8-prefill-paired.json`. The harness rejects output divergence and EOS. Run host/descriptor tests with `uv run --locked python -m unittest tests.test_prefill_transport tests.test_prefill_descriptors tests.test_suffix_chat tests.test_projection_scheduler`.
