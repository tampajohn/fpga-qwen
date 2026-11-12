# Fixed KV arrays with the QK2 attention kernel

This experiment changes only the existing `--static-kv` build option on the current QK2 hybrid profile. Model arithmetic, weights, KV dimensions and runtime source hashes are identical. The reference `c5b414d226bafba3dc9419961e9adc626cf02ab7280aeefba159385410ffedf0` uses deque-backed dynamic SV arrays; candidate `63535681aed2abab6549cb4eecba93372a2e074a13cff3653740d39b5ee376e1` uses fixed SV arrays. Each model's combined KV payload remains 1,792 MiB. This is a payload size, not a measured process RSS.

Historical static-KV results in [projection_optimization.md](projection_optimization.md) were single sequential warm runs with substantial host drift. They neither establish nor rule out a benefit for the current parallel/QK2 kernels. The new comparison must use interleaved pairs and isolate the storage flag.

## Exactness and compiler evidence

[Thirty-eight cross-artifact full-logit comparisons](../reports/static-kv-qk2-gates.json) pass bitwise, covering counts 1–8, prefixes through 452 tokens, truncation/overwrite and sampled verification with 1/3/17 workers. The [33-case canonical cache suite](../reports/static-kv-canonical-gates.json) and [matching FP32 decode-state exclusion](../reports/static-kv-decode-exclusion.json) pass against fresh same-route inference. The new fixed-array artifact has not yet run an actual 4096-token capacity test.

Independent assembly review confirms the QK2 inner loop at `0x100024278–0x100024298` uses direct indexed key loads and retains independent `fmla.2d` accumulation. The loop has nine instructions instead of the dynamic-array loop's roughly nineteen, removing deque metadata/block-map lookup. The value loop at `0x1000244ac–0x1000244e0` similarly uses direct loads with the existing bounds checks and scalar `fmadd`; it has fourteen instructions versus roughly twenty-five. There is no new value-loop SIMD. Instruction-count reductions are not performance results.

`tools/compare_static_kv.py` verifies identical source hashes and that `static_kv` is the sole differing build option. It uses the same hybrid precision profile, B8 prefill segmentation, eight workers, fixed greedy output IDs, logical KV reset and alternating execution order. Direct acknowledged/closed dumps preserve warmed pools. Startup is recorded separately from inference; no process RSS measurement has been taken.

## Paired timing

[Three alternating pairs](../reports/static-kv-qk2-comparison.json) per workload used 85/452-token prompts and 32 greedy output tokens. Warm/final full-logit bits and every generated ID match. Ratios below are baseline time divided by fixed-KV time; values above one favor fixed KV.

| Workload | Prefill median | Decode median (range) | Total median |
| --- | ---: | ---: | ---: |
| 85-token prompt | 1.052× | 0.904× (0.902–1.068×) | 0.986× |
| 452-token prompt | 1.091× | 1.264× (1.086–1.294×) | 1.103× |

All three long-context prefill and decode pairs improved, but two short-context decode pairs regressed. This run shows a workload-dependent tradeoff, not a universal improvement. The candidate remains experimental; do not switch the default solely on the favorable long-context row. Host drift, allocation placement and generated code layout remain possible contributors. These results do not establish behavior at other context lengths.

Startup diagnostics appear separately in the raw report and were not scored as inference. Both resident models and warmed OS caches were retained; no memory eviction or process RSS claim is made. No runtime source, UI, default or live-service change is included.

### Short-run variability

The three baseline short decode intervals were 0.4989/0.5258/0.5127 seconds; fixed KV took 0.4670/0.5832/0.5672 seconds in AB/BA/AB order. Candidate decode therefore worsened after its first pair despite nearly constant candidate prefill times (0.6597/0.6628/0.6625 seconds). This does not isolate a structural regression or simple ordering bias. A later bounded confirmation should repeat the same 85-token/32-output workload with additional alternating pairs; lengthening decode would also change context length and answer a different question. No confirmation has been run in this report.
