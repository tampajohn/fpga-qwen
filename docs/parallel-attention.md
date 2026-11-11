# Optional parallel attention

Build an isolated candidate with:

```sh
python3 tools/build_projection_variant.py --lto --unchecked --masked-activation --unsigned-indices --fast-attention --compact-projection --parallel-attention
```

This option leaves the default profile unchanged. The existing projection pool
also executes attention head partitions. Main SV retains Q/K normalization,
RoPE and KV writes; worker SV computes dot products, softmax and weighted values.
There is no second pool or host neural math. C++ reads the main model's const KV
storage through scalar transport callbacks; it does not duplicate the cache.

For N workers, worker i owns heads `[16*i/N,16*(i+1)/N)`. Output copying scales
these bounds by128; empty partitions still acknowledge completion. Each worker
has its own32KiB scores array. Existing job publication/completion barriers keep
KV stable while workers read. A lifetime guard joins the pool before destroying
the main model on normal exit and exceptions. Current process-kill cancellation
remains unchanged.

Generated C++ inspection confirmed KV writes precede synchronous dispatch.
ThinLTO eliminated the scalar KV callback symbols and their call boundaries in
the inspected executable. Attention preserves each head's ascending-dimension
Q·K and ascending-time weighted-value reductions; projection lane selection is
independent.

The initial combined unsigned+attention binary SHA256 is
`fe7bb377e3059069ada818f8385003f1015c28bdbf9166bd3cd8828d655a185a`.
Against unsigned baseline `becad933...`, the452-token prompt plus4 forced decode
positions passed exact full-logit bit-pattern comparison (maximum error0).
A preliminary single warm repeat with16 generated tokens measured:

| Context | Unsigned baseline tok/s | Parallel attention tok/s | Baseline TTFT | Candidate TTFT |
| --- | ---: | ---: | ---: | ---: |
| 85 prompt tokens | 38.24 | 41.92 | 1.77s | 1.61s |
| 452 prompt tokens | 12.32 | 32.55 | 18.12s | 11.52s |

These are preliminary local CPU observations, not formal repeated estimates or
FPGA timings. Raw reports are `build/attention-combined-gate.json` and
`build/attention-baseline-quick.json`. Broader quality gates and repeated paired
measurements are still required before promotion.

`tests.test_parallel_attention` takes `SV_ATTENTION_BINARY`, optionally
`SV_ATTENTION_REFERENCE`, and `SV_TEST_LANES`. Supply a separate reference at the
same projection/worker settings when testing attention in isolation. The
unsigned projection path can differ from workers0 scalar by tiny FP64 rounding
amounts, so workers0 is not an isolation control for that build. The original
pre-unsigned candidate did pass bitwise workers0/1/3/8/17 comparisons.

The synthetic scheduler fixture checks CV/hybrid scheduling with the attention
option enabled/disabled, head copy ranges1/3/8/17, and alternating projection and
attention operations. It verifies transport scheduling, not neural arithmetic.
The full-model fixture checks full logits, same-uniform sampling, truncation,
reset and process restart. `tests.test_kv_layout` supplies longer suffix-replacement
coverage when run with the candidate/reference environment variables.
