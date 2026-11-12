# Optional paired-head attention experiment

`--attention-head-pairs` selects an isolated candidate for scalar attention after applying the archived reproduction patch; the active runtime does not contain this experiment. It requires parallel attention and rejects the reassociated `--attention-dot-lanes8` kernel. Existing defaults remain unchanged. The artifact passed bounded exactness gates, but matched timings did not establish a consistent gain; it remains optional and is not recommended for promotion.

Operation 6 partitions eight KV-head pairs rather than sixteen individual query heads. Each pair owns two contiguous 128-element outputs. It stages the two query vectors as 128 adjacent pairs in SV, shares each key load, and retains each head's original ascending-feature accumulation. Softmax and value accumulation retain ascending-time order. Per-worker score scratch is persistent 4096×2 FP64 storage; query packing is 128×2.

Only worker counts 1/2/4/8 select operation 6. Other counts use the unchanged operation 1 and its sixteen-head partition, avoiding split ownership or reduced parallelism for odd worker counts and 17 workers. Both routes validate layer/position/shape before publication; attention does not read projection weights. Operation 6 copies 256 outputs per pair. Batched prefill/verification attention continues to use operation 5, so it does not select this scalar kernel.

Required evidence: actual linked hot-loop SIMD with independent head lanes and no horizontal reduction; bitwise full-vocabulary logits and seeded outputs against the original scalar attention; worker eligibility/fallback and repartitioning; canonical cache/truncation; near-capacity boundaries. Matched short/long decode timing must precede any performance claim. The attention fraction sets a limit on total gains even if QK itself becomes much faster.

## Initial evidence

Candidate SHA256 `5242c2664aac95ed4214a1aba90805284e8a8e36722b15fa26dc5f19977cc5eb` passed [49 full-vocabulary bitwise comparisons](../reports/attention-head-pair-gates.json) against the original batched-attention hybrid `07e22976`. Cases cover sampled scalar decode after prefixes 0/1/7/8/63/452, eligible worker counts 1/2/4/8, fallback counts 3/7/17, truncation/overwrite and restart. The transport mock passes both CV and hybrid wait paths across 1/2/3/4/5/7/8/9/16/17/32 workers, including output canaries, invalid shapes/positions, and failure joins.

The linked `Vprojection_worker::evalNba` selects operation 6 at `0x1000250f0`. Its inner dot loop `0x1000252b4–0x1000252f4` contains a shared scalar key load at `0x1000252dc`, packed query load at `0x1000252e0`, and `fmla.2d` at `0x1000252e4`. The accumulator stays in `v0` for 128 features; there are no calls, horizontal reductions or accumulator spills in that loop. Query packing runs once per head pair before the history loop. Numerical parity is tested separately; SIMD alone is not a speed result.

The new kernel has not been exercised at the actual 4096-token capacity boundary. Independent source, assembly and comparator review found no blocker, but mixed timing results do not justify promotion or the expensive boundary run. No UI/default change is included.

## Matched timing decision

[Three alternating pairs per workload](../reports/attention-head-pair-comparison.json) used identical 85/452-token prompts, 32 greedy outputs, eight workers, warmed pools and direct closed-file dumps. All fixed output IDs and warm/final logits matched bitwise. Models stayed resident and logical KV was reset before each request; host drift remains possible.

| Workload | Median decode ratio | Decode ratio range | Median total ratio |
| --- | ---: | ---: | ---: |
| 85-token prompt | 1.075× | 0.980–1.152× | 1.058× |
| 452-token prompt | 0.997× | 0.894–1.086× | 1.005× |

Ratios are baseline time divided by candidate time, so values above one favor the candidate. The short workload has a modest but inconsistent benefit; the longer workload has no established decode gain. Observed SIMD does not translate into a consistent end-to-end improvement here. Archive this candidate as optional evidence; do not combine it with QK2 or multiply historical speed ratios. This rejects this measured layout, not every possible GQA optimization.

## Archived reproduction

Implementation commit `d9120db` is preserved on the isolated `codex/attention-head-pairs` branch, based on `e508193`. The original archive commit did not change active SV, scheduler, transport, builder, UI or defaults. The later combined experiment restores its opt-in runtime implementation as a prerequisite for paired value accumulation; the original QK-only timing result and reproduction remain unchanged. Defaults are still unchanged.

To reproduce, create a separate worktree at `e508193`, apply [attention-head-pairs.patch](../tools/experiments/attention-head-pairs.patch), and copy `tools/check_attention_head_pairs.py`, `tools/compare_attention_head_pairs.py`, and the archived `tools/experiments/test_attention_head_pairs_scheduler.py` into that worktree (placing the scheduler fixture under `tests/`). Build with the reviewed [shortreal compiler](shortreal_compiler_milestone.md), substituting its local path below:

```sh
python3 tools/build_projection_variant.py --storage scalar --opt O3 \
  --lto --unchecked --masked-activation --unsigned-indices \
  --fast-attention --parallel-attention --batched-prefill --block-unroll \
  --compact-projection --group-projections --prefill-b8 --scalar-group \
  --fp32-projection-b1 --fp32-verification-b4 --batch-attention \
  --shortreal-compiler /path/to/verilator-shortreal --attention-head-pairs
python3 -m unittest tests.test_attention_head_pairs_scheduler -v
```

Build the reference with the same options except `--attention-head-pairs`. Run the real checker before the paired comparator, passing the two resulting binaries through their `--reference` and `--candidate` options. Independent review validated the original artifact; a rebuild must repeat exactness gates before reusing its speed measurements.

Patch SHA256: `22f3cf01fe711e459d8f7e4c5529c28132d51a938eeb5cf2303c1ded064fc6d1`. Reverse application was checked against the preserved candidate runtime, confirming its correspondence to the stated base diff. Applying this patch to newer main revisions may require conflict resolution; use the stated base for reproduction.
