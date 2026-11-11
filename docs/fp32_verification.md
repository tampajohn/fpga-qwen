# Experimental exact FP32 B4 verification

The opt-in `--fp32-verification-b4` builder flag adds a separate strict FP32 B4 hidden-projection archive to the hybrid model. It requires `--fp32-projection-b1`; existing defaults and Python/UI profile validation are unchanged. The experimental runtime reports `max_verification_block: 4` and `verification_projection: fp32`.

Command purpose explicitly selects operation 4. The coordinator enters a scoped verification route only after validating a `block` request; the scope clears on completion or exceptions. FP64 `prefill` requests retain operations 2/3, including every short tail. Scalar vocabulary projection retains FP32 B1. Neither canonical cache bookkeeping nor its eligible boundary changes.

The B4 module uses the existing FP32 SystemVerilog kernel, eight accumulation lanes and strict conversion/add/multiply rounding. Workers pack main stride-eight inputs into stride-four storage once per grouped dispatch, zero inactive lanes, and copy only active output positions. Matrix partitions and descriptors remain immutable until all workers finish. No model arithmetic moves into C++.

[The gate report](../reports/fp32-verification-gates.json) records the tested source and binary hashes. Twenty-four sampled cases cover verification sizes 1–4 after prefill tails 0/1/2/3/4/7. Their final and subsequent scalar-decode full logits match sequential FP32 B1 bitwise. A separate original-hybrid comparison preserves exact prefill behavior for lengths 1–8, 17, 85 and 452; greedy verification and rollback also match with 1/3/17 workers. Scheduler mocks cover both wait implementations, packing/canaries, group-copy accounting and invalid requests. The existing real block suite passes malformed-request, cancellation/restart, sampled and cache replacement checks; its expensive actual4096 capacity case was not rerun.

This is a transport/kernel candidate, not enabled suffix chat. Independent source review found no routing/publication blocker. Seeded end-to-end suffix parity, full model quality coverage, boundary capacity and matched acceptance/performance measurements remain required before promotion. No throughput improvement is claimed here.

Reproduction: build with the current hybrid flags plus `--fp32-verification-b4`; set `SV_FP32_VERIFY_BINARY` and `SV_FP32_REFERENCE_BINARY` to candidate and original hybrid paths, then run `uv run --locked python -m unittest tests.test_fp32_verification_real tests.test_fp32_verification_scheduler`. The existing block suite accepts `SV_BLOCK_BINARY` set to the candidate.
