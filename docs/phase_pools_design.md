# Proposed persistent pools by command purpose

Status: optional implementation and correctness gates complete; both timing comparisons and actual capacity validation are complete; no default or UI promotion is included. Worktree base `4abf9cd`; deployed combined artifact f17 remains immutable. The measured worker sweep motivates testing twelve prefill workers while keeping eight scalar/verification workers. A gain is a hypothesis, not a multiplied estimate from separate sweep results.

## Bounded first implementation

Add an opt-in `--phase-pools-experiment` C++ build flag. Keep one eight-worker decode/verification pool and one twelve-worker prefill pool alive for the entire process. Both use unchanged `ProjectionPool` scheduling, immutable checkpoint mapping, and the same single coordinator-owned main SV model/KV arrays. No second model or KV store is allocated. Only one pool executes at any instant: every `run`/`run_group` is synchronous before the coordinator can dispatch another job. Extra workers retain only existing worker-model/scratch allocations. Measure actual startup/RSS separately; do not claim that twenty resident threads run concurrently.

For a fair same-executable comparison, the experimental binary creates both pools once and reads a trusted startup-only environment selector `QWEN_EXPERIMENTAL_PHASE_POOLS=OFF|PREFILL_12` (unset means OFF). OFF routes every job through the eight-worker pool while the other pool sleeps. PREFILL_12 routes only explicit wire `prefill` commands to the twelve-worker pool. This controls code layout and resident allocation across the two timing modes. Normal builds do not inspect this variable or construct an extra pool. No arbitrary HTTP scheduler control is added.

Use coordinator-owned `CommandPurpose { scalar, prefill, verification }` and an exception-safe scope guard around a **fully validated** command's tick. Replace the separate verification boolean with an equivalent predicate on this enum, or retain it with an assertion that both match; prefer one enum to avoid contradictory state. Pool selection is based only on purpose, never count or `do_logits`:

| Wire operation | Purpose / pool | Existing neural operation retained |
|---|---|---|
| `prefill count tokens` for count1..8 | prefill /12 when enabled | Hidden op2 for tails1..4, op3 for5..8; attention op5 |
| `block count logits sample ...` for count1..4 | verification /8 | Hidden op4; vocab op0; attention op5 |
| `token` or `sample`, including logits0 | scalar /8 | Hidden/vocab op0; paired-head attention op6 |
| `reset`, `truncate`, `dump`, diagnostics | no inference purpose change | No pool reconstruction; scalar diagnostic dispatch uses8 |

All DPI dispatch sites obtain the selected pool, including scalar projection, scalar group, grouped block projection, ordinary attention and batched attention. Thus even an unexpected scalar fallback inside an explicit prefill remains assigned to its purpose consistently. Head-pair eligibility is checked on that selected pool. Existing validators, strides, FP32 operation selection and group descriptors are unchanged. Block verification must never drift onto the12-worker pool just because it contains multiple tokens.

Preserve `groupstats`' existing four-field aggregate contract by summing both pools' cumulative counters. Add read-only `phase_pools` JSON with selector, resident counts, selected prefill/decode counts, command-purpose dispatch counters and per-pool group counters. Capture initial process setup once; no warmup or request creates a pool. Metadata must distinguish12 prefill /8 decode from a blanket CPU-thread count and state no affinity guarantee.

For this bounded artifact require startup workers8 and lanes8. Matching `workers 8`/`lanes 8` commands are acknowledged without rebuilding either pool (the Python adapter already sends a lanes command on startup). Reject other values explicitly rather than silently mutating half the configuration. Normal builds retain existing reconfiguration behavior. Invalid selector is rejected before threads spawn. Lifetime management must join **both** pools before the attention root/main SV model or checkpoint mapping becomes invalid, including partial second-pool construction, failed commands and cancellation/process termination. Allocate pools under one ownership guard so exception unwinding cannot leave a reader alive.

## Why not an active subset initially

A single twelve-thread pool could use eight active workers for scalar work, saving four idle worker models. However, `ProjectionPool` currently partitions by the captured constructor count, increments readiness for all threads and uses full-pool completion counters/generation epochs. Active-subset execution would change partition denominators, participant counts and dormant-worker wakeup semantics in both CV and hybrid-wait modes. A stale generation or incorrectly decremented pending count could return before writes complete or deadlock. Two unchanged pools have a larger idle footprint but a much smaller synchronization change. Defer active subsets until this composition proves worthwhile.

## Targeted transport mock plan

1. Inject minimal stub pool objects recording constructor count, run operation, descriptor metadata and lifetime events. Replay explicit prefill1/2/3/4/5/6/7/8, verification1..4 with logits0/1 and scalar token/sample logits0/1. Assert12 only for prefill when enabled;8 everywhere else; OFF always8. Assert op2/3/4/0 and group byte counters remain correct.
2. Reject malformed counts/tokens/context/sample uniforms before command-purpose entry or dispatch. Test the scope restores scalar purpose after a tick exception. Distinguish prefill count1 from scalar and verification count4 from prefill count4.
3. Assert one construction of each pool across repeated commands/reset/truncate/unchanged configuration, no request-time recreation. Changed worker/lane requests fail without partially mutating state.
4. Inject second-pool startup failure and run failures from both pools. Assert all started workers joined before the shared model/root/weight lifetime ends. Exercise existing CV/hybrid scheduler fixtures unchanged to confirm each underlying pool still owns its original synchronization.
5. Assert aggregate `groupstats` equals the sum of per-pool counters and read-only telemetry reports exact counts/selector. Reject unknown environment values and prevent protocol/HTTP mutation.

## Real gates and measurement

First require byte-identical full logits/IDs to frozen f17 after short85/long452 contexts, all prefill/verification tails and sampled uniforms, plus existing canonical-prefix, suffix rollback, cancel/restart and truncation gates. Include reused prefixes that turn into one-token tails, since routing is purpose-based. Sharing the same SV/KV means no precision-policy change is intended; still demonstrate it, rather than infer it from source.

Only after gates and independent source review, compare OFF versus PREFILL_12 in the same frozen binary with three or more alternating short/long pairs, B8 prefill and32 fixed greedy outputs. Direct dumps preserve warm pools. Record startup separately, prefill/TTFT/decode/total wall time and exact consumed positions. Validate telemetry around requests outside timed intervals. Compare with deployed f17 as a separate check if the idle extra pool changes baseline performance. Success requires a measured useful prefill/total improvement without a reproducible decode regression; no default/UI change follows from the worker sweep alone.

## Implemented artifact and bounded gates

Frozen artifact `162bed6577d91b19a57ea0f83d244d3fbb2666e329e90733f56f347267e8580d` implements the design above. The verification flag retained for existing arithmetic operation selection is checked against the purpose enum at every pool selection. An early owner guard covers partial startup; the late attention lifetime guard joins both pools before shared model/root destruction. Construction telemetry increments after actual successful constructions; matching configuration commands leave it at2.

[Gate evidence](../reports/phase-pools-gates.json) includes exact full-vocabulary/ID comparisons against f17 for all8prefill tails, sampled verification1..4,85/452contexts and no-logits verification1..4 followed by scalar continuation. Logical counters prove12pool receives only prefill and OFF never dispatches it; aggregate groupstats equals the sum of per-pool counters. Policy mocks cover scopes/configuration/exception restoration. Existing canonical-prefix, matching-decode-token exclusion and seeded suffix/rejection/cancellation/restart checks passed in enabled mode.

The first generic canonical run stopped at an intentionally unsupported workers1 reconfiguration, not a numerical mismatch. An explicit `SV_CANONICAL_FIXED_WORKERS=8` test mode then passed the full supported lifecycle path; default fixture1/3/17 behavior remains unchanged. Second-pool startup-failure cleanup was source-reviewed but not independently fault-injected; do not describe it as a tested failure path. Actual 4096-token capacity and both timing comparisons are recorded below; deployment changes require their own validation. No default or UI change is included.

## Same-binary paired timing

The [paired report](../reports/phase-pools-paired.json) compares OFF versus PREFILL_12 in the identical162bed binary, both retaining8+12worker pools. Three alternating pairs per85/452-token prompt generated32fixed greedy outputs. Every scored pair passed exact output ID, consumed-position and positive phase-dispatch counter checks. Full-vocabulary dumps matched bit-for-bit after warmup and after the final pair of each profile; intermediate scored pairs did not have separate full-logit dumps. Telemetry was collected outside the request clock.

All six prefill and total ratios favored PREFILL_12. Short prefill median1.104× (range1.101–1.143), total1.044× (1.033–1.080); long prefill median1.093× (1.079–1.130), total1.076× (1.066–1.120). Short decode median0.969× (0.953–1.005), long0.997× (0.972–1.061). Thus prefill gains translate to total gains, but the short decode loss means the no-decode-cost objective is not yet established. These ratios must not be treated as direct speedups over deployedf17: OFF also has an idle extra pool and a separately compiled host. A direct frozen-f17 comparison and deployment gates remain necessary. No default/UI change or affinity claim is included.

## Direct deployed comparison

The [direct report](../reports/phase-pools-deployed.json) compares frozen deployed f17 with frozen162bed PREFILL_12, using three alternating85/452×32 pairs, warm pools and identical fixed output work. It preserves before hashes, the comparator's unchanged-hash assertions, and explicit post-terminal hash readings. The original gate's missing reference-hash field is documented separately: its executed command supplied the same f17 path, but no retrospective gate-time hash claim is made.

All six prefill and total pairs favored the phase candidate. Short medians: prefill1.093× (1.079–1.113), decode1.026× (1.001–1.050), total1.072× (1.043–1.076). Long medians: prefill1.179× (1.101–1.190), decode1.003× (0.926–1.037), total1.152× (1.087–1.160). All scored pairs passed output ID, consumed-position and positive candidate purpose-counter checks. Full-logit bits matched at warmup and after the final pair of each profile; intermediate scored pairs were not independently dumped. This is evidence of a total-request gain against deployment, with mixed long decode results. It does not isolate the idle pool's cost or prove a decode gain; separately linked host code, thread residency and scheduling all belong to this composition. Do not multiply it with same-binary ratios or compare runs as though host conditions were identical. No default promotion is included.

Interpretation of the direct total ratios: a median 1.072× speedup corresponds to approximately 6.7% less elapsed time, and 1.152× corresponds to approximately 13.2% less elapsed time. Speedup ratios and percentage latency reductions are different quantities.

## Actual context-capacity gate

The [capacity report](../reports/phase-pools-capacity.json) records a real fill to 4096 tokens for frozen phase artifact 162bed and frozen f17, without injected positions. After filling 4088 tokens, an eight-token prefill reaches the limit. Truncating to 4095 then executing a scalar token exercises operation 6 at the maximum causal position; truncating to 4092 then verifying four tokens exercises operation 5 there. Both final full-vocabulary dumps and IDs/positions matched bit-for-bit. A one-token prefill tail reaches 4096, and a five-token request from 4092 is rejected.

Candidate telemetry at the boundary confirms that prefill dispatches used the twelve-worker pool, scalar/verification dispatches used the eight-worker pool, and exactly two pools were constructed. Session 56923 exited successfully and all test processes closed. Explicit post-terminal binary hashes still match both frozen artifacts. The approximately 292-second test duration is correctness runtime, not benchmark evidence; unrelated compiler work could overlap. The [wrapper](../tools/check_phase_capacity.py) reuses the existing capacity test and adds phase telemetry.
