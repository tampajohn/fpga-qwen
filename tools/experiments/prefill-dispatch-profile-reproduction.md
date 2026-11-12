# Prefill dispatch attribution

This experiment measures existing synchronous dispatch boundaries; it adds no neural math to the host and changes no SV arithmetic, pool, or descriptor source. It compares the FP32-scalar/FP64-prefill hybrid against the optional FP32-prefill profile. The initial report retains its explicit pool-recreation caveat; it is not an overhead or speed claim.

Initial enabled-run averages:

| Workload/profile | Eval time | Hidden projections | Attention | Remaining eval |
| --- | --- | --- | --- | --- |
| Short, FP64 prefill | 0.784 s | 69.8% | 17.1% | 13.2% |
| Short, FP32 prefill | 0.753 s | 68.7% | 17.2% | 14.2% |
| Long, FP64 prefill | 6.083 s | 50.1% | 40.4% | 9.5% |
| Long, FP32 prefill | 5.636 s | 49.7% | 40.2% | 10.1% |

Outside-eval transport averages 0.76–0.77 ms short and 4.26–4.33 ms long, under 0.1% of prefill. Remaining eval includes useful main-SV normalization, activation and RoPE work, generated wrapper copies and coordination; it is not pure host overhead. Vocabulary projection is absent from no-logit prefill and measured separately in first prediction/decode. Dispatch scopes include scheduling, worker copies, SV arithmetic and waits. Grouped QKV/gate-up scopes cannot split their constituent matrices.

The standalone kernel repeatedly evaluated one layer's matrix with synthetic inputs and excluded host input/output copies and worker scheduling. Full-model prefill streams 28 layers, includes those costs, and spends about 40% of long-context eval in unchanged attention. Here hidden dispatch improves only about 6% short / 9% long, versus the warmed microkernel's 12–19%. Even unchanged attention changes by about 9% across long runs, demonstrating drift. These observations explain why standalone ratios are poor end-to-end predictions; they do not prove a unique microarchitectural cause.

The next substantial long-prefill target is attention, not Python transport. Current prefill submits one attention job per token per layer (12,628 for the 451-token no-logit prefix), rereading overlapping causal KV prefixes. A separately gated experiment could batch query tokens while preserving each query's arithmetic order and causal bound. This is a proposal, not measured performance.

## Continuously warm-pool confirmation

The corrected driver reads the explicitly closed dump file directly, validates full byte count and finite logits, and does not issue the historical lane barrier. One clean on/off pair per workload/profile confirms the attribution: reference short hidden/attention/remaining 70.3/16.7/13.0%; reference long 50.6/39.9/9.6%. Candidate long is 49.8/40.0/10.2%. Outside-eval transport stays below 0.1%. Short candidate attention varied to 20.4%, reinforcing that these timings are diagnostic rather than a precise speed ratio.

Prefill on/off wall ratios are 1.044/0.987 short and 0.982/0.983 long (reference/candidate). Single-token and decode phases fluctuate more. This does not demonstrate zero instrumentation cost; it does show the central long-context attribution survives removing worker recreation. All enabled/disabled final full-vocabulary logits and generated IDs match exactly. An earlier confirmation launch was stopped when a timing grant was revoked; its partial report is excluded entirely.

## Reproduction

Create a separate worktree at exact base `afe04a2` and apply `tools/experiments/prefill-dispatch-profile.patch`. The patch adapts the earlier dispatch profiler, adds only C++ timing scopes/commands and an optional builder flag, and leaves neural SV/pool sources unchanged. Build both selected hybrid B4 profiles with `--profile-dispatch`; add `--fp32-prefill-b8` only to the candidate. Supply reviewed `--shortreal-compiler` as usual. The new artifacts use distinct directory tags. The timer header is hashed unconditionally since full_model.cpp includes it even with timers disabled.

Run `python tools/profile_prefill_dispatch.py --reference <profiled-FP64-prefill> --candidate <profiled-FP32-prefill> --pairs 1 --tokens 16 --output build/profile.json` in an exclusive CPU window. The report includes runtime capabilities, binary/build provenance, checkpoint/tokenizer hashes, raw counters, prompt/output IDs, phase boundaries and driver SHA. The archived initial two-pair report explicitly notes its historical barrier asymmetry. The clean confirmation report is the final attribution check.
