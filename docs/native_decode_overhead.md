# Native decode overhead diagnostic

Build20's measured adapter/protocol overhead is small on this sampled workload. It does not currently justify prioritizing Python-to-native command batching. This is a component diagnostic, not a throughput comparison or proof about all contexts.

The real `NativeCPUChat` adapter processed the frozen 452-token prompt with a 128-token cap, 12 decode workers and 6 prefill workers, without prefix reuse or suffix drafting. Greedy generation stopped naturally after 60 visible tokens; seeded chat stopped after 75. Each mode ran two plain and two instrumented visits in reversed order, with process startup and request warmup outside attribution. All visible token IDs, text, and finish reasons matched within each mode. The adapter suppresses the terminal EOS token ID, so this checks the EOS stop reason rather than independently comparing the actual terminal token ID.

| Mode | Decode command residual / command wall time, two instrumented visits | Whole-generation tokenizer time | Whole-generation JSON encoding time |
|---|---|---|---|
| Greedy | 0.77%, 0.53% | 1.08–1.27 ms | 0.83–0.99 ms |
| Seeded chat | 0.77%, 0.86% | 1.55–1.62 ms | 1.12–1.17 ms |

The residual subtracts the native command's reported model/history/sampling interval from Python command wall time. It includes parsing, response formatting, pipe handling, Python work, scheduling, and clock effects; it is **not pure IPC time**. Chat residuals were about 64–65 microseconds per step. Existing native timing combines model execution and sampling, so it cannot yet isolate the serial vocabulary scan or sampled top-20 selection.

JSON timing reproduces server event encoding only. It excludes actual HTTP/socket writes, backpressure, browser rendering, and network latency. Native execution includes pool synchronization and worker scheduling; these measurements do not establish how much synchronization is removable. Plain/instrumented throughput varied materially across visits; no performance claim is made from those differences or from these two repetitions.

The complete sanitized observations, including every visit, are retained in [greedy](../reports/native-cpu-build20-overhead452-greedy-v1.json) and [seeded chat](../reports/native-cpu-build20-overhead452-chat-v1.json), with raw-report and helper hashes in the [archive inventory](../reports/native-cpu-build20-overhead452-archive-v1.json). The executable measurement helper is [profile_native_decode_overhead.py](../tools/profile_native_decode_overhead.py). The existing independent numerical limitations of build20 remain unchanged; this stream diagnostic is not a replacement for numerical or quality gates.

An isolated source-only follow-up splits native model, consumed-history, and sampler wall time with coordinator timestamps. It must first preserve the baseline trajectories and full logits; no result from that follow-up is included here.
