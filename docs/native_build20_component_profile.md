# Build20 buffered component diagnostic

Fresh instrumented build20 measurements show that hidden projection groups remain the largest part of observed decode wall time. The earlier build18 attribution is historical: build19/20 changed KV layout and paired prefill. These new observations support investigating model work rather than assuming transport is the main bottleneck. They do not identify a specific hardware cause or establish an optimization speedup.

| Prompt / warmed repeat | Projection groups + residuals | Attention | Vocabulary head |
|---|---:|---:|---:|
|85 /0|75.0%|10.1%|12.0%|
|85 /1|75.0%|10.3%|12.0%|
|3968 /0|51.4%|36.3%|9.9%|
|3968 /1|58.2%|32.0%|7.8%|

The projection category sums QKV, output projection plus residual addition, gate/up, and down projection plus residual addition. It is not a pure matrix-kernel timer. Each request produced32 predictions; the table aggregates its31 ordinary decode calls, spanning consumed contexts86–116 or3969–3999. Prefill is retained separately in the reports, not mixed into these percentages.

[85-token report](../reports/native-cpu-build20-components85-v1.json), [3968-token report](../reports/native-cpu-build20-components3968-v1.json), and [derived observations/provenance](../reports/native-cpu-build20-components-v1-archive.json) retain all first-use and warmed records, output IDs, off/on controls, and hashes. First-use denotes a fresh process, not guaranteed cold hardware or filesystem caches. Near4K warmed mean model decode times differed materially between repeats (14.22ms and18.47ms); two controls do not estimate a reliable throughput effect.

## Scope and correctness

Eleven coordinator-clock intervals surround the existing synchronous phases; vocabulary projection is timed separately. Durations include worker waits, scheduling and contention. They are elapsed model wall intervals, not summed worker CPU shares, removable synchronization cost, or causal microarchitectural attribution. Clock/storage overhead can perturb the measured phases and is charged to following work; residual unassigned time is retained.

A fixed256-record buffer collects data without per-step diagnostic I/O. `profile-on`, `profile-off`, `profile-reset`, and `profile-read` are confined to this isolated executable. Export happens after the full request. Ordinary response JSON remains, and no tensor dump or diagnostic export occurs inside measured decode blocks. Each visit explicitly resets counters; source checks validate phase calls, positions and nonoverlapping accounted durations. B2 verification is deliberately not attributed.

Every full-vocabulary position in32-token reference trajectories was compared against pristine build20 with profiling both disabled and enabled:128 successful full-array comparisons across the two contexts. All timed output IDs also match. This is finite relative identity, not an independent absolute-oracle repair; build20's disclosed63/64 oracle failures remain failures. The diagnostic source was not deployed.

The [build manifest](../reports/native-cpu-build20-components-v1-build.json), [instrumentation metadata](../reports/native-cpu-build20-components-v1-instrumentation.json), and [source patch](../tools/experiments/native-component-profile-v1/candidate.patch) bind the experiment. Exact [generator](../tools/experiments/native-component-profile-v1/prepare.py.txt) and [capture driver](../tools/experiments/native-component-profile-v1/capture.py.txt) snapshots must be restored to their recorded original paths before reproduction; their root resolution is relative to those paths. Root serialized all execution. No raw float dumps are published.
