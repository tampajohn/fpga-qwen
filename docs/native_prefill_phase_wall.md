# Build18 prefill wall-time diagnostic

The instrumented build18 attributes about **71–72% of near-4K prefill wall time
to attention**, compared with **39–41% at 1024 tokens**. This identifies where
to investigate long-prefill latency. It does not measure a new optimization,
CPU utilization, or an uninstrumented speedup.

| Prompt | Fresh logical reset | Append wall seconds | Attention seconds | Attention fraction |
|---|---:|---:|---:|---:|
| 1024 | 1 | 1.276360 | 0.497185 | 39.0% |
| 1024 | 2 | 1.278998 | 0.526209 | 41.1% |
| 3968 | 1 | 10.925702 | 7.783134 | 71.2% |
| 3968 | 2 | 11.684870 | 8.376800 | 71.7% |

Each context used one resident process, two fresh logical resets, the frozen
prompt IDs and one teacher-forced continuation. All four full-vocabulary arrays
per context were byte-identical to immutable build18, including repeated
predictions. The reports retain both prefill and continuation records, all
per-layer/per-chunk matrices, input IDs, artifact hashes and original report
hashes: [1024](../reports/native-cpu-build18-phase1024.json),
[3968](../reports/native-cpu-build18-phase3968.json),
[instrumentation metadata](../reports/native-cpu-build18-phase-instrumentation.json)
and [diagnostic build](../reports/native-cpu-build18-phase-build.json).

Coordinator `steady_clock` boundaries surround synchronous phases, including
worker completion and BLAS calls. The eleven phases separate setup/embedding,
input norm, QKV projections, QK norm/RoPE/KV storage, attention, output
projection/residual, post-attention norm, gate/up projections, activation,
down projection/residual and final norm. Vocabulary-head time is separate.
Rows 0–27 describe layers; row28 holds setup and final norm. Canonical chunk
counts, contiguous positions, legal phase slots and accounted wall-time sums
are checked before accepting a report.

These durations include scheduling, contention and synchronization in the
enclosing phase. They are not sums of worker CPU time, and they do not assign
sleeping-thread samples to neural work. Timestamp and storage overhead is
charged to the following phase; formatting and stderr I/O happen after the
recorded append interval. Small unassigned boundary gaps are reported. Two
visits are diagnostic observations, not a confidence interval or a performance
confirmation. First-use effects and unrelated host work were not controlled.

## Reproduction and provenance

The [source generator](../tools/experiments/prefill_phase_wall/prepare.py) and
[guarded capture driver](../tools/experiments/prefill_phase_wall/capture_phase_wall.py)
are byte-identical copies of the executed local files; their matching directory
depth preserves repository-root resolution. The
[instrumentation patch](../tools/experiments/native-phase-wall-instrumentation.patch)
contains only diagnostic additions to build18's core. No instrumentation was
added to the live engine.

The generator requires engine sources matching the immutable build18 manifest
and refuses an existing diagnostic source directory. Preserve original evidence;
use a separate checkout with the matching build18 sources and local assets for
reproduction. Build the generated source with the existing guarded native build
tool, then run the capture driver under `tools/benchmark_guard.py`, supplying
`--binary`, `--reference-capture` and a fresh `--directory`. Reference captures
are `build-18-context1024-logits/report.json` and
`build-18-context3968-logits/report.json` under `build/native-cpu/`.

The driver retains explicit stderr logs and rejects source/pack/hash changes,
malformed timing records or any full-logit difference. Sanitized reports keep
hashes of the original local records; redacted paths are provenance labels,
not executable paths. The diagnostic executable is separate from the live
build18 binary and is not eligible for deployment or headline timing claims.
