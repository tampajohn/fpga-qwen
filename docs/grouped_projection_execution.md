# Bounded descriptors for grouped B4 projections

This opt-in experiment groups Q/K/V projections sharing the same normalized input into one worker submission, and gate/up into another. The descriptors contain a matrix base, row count, width, and output offset. They describe existing dense ranges; there is no per-weight lookup dictionary, approximate skipping, cached context-dependent product, or host implementation of model arithmetic.

Each worker copies the shared B4 input once, executes the existing SystemVerilog projection task separately for each descriptor, and copies its disjoint output rows. The coordinator waits once for the whole group. Per-matrix accumulation order and precision are unchanged. Single-token dispatch and serial-worker fallback remain available; the feature requires explicit `--group-projections` in the isolated builder and does not change the default profile.

## Why the experiment was bounded

Before implementation, eight-worker microbenchmarks measured approximately 21–23 microseconds per tiny dispatch and 27–36 microseconds with width 1,024. Those timings include tiny matrix work and are not pure barrier measurements. Avoiding 84 dispatches suggests a synchronization opportunity around 1.8 ms per B4 block, plus eliminated copies. This supported a small experiment, not a prediction of a dramatic speedup. The macOS sample's whole-process wait counts include idle workers and were not interpreted as wall-time percentages. Raw microbench evidence: [grouped-projection-dispatch-profile.json](../reports/grouped-projection-dispatch-profile.json).

Per B4 block, the 28 layers previously issued 196 projection submissions. Grouping reduces them to 112: 56 groups describe 140 matrices, alongside 56 unchanged output/down projections. There are still 112 per-token attention submissions, so total prefill dispatches fall from 308 to 224. No vocabulary projection is included in these no-logits prefill counts.

Runtime counter assertions confirm **84 avoided dispatches** and **22,020,096 bytes of avoided worker activation copies** per B4 block with eight workers. The input port retains stride four even for partial blocks. These counters exclude main-model DPI temporaries and do not claim reduced weight traffic. The same fixture checks counts at one, three, eight, and seventeen workers.

## Correctness and validation bounds

The descriptor validator is shared production code in `sim/projection_descriptors.h`. It checks two or three jobs, one through four tokens, non-null buffers, width exactly 1,024, rows in 1–3,072, contiguous nonoverlapping output offsets, total rows at most 6,144, and every complete weight span. Validation runs before worker publication. The new header is included in the isolated build manifest and Makefile dependencies.

The scheduler transport mock tests both condition-variable and hybrid publication paths. Invalid counts, token counts, null buffers, holes/overlaps, wrong width, oversized rows/totals, and negative/out-of-bounds weight spans reject without any worker evaluation or counter change. A normal job following a group verifies that stale descriptors are not reused. These are scheduler/validation tests, not numerical model substitutes.

Real tests compare full-vocabulary bit patterns and sampled output triples against an independent baseline across worker counts 1/3/8/17 and block sizes 1/2/3/4. The existing block suite additionally passes causal future-token exclusion, truncation/replacement, reset/restart, malformed-wire validation, and cancellation. The expensive 4,096-token boundary test was not repeated because grouping does not change KV allocation or the existing block capacity guard. Short (85-token) and long (452-token) benchmark gates compare eight subsequent full-logit positions exactly, with maximum error zero.

The ordinary no-group profile also builds successfully. Its sampled B4 output and full logits, followed by a single-token output/full-logit check, are bit-identical to the independent live baseline. Its shared scheduler mock paths remain covered. Evidence: [default isolation](../reports/grouped-projection-default-isolation.json).

## Initial paired evidence

Both reference and candidate use B4 prefill, eight workers, the same prompt IDs, and 16 greedy generated tokens. Each request resets logical KV. The first two pairs alternate reference/candidate then candidate/reference; a third reference/candidate pair uses fresh warmed processes and is identified separately in the raw report. Every measured output sequence matches exactly, with no EOS in the fixed workload.

| Prompt tokens | Prefill speed ratios, candidate vs reference | Median ratio | Median TTFT reference → candidate |
|---:|---|---:|---:|
| 85 | 1.061×, 1.116×, 1.054× | 1.061× | 1.086 → 0.989 s |
| 452 | 1.092×, 1.063×, 1.109× | 1.092× | 7.445 → 6.898 s |

Ratios are calculated per pair; a ratio of aggregate medians need not equal their median. Another agent's PGO driver/model were stopped with SIGSTOP during these scored runs, so they consumed no CPU but retained memory residency. User applications remained active. This caveat and the fresh-process third pair limit how broadly these small local gains should be interpreted. Raw evidence: [grouped-projection-paired.json](../reports/grouped-projection-paired.json). Its historical `serial` field names the reference artifact; **reference_block_size is 4**, not single-token prefill.

Extracting validation into the standalone header and rebuilding produced a byte-identical executable: SHA256 `675ace2b0b9de087cff6d648d7256dacd2f896bfe5ace2659a88572ddb4eb0ea` before and after. The real grouped/tail fixture was rerun after rebuilding. Build options, commands, and source hashes are in [grouped-projection-manifest.json](../reports/grouped-projection-manifest.json).

After final rebuilding and negative-validation tests, an independent warmed confirmation pair again improved prefill: **1.038× for 85 tokens and 1.076× for 452 tokens**, with exact logits and identical generated IDs. PGO training had finished before this confirmation, so no paused training model remained. Raw evidence: [final confirmation](../reports/grouped-projection-final-confirmation.json). These small, consistent local gains support integration review but do not establish a general confidence interval or a reduction in required dense arithmetic. The experiment remains opt-in in this branch.

## Reproduction

Build with the existing validated B4 flags plus `--group-projections`. Compare using `tools/benchmark_blocks.py --reference-binary /path/to/baseline --binary /path/to/grouped --reference-block-size 4 --block-size 4 --profiles short,long --pairs 3 --tokens 16 --gate-tokens 8`.

Run the real partition/counter fixture with `SV_GROUP_REFERENCE=/path/to/baseline SV_GROUP_BINARY=/path/to/grouped uv run --locked python -m unittest tests.test_grouped_projections -v`; run the compiler-only scheduler tests with `uv run --locked python -m unittest tests.test_projection_scheduler -v`.
