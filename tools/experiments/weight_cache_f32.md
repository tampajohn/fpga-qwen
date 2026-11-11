# Optional startup-expanded FP32 weight transport

This experiment was rejected: the cache slowed every measured pair and increased
startup time and resident memory. The production source retains BF16 transport.

The unapplied source and helper patches reproduce the experiment against commit
`e0da2f5`. Apply them in an isolated checkout of that revision, then build with
`--weight-cache f32` in addition to the selected unsigned-index flags. The helper
supplies `-DWEIGHT_CACHE_F32` and a distinct build tag; no SV define is required.
Run the transport checker only after applying the source patch.

The patch expands each mapped 16-bit word into its IEEE binary32 bit pattern
once at startup, then makes the contiguous anonymous mapping read-only with
`mprotect`. Checked, validated-unchecked, and row accessors widen those cached
floats to SV `real` doubles. Span validation and default BF16 mmap behavior are
unchanged. All finite BF16 values, including signed zero and subnormals, are
exactly representable in both binary32 and binary64. This is a storage/transport
experiment, not lower-precision arithmetic or host inference.

The entire file is addressed in 16-bit word offsets, so the cache includes a
small amount of safetensors header/padding. This preserves existing offsets and
checks. Header bit patterns are copied as bits; no float arithmetic is performed
on them. Only validated model tensor locations are used as neural weights.

Memory is `4 * floor(checkpoint_file_bytes / 2)` additional anonymous bytes,
3,006,600,656 bytes (about 2.80 GiB) for the measured 1,503,300,328-byte
checkpoint, alongside the original file mapping
and existing roughly 1.75 GiB KV allocation. This checkpoint stores both embedding and output-head tensors; the estimate
must include both. The cache doubles weight storage
width versus BF16 and may hurt cache/bandwidth performance. The startup pass
faults the checkpoint and initializes every cache page; startup latency, cold
TTFT and RSS must be reported. Startup bytes/time are logged to stderr before
the ready handshake, so this cost cannot silently disappear into prefill.

## Transport-only test plan

Run in a lightweight standalone harness before any model benchmark:

1. Supply all 65,536 uint16 patterns as synthetic mapped words. Expand through
   the patched startup routine. For every finite BF16 pattern, compare binary64
   result bits from checked/unchecked accessors against the existing bit-expand
   then widen path. Include signed-zero bit equality; do not use approximate
   float equality. Record infinities and NaN classification separately, without
   promising NaN payload preservation across float-to-double conversions.
2. Exercise rows of lengths 1 and 3072 at valid endpoints, verifying each double
   bit pattern. Stub `svGetArrayPtr` only; no matrix multiplication is needed.
3. Check negative addresses, address==words, negative spans, crossing-end spans,
   zero/overlarge row lengths and null destinations still fail identically.
   Do not invoke the intentionally unchecked accessor outside a validated span.
4. Launch concurrent read-only access from eight threads over deterministic
   indices. Ensure the cache finishes initialization before any pool starts.
   Verify the mapping is read-only in a child process: an attempted write must
   fault, without terminating the test runner.
5. Check allocation/protection errors produce failure before `ready`, and normal
   shutdown joins workers before unmapping. Compare a build without the flag to
   ensure the original path is still selected.

Then run the existing full-logit, sampler, cache/restart and projection-span gates
against the isolated candidate. Alternate matched performance runs with baseline
at the same worker/lane count; measure startup separately, steady decode, prompt
TTFT and peak RSS. A transport instruction reduction is not evidence of a model
speedup. No compile, inference, speed or quality result is claimed yet.

## Reviewed build and measurement protocol

The separate `weight_cache_f32_helper.patch` adds `--weight-cache none|f32`
(default `none`), a `-weight-cache-f32` artifact suffix, and the C++ define only.
Both patches remain unapplied pending the exclusive build window. Compare cache
on/off with identical unsigned-index, attention, projection-buffer, lane and
worker settings; do not bundle a KV-layout or CPU-target change.

The mapping returned by `mmap` is page-aligned, and `mprotect` can accept the
non-page-multiple byte length. Initialization completes before pool construction,
and normal shutdown joins the pool before unmapping. If protection fails, the
allocation is explicitly unmapped. Other fatal initialization errors exit the
process, which reclaims any still-live mappings. Existing model and tensor span
validation is unchanged; the full-file word-index convention is preserved.

Before model timing, compile a transport-only fixture from the actual patched
accessor/startup definitions (with Verilator array-pointer transport stubbed),
then execute the finite-pattern, boundary and read-only tests above. Keep this
fixture separate from inference timing. Compare full vocabulary dumps against
cache-off at the same eight-accumulator setting and require bit identity, then
run sampling, prefix replacement and reset tests. A bit-expansion cache should
not need a floating-point tolerance exception.

For model timing, record the SHA256 of each executable and checkpoint and use
identical 85/452-token prompt IDs, eight workers, eight accumulators and 32 greedy
output tokens. First record launch-to-ready time and the cache's stderr
`bytes`/`startup_seconds` line. Sample child-process RSS after ready, after prefill,
and after decode using `ps -o rss= -p PID` (KiB on this host); label these measured
snapshots rather than peak memory. For an actual peak, launch each standalone
process under `/usr/bin/time -l` and retain its maximum-resident-set-size field
separately. Do not substitute allocation size for measured RSS.

Use the interleaved AB/BA benchmark for repeated steady timings, with both
processes resident but only one inferring at a time. Record startup separately
from request TTFT and report both their sum and the individual measurements for
a first-use experience. Repeated process launches do not flush the operating
system file cache: call them fresh-process measurements, not cold-disk tests.
The widened cache may trade a few conversion instructions for twice the weight
read bandwidth, so reject it if the paired measurements do not show a useful win.

## Result: rejected (2026-11-11)

Tested from base commit `e0da2f5`, using unsigned indices, dynamic KV, eight
workers and eight accumulators. After the experiment, only the cache changes in
`sim/full_model.cpp` and the build helper were reverted; the two patches preserve
reproduction. Apply both patches on that base, then build:

```sh
uv run --locked tools/build_projection_variant.py --lto --unchecked \
  --masked-activation --unsigned-indices --fast-attention --compact-projection \
  --weight-cache f32
uv run --locked python tools/experiments/check_weight_cache.py
```

Candidate SHA256:
`4e6b869a04f4477f111dbb5e5ecaf3b58aa32e572220f0c0a2b649975e2784a0`.
Baseline SHA256:
`becad933e07b45cc027717ae8f04b842c2db46d32b5a5b3e0bc736a6e85cb07e`.

The actual transport definitions passed all 65,536 BF16 patterns: 65,280 finite
patterns bit-exact, 254 NaN classifications, and both infinities. Signed zero,
row endpoints, invalid bounds, eight concurrent readers and a child-process
attempted write to the read-only mapping passed. Five continuation positions
after an 85-token prompt had bit-identical full-vocabulary logits, and all paired
greedy token sequences matched. No broader quality claim is needed to reject the
candidate on performance.

Two interleaved AB/BA pairs with 32 output tokens:

| Context | Cache-off tok/s | Cache-on tok/s | On/off ratio |
| --- | ---: | ---: | ---: |
| Short, pair 1 | 34.65 | 32.88 | 0.949 |
| Short, pair 2 | 33.46 | 31.82 | 0.951 |
| Long, pair 1 | 14.57 | 11.51 | 0.790 |
| Long, pair 2 | 15.09 | 13.42 | 0.889 |

The cache was slower in every measured pair. Warm TTFT also regressed: baseline
TTFT divided by candidate TTFT ranged 0.893–0.928 short and 0.823–0.882 long.
Launch-to-ready took 0.132 seconds without the cache and 0.447 seconds with it;
the candidate separately logged 0.279 seconds to populate/protect the cache.
These are one launch each with no disk-cache eviction, not cold-disk estimates.

Measured RSS snapshots: baseline 1,820.7 MiB at ready and 2,959.0–2,959.3 MiB
after requests; candidate 6,120.6 MiB at ready and after requests. Each process's
RSS includes shared/file-backed pages, so these are neither private-memory nor
combined physical-memory estimates. They are snapshots, not measured peaks.
The explicit new anonymous allocation was 3,006,600,656 bytes (~2.80 GiB);
startup also faulted the original file mapping. Do not equate the RSS difference
with the new allocation alone.

Read-only disassembly confirmed that cached transport removes BF16 expansion
`shll.4s` instructions. That instruction reduction did not compensate for wider
weight storage in these measurements. The production default remains BF16 mmap.

Results: [paired timing](../../reports/weight-cache-rejected-paired.json),
[RSS/startup](../../reports/weight-cache-rejected-memory.json), and
[exact-logit gate](../../reports/weight-cache-rejected-gate.json).
