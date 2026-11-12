# Plain native engine interface

CPU-only original implementation. No llama/ggml neural kernels or runtime
dependencies. C++20 plus Apple Accelerate is allowed; no Torch or GPU.

## Weight pack v1

Exporter creates `weights.bin`, `weights.bin.index`, `weights.bin.manifest.json`.
Binary first 64 bytes: ASCII `QWCPU01` followed by 57 zero bytes. All tensor
offsets are absolute, aligned to 64 bytes, and non-overlapping. Little endian.
The text index starts `QWCPU01 310` then exactly 310 whitespace-separated rows:

`checkpoint_tensor_name dtype rows columns offset_bytes length_bytes`

dtype is `F16` for 2D matrices and `F32` for 1D norms. Norms use rows=length,
columns=1. Names retain original checkpoint spelling. Store the embedding once;
omit `lm_head.weight` after verifying it equals the embedding. Core reuses the
embedding as output weights. Validate exact names/shapes, all bounds/lengths,
duplicates, header and index count. Manifest records checkpoint, index, binary
and tensor hashes plus rounding rules. Conversion: BF16 widened to FP32, 2D
rounded to FP16, 1D preserved FP32. Reject nonfinite or overflowed values.

## C++ ownership

`core.hpp` is the fixed API. Core owner writes `core.cpp` and optional private
headers only. Root owns `main.cpp`, build tooling and core API changes.
`prefill` appends all IDs and computes logits only at its last position; `step`
appends one ID and computes logits. `reset` invalidates all logical KV/logits.
`truncate(position)` retains KV before that position and invalidates logits,
including when position equals the current position. It checks the range
`0..current_position` before changing state; invalid targets leave the core
state unchanged. The CLI error handler subsequently resets all state on errors.
KV is FP16 at the current position before attention, not only past positions.
Output is full FP32 logits. Exceptions report errors; no silent truncation.

## Persistent process protocol v1

Executable startup: `--weights PATH --threads N --prefill-threads N --context N`.
Stdout is exclusively one JSON object per line; diagnostics use stderr.
Startup emits `{"ready":true,"protocol":1,...}` with actual precision/settings.
Commands (one line, one response):

- `reset`: clear KV/history; response `{"ok":true,"position":0}`.
- `rewind N`: retain the first N consumed token IDs and their KV; rebuild sampler
  presence from those retained IDs and discard logits; response
  `{"ok":true,"position":N}`. Require `0 <= N <= current_position`.
- `mode greedy` or `mode chat SEED`: choose decoding and reset the RNG;
  response `{"ok":true}`. Call again for each request, including cache hits.
- `prefill ID ID ...`: append IDs; response has `token`, `position`, `seconds`.
- `step ID`: append one ID; same response fields.
- `dump PATH`: write current full vocabulary FP32 little-endian logits;
  response `{"ok":true,"count":151936,"position":...}`. Path is remainder
  of command line, so spaces are retained. No newline paths.
- `dump-hidden PATH`: diagnostic export of the current final RMS-normalized
  1024-element hidden vector, already computed for the current logits. Writes
  FP32 little endian and responds with
  `{"ok":true,"count":1024,"position":...,"format":"f32le"}`. It adds no
  forward computation. Reset/rewind invalidates this export just like `dump`.
  The C++ `hidden()` accessor returns a borrowed `span<const float>` valid only
  until the next model mutation; callers must copy it if they need persistence.
- `stats`: return current position/settings.
- `quit`: acknowledge then exit.

Greedy chooses the lowest ID on exact ties. Chat uses temperature .7, top-k20,
top-p .8, presence penalty1.5 with request history, matching existing UI values.
Errors emit `{"error":"...","position":0}` and reset all model, consumed-token
history and sampler-presence state, including invalid rewind commands. Callers
must reset/restart after inference errors. Rewind does not itself reset the RNG.
Process owns CPU threads and mmap, with no model download or build at launch.

## Browser and benchmark adapter

Backend ID `native-cpu-f16`. Startup/stats advertise `prefix_reuse: true` and
`prefix_reuse_chunk: 512`; older binaries advertising false require fresh prefill.
Exact reuse is restricted to complete canonical 512-token prefill chunks. The
adapter intersects the new token common prefix with previously completed full
prefill chunks, rounds down to 512, and retains at least one new-request token
to recompute logits. It must not promote decoded KV or a partial prefill chunk
into that cache: changing SGEMM batch shapes can change numerical results.
After rewind, prefill the entire remaining suffix so its chunk boundaries match
a fresh request. These restrictions belong to the adapter; the core rewind
primitive permits arbitrary valid positions without promising fresh-request
bit identity for arbitrary batch histories. Suffix drafting remains unsupported.
Preserve ChatProtocol and existing NDJSON event
contracts, EOS handling, natural cap and cancellation. No promotion/default
selection until correctness and performance evidence. Python adapter may use
tokenizers/stdlib; no Python neural math in inference. Benchmark adapter uses
same process protocol. Root owns acceptance integration with existing gates.
