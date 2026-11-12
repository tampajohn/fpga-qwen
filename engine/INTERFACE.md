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
KV is FP16 at the current position before attention, not only past positions.
Output is full FP32 logits. Exceptions report errors; no silent truncation.

## Persistent process protocol v1

Executable startup: `--weights PATH --threads N --prefill-threads N --context N`.
Stdout is exclusively one JSON object per line; diagnostics use stderr.
Startup emits `{"ready":true,"protocol":1,...}` with actual precision/settings.
Commands (one line, one response):

- `reset`: clear KV/history; response `{"ok":true,"position":0}`.
- `mode greedy` or `mode chat SEED`: choose decoding; response `{"ok":true}`.
- `prefill ID ID ...`: append IDs; response has `token`, `position`, `seconds`.
- `step ID`: append one ID; same response fields.
- `dump PATH`: write current full vocabulary FP32 little-endian logits;
  response `{"ok":true,"count":151936,"position":...}`. Path is remainder
  of command line, so spaces are retained. No newline paths.
- `stats`: return current position/settings.
- `quit`: acknowledge then exit.

Greedy chooses the lowest ID on exact ties. Chat uses temperature .7, top-k20,
top-p .8, presence penalty1.5 with request history, matching existing UI values.
Errors emit `{"error":"..."}`; callers must reset/restart after inference errors.
Process owns CPU threads and mmap, with no model download or build at launch.

## Browser and benchmark adapter

New backend ID `native-cpu-f16`. Initially prefix/suffix reuse unsupported;
advertise this truthfully. Preserve ChatProtocol and existing NDJSON event
contracts, EOS handling, natural cap and cancellation. No promotion/default
selection until correctness and performance evidence. Python adapter may use
tokenizers/stdlib; no Python neural math in inference. Benchmark adapter uses
same process protocol. Root owns acceptance integration with existing gates.
