# Isolated FP32 KV storage fixture

This experiment validates storage and conversion only. It does not change model KV precision, inference profiles, cache policy or defaults. Build and check with:

```
python3 tools/build_fp32_kv_fixture.py --compiler-root /path/to/reviewed/verilator-shortreal
```

The module has parameterized fixed internal `shortreal keys[WORDS]` and `values[WORDS]`, with real write/read ports. Its clocked write explicitly narrows in SV; the real read ports explicitly widen in SV and keep storage reachable. The generated root contains `VlUnpacked<float,16>` for each array and double real ports. The harness additionally reads the generated float elements through `rootp` and checks exact widening against every valid SV read. That transport does not invoke a worker model or narrow/write on the host. All generated model code uses the reviewed compiler's strict flags; no generated source is edited.

The [report](../reports/fp32-kv-storage-fixture.json) records source/compiler/executable hashes and checks initial zeros, same-edge write-before-read, nearest-even ties, signed zeros, subnormals, finite overflow to infinities, first/last indices, rejected negative/past-end addresses, disabled writes, and all-slot preservation after invalid writes. Expected rounding uses the existing independent exact-rational oracle, with explicit signed-zero and overflow handling. This is a tiny 16-word fixture, not a quality or speed result.

A production shape would need 117440512 elements per K/V array: 939524096 bytes total (896 MiB), before model/object overhead. It would require one coordinator-owned heap module, no duplicate old KV arrays, and synchronized read-only callbacks. Neither that allocation nor integration has been tested here. This fixture uses flat indices; layer/position/feature mapping, logical truncation/reset, causal bounds, cross-route cache behavior, model quality and performance remain separate integration gates. Initial zeroing is fixture construction only; no reset/truncate command clears storage. No shortreal ports, DPI float ABI, public FLOAT annotations or host neural arithmetic are used.
