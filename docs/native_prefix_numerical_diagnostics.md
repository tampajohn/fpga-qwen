# Canonical-prefix numerical diagnostics — November13 four-hour pass

The direct-division arithmetic candidate is **not qualified for performance or
release**. Its512-token canonical variant fails the unchanged full-vocabulary
absolute-error gate at context2048, even though its newly exposed63-token and
near4K sampled checks passed. Live build19 remains unchanged. The same boundary63
failure also appeared in baseline19, so the original boundary failure cannot be
attributed to finer64/128 canonical chunks alone.

The experiment replaces reciprocal-multiply normalization/attention scaling with
direct division in isolated source. These are numerical changes, unlike a pure
cache-layout transformation. A favorable result at one fixture cannot replace
held-out validation elsewhere. No limits have been relaxed.

## Completed broad512 reports, including the failure

The serialized script completed eight additional capture/oracle reports before
stopping. Each capture repeats the same trajectory twice; the saved independent
FP32 NumPy vectors are reused with fixed maximum absolute error≤0.01, RMS≤0.002,
rtol0 and greedy-token identity. The report maxima below cover their listed
sampled positions, not all context tokens.

| Fixture | Compared positions | Maximum absolute error | Maximum RMS | Gate |
|---|---:|---:|---:|---|
| [tiny](../reports/native-cpu-norm-attention-direct-div-v1-tiny-oracle.json) | 4 | 0.002367496 | 0.000503154 | Pass |
| [short](../reports/native-cpu-norm-attention-direct-div-v1-short-oracle.json) | 8 | 0.005421638 | 0.001483050 | Pass |
| [long](../reports/native-cpu-norm-attention-direct-div-v1-long-oracle.json) | 4 | 0.005851746 | 0.001065842 | Pass |
| [boundary511](../reports/native-cpu-norm-attention-direct-div-v1-boundary511-oracle.json) | 2 | 0.004824638 | 0.001194841 | Pass |
| [boundary512](../reports/native-cpu-norm-attention-direct-div-v1-boundary512-oracle.json) | 2 | 0.005043507 | 0.001141784 | Pass |
| [boundary513](../reports/native-cpu-norm-attention-direct-div-v1-boundary513-oracle.json) | 2 | 0.004270554 | 0.000981452 | Pass |
| [context1024](../reports/native-cpu-norm-attention-direct-div-v1-context1024-oracle.json) | 2 | 0.008133411 | 0.001675268 | Pass |
| [context2048](../reports/native-cpu-norm-attention-direct-div-v1-context2048-oracle.json) | 2 | 0.012651443 | 0.001877199 | FAIL |

At2048 the absolute error is **0.012651443 >0.01**; RMS0.001877199 passes.
The teacher-forced2049 row passes both limits (maximum0.005335808,
RMS0.001093468). Greedy agreement does not override the failed full-logit gate.
The earlier [63/64 result](../reports/native-cpu-norm-attention-direct-div-v1-boundary63-oracle.json)
and [3968/3969 result](../reports/native-cpu-norm-attention-direct-div-v1-context3968-oracle.json)
remain archived alongside this failure. Together these cover the original
28-position matrix for the512 variant with one failing position; this is not
an aggregate pass. The additional63/64 fixture is separate coverage.

The multi-candidate script stopped before beginning either128- or64-token
variant's broad matrix. Their separate short-boundary results do not substitute
for that missing qualification. No fresh9/18-case quality or performance run
was authorized from this failed arithmetic. Earlier failures, including the
norm-only direct-division RMS miss, remain immutable evidence.

## Diagnostic next step, not a new acceptance contract

A sequential-step prompt control consumes the same fixed63 prompt IDs through
the B1 path and ignores intermediate predictions. It preserves two repeats,
full-vocabulary arrays and the fixed teacher continuation. This distinguishes
canonical batch-prefill from sequential processing; it is not a canonical-cache
or browser-performance test. Its results are not asserted in this document.

If further localization is needed, isolated native/NumPy copies can record the
same last-token stage boundaries by layer: input normalization, Q/K after
normalization/RoPE, rounded K/V, attention output, post-attention residual,
MLP activation and post-MLP residual. Compare finite/max-absolute/RMS errors to
locate their growth without altering math or tolerances. Different but valid
FP32 reduction associations can produce nonzero differences; the first differing
array alone does not establish an implementation defect. No stagewise capture
or result is claimed here.
