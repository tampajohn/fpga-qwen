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
variant's broad matrix. Later isolated2048 controls for both variants are
recorded below; they do not complete the missing broad matrix. Their separate short-boundary results do not substitute
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

## Qualified stage traces: observed differences, not a general bug diagnosis

The [sequential63 control](../reports/native-cpu-build19-sequential63-diagnostic-oracle.json)
passes the unchanged final gate (maximum absolute error0.00744188,
RMS0.00145475), while baseline19's batched63 path fails. Both consume the same
fixed prompt IDs. This isolates a processing-path effect on this fixture;
it does not make sequential replay a performance solution or relax the gate.

The [observational analysis](../reports/native-cpu-native63-stage-analysis-v1.json)
passed its prerequisites. The [NumPy trace](../reports/native-cpu-numpy63-layer-trace-v1.json)
retains exactly the original saved oracle's final full-vocabulary bytes.
Instrumented native [batch](../reports/native-cpu-native63-batched-trace-identity-v1.json)
and [sequential](../reports/native-cpu-native63-sequential-trace-identity-v1.json)
outputs each match their own pristine build19 reference at both selected
positions and both repeats: four full arrays per path. All364 intermediate
arrays per native path also repeat exactly. This qualifies the instrumentation
for observation; it is not evidence that the failing native path meets the
independent numerical tolerance.

At layer0, native batch and sequential **input-normalized hidden vectors are
byte-identical**. Their Q/K/V projection outputs then differ:

| Layer0 projection, batch minus sequential | Maximum absolute difference | RMS difference |
|---|---:|---:|
| Q | 0.0000123978 | 0.000000750911 |
| K | 0.00000858307 | 0.000000826856 |
| V | 0.00000131130 | 0.000000138276 |

This is a local identical-input difference between the SGEMM batch and B1
projection paths. Different floating-point reduction associations are a
plausible explanation; the trace does not establish that either implementation
is defective. After normalization/RoPE and FP16 rounding, layer0 canonical
prefix K/V maxima differ by0.00390625 and0.000244141. The archive retains all63
positions rather than attributing a history-wide maximum solely to the target
row. Both native paths already differ slightly from the oracle at layer0 input
normalization, so projection association is not the only numerical difference.

By layer27, post-MLP residual RMS differences versus the oracle are approximately
0.008146 for batch and0.006070 for sequential. Sequential is not closer at
every intermediate layer; errors grow and interact through normalized
projections, rounded KV history, attention and residual updates. Large final
error cannot be assigned to a single stage merely because it has the largest
observed difference. No per-stage acceptance threshold or winner is invented.

The full analysis preserves13 phases×28 layers for batch/oracle,
sequential/oracle and batch/sequential, plus every canonical-KV position's
metrics. Raw tensors stay local; hashes, shapes, source/build bindings and all
metrics are archived. See [native build](../reports/native-cpu-native-target62-trace-v1-build.json),
[instrumentation metadata](../reports/native-cpu-native-target62-trace-v1-experiment.json),
[batch capture](../reports/native-cpu-native63-batched-trace-capture-v1.json) and
[sequential capture](../reports/native-cpu-native63-sequential-trace-capture-v1.json).

Reproduction sources are preserved under
[`tools/experiments/numerical_diagnostics/`](../tools/experiments/numerical_diagnostics/):
byte-identical copied oracle, capture and analysis Python sources retain their
repository-root resolution at the same directory depth. Sanitized provenance
JSON uses `$REPO` placeholders and must be mapped back to local paths before
reproduction; executed local hashes remain in the reports. The
[native source generator](../tools/experiments/prepare-native-target62-trace.py)
and [observational patch](../tools/experiments/native-target62-trace.patch)
are separate from the unchanged live engine. Fresh directories and a guarded,
serialized run are required. None of these diagnostics claims timing or fixes
the failed broad direct-division candidate.


## Independent2048 controls for128/64: both fail

After the serialized broad script stopped, root explicitly ran isolated2048/2049 controls for both finer candidates. The [128-token oracle gate](../reports/native-cpu-norm-attention-direct-div-prefix128-v2-context2048-oracle.json) and [64-token gate](../reports/native-cpu-norm-attention-direct-div-prefix64-v2-context2048-oracle.json) are terminal failures under the same unchanged limits.

| Canonical chunk | Position2048 max absolute / RMS | Position2049 max absolute / RMS | Qualification |
|---|---:|---:|---|
| 512 | 0.012651443 /0.001877199 | 0.005335808 /0.001093468 | FAIL at2048 absolute error |
| 128 | 0.012651443 /0.001877199 | 0.005335808 /0.001093468 | FAIL at2048 absolute error |
| 64 | 0.012651443 /0.001877199 | 0.005335808 /0.001093468 | FAIL at2048 absolute error |

All greedy IDs match, but that does not override the full-logit failure. The [128 capture](../reports/native-cpu-norm-attention-direct-div-prefix128-v2-context2048-capture.json) and [64 capture](../reports/native-cpu-norm-attention-direct-div-prefix64-v2-context2048-capture.json) record exactly the same full-vector SHA as512 at each position and both repeats: `7fb2ed2d82bc71b98f1e21a26416c5ade13d277bcef96f4be228d097052854b7` at2048 and `45293f163804f175c4cf40e67651043dcce703ed93bdb3f4c321b11f08a346ac` at2049. This is fixture-specific equality, not a general proof that granularity never affects arithmetic.

None of the three direct-division variants qualifies for fresh quality or performance measurement. The128/64 full broad matrices remain incomplete; short-boundary successes and these isolated controls are preserved without an aggregate pass. Live build19 is unchanged, and no native source change is made from this publication.


## Fixed split-K128 projection experiment rejected

An isolated [source patch](../tools/experiments/native-prefill-splitk128-v1.patch) changed ordinary batched wide-weight projections to independent K128 SGEMM partials followed by ascending FP32 merges. [Preparation metadata](../reports/native-cpu-prefill-splitk128-v1-experiment.json) and the [completed build](../reports/native-cpu-prefill-splitk128-v1-manifest.json) bind the experiment. The hypothesis was motivated by the matched-input projection roundoff diagnostic; it was not a proved fix. It retains baseline19 norm/attention math and does not combine the failed direct-division branch. A6 MiB reusable FP32 scratch buffer and80 rather than7 SGEMM calls per layer/chunk are disclosed, but no speed claim was measured.

The [two-repeat capture](../reports/native-cpu-prefill-splitk128-v1-boundary63-capture.json) and [fixed oracle gate](../reports/native-cpu-prefill-splitk128-v1-boundary63-oracle.json) completed. **Both selected positions fail**, despite unchanged greedy IDs:

| Position | Maximum absolute error (limit0.01) | RMS (limit0.002) | Result |
|---|---:|---:|---|
| 63 | 0.012241602 | 0.002866313 | Both limits fail |
| 64 | 0.010024786 | 0.002008769 | Both limits fail |

The candidate is rejected at this tripwire. There is no2048/3968 qualification, broader quality, timing or release claim for it; no tolerance change or selective treatment of the near-limit64 row is made. Live build19 remains unchanged. This failed alternative remains evidence alongside the earlier baseline and direct-division failures.

## Distinguishing exact nonregression from absolute qualification

For a separate arithmetic-preserving optimization, full-vector byte identity to build19—including the known63 failure—can establish sampled numerical nonregression. It cannot turn that inherited failure into an independent-oracle pass. An honest record must state both outcomes: exact baseline equivalence passed for the listed arrays, while the absolute63 contract remains failed and unresolved. Existing28-position qualification stays scoped to those positions; newly exposed failures are not excluded from the record.

Such a record is not a numerical-fix claim or authorization to override an absolute-pass release requirement. If the release contract requires every declared absolute gate to pass, exact baseline identity alone does not satisfy it. Diagnostic performance evidence, if separately authorized after exactness checks, must carry the inherited numerical limitation rather than be labelled fully qualified. No new acceptance thresholds or silent release exception are introduced here.
