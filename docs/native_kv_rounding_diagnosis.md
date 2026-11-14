# Numerical diagnosis: captured KV inputs dominate the first attention difference

The completed [controlled-input attribution](../reports/native-cpu-kv-attention-attribution-v1.json)
finds that **rounded key differences dominate layer 0 attention drift on the
63-token fixture**. It does not identify a bad Half conversion or establish the
cause of the final-logit failure. The next useful observation is all-prefix
layer 0 data before KV rounding, rather than another attention-kernel change.

This diagnostic uses the previously identity-qualified native batch, native
sequential and NumPy traces at zero-based target position 62. All 168 target KV
cast checks (28 layers × 3 paths × K/V) are bit-identical: casting each captured
precast tensor to FP16 and widening produces exactly its captured canonical
target KV. This rules out a differing cast result on those sampled values; it
does not rule out sensitivity to different values arriving at the same cast.

## Local input differences versus attention arithmetic

All counterfactuals use the same FP64 attention calculation on captured inputs.
The eight combinations of native/oracle Q, K and V permit a three-input Shapley
vector decomposition, including interactions. This is a local diagnostic
calculation, **not a replacement network oracle**. Its algebraic reconstruction
check changes no neural acceptance tolerance.

| Layer | Actual batch–oracle attended RMS | Input difference under common kernel | Native attention arithmetic residual |
|---:|---:|---:|---:|
| 0 | 1.31e-6 | 1.30e-6 | approximately 1e-8 |
| 1 | 6.15e-6 | 6.15e-6 | approximately 2e-8 |
| 2 | 1.126e-5 | 1.126e-5 | approximately 1e-8 |
| 27 | 2.678e-3 | 2.678e-3 | approximately 1e-6 |

At layer 0, the rounded-key contribution has RMS 1.27e-6 and signed projection
0.953 onto the total input-driven difference. Rounded values contribute
RMS 2.9e-7 / projection 0.052; queries contribute RMS 2e-8 / projection −0.005.
These projections describe directional contributions, not percentages of final
network error. Magnitudes are not additive and negative contributions represent
cancellation.

The dominant input changes later: at layers 1/2, rounded values have signed
projections 0.902/0.895. By layer 27, values contribute 0.579, keys 0.233 and
queries 0.188. Sequential replay is not closer to the oracle at every layer.
The report preserves all 28 layers rather than selecting only favorable ones.
Small local attention arithmetic residuals compared with captured-input effects
argue against rewriting QK/softmax/PV as the next diagnostic on this fixture.
They do not prove that every attention implementation is equivalent.

## Rounding sensitivity is visible; whole-history precast evidence is missing

For the layer 0 target row, batch versus oracle differences are:

| Tensor | Precast RMS | Postcast RMS | Post/pre RMS ratio | Changed postcast elements |
|---|---:|---:|---:|---:|
| K after norm/RoPE | 9.704e-6 | 2.158e-5 | 2.224 | 4/1024 |
| V projection | 1.302e-7 | 3.576e-7 | 2.748 | 3/1024 |

Across the entire 63-token layer 0 canonical history, 204 key and 298 value elements
out of 64512 differ. The trace has only the target row's precast tensors, so its
rounding ratio cannot be generalized to the history. Sequential-versus-oracle
target differences are actually reduced by rounding in this layer, another
reason not to assert a universal amplification factor.

Earlier matched-input projection work found byte-identical native normalized
inputs followed by differing batch/B1 Q/K/V projections, and larger SGEMM RMS
roundoff on that local exact-product reference. The layer 0 packed Q/K/V matrices
matched the checkpoint BF16→FP16 conversion. Together, these observations support
investigating projection differences crossing FP16 bins, with normalization and
RoPE as additional transformations. They do not establish a library defect,
global pack correctness or a final-model fix. Earlier direct-division and
B1-projection candidates passed 63 but failed 2048 and remain rejected.

## Next controlled observation

Capture **all 63 layer 0 rows only**, with:

1. Embedding/input-normalized hidden vectors and K/V projections.
2. K after normalization, then after RoPE, before FP16 rounding.
3. Canonical rounded K/V for those same positions.

Use original batch, sequential and independent-oracle paths. Observers must not
change arithmetic or evaluation order. Require native final arrays to match the
pristine binary on both selected prediction positions and repeats, and oracle
final arrays to match its persisted original reference before interpreting any
new stage data. Include every row/channel, not only large-error elements.

Then count and locate half-bin crossings, compare pre/post-rounding differences
for the complete history, and perform matched-input stage comparisons to separate
projection association from K-normalization/RoPE effects. The current local
Shapley result identifies which input warrants this capture; it does not by
itself attribute the final-logit discrepancy to those crossings.

After identifying the mechanism on 63, repeat the same narrowly selected
observations on fresh 64 and fresh 2048 with their existing teacher continuations.
Do not launch another arithmetic correction before this distinction is clear.
Any eventual correction retains max absolute 0.01, RMS 0.002, rtol 0 and greedy
identity across the full existing and fine-boundary matrix before timing.
Build20 runtime and its known absolute failures are unchanged by this analysis.

## Reproduction and provenance

The [helper snapshot](../tools/experiments/analyze-kv-attention-attribution-v1.py.txt)
is byte-identical to the independently reviewed, root-executed source. Restore
it to `build/native-cpu/numerical-diagnostics/analyze_kv_attention_attribution_v1.py`
so its repository-root resolution remains correct. The
[provenance record](../reports/native-cpu-kv-attention-attribution-v1-provenance.json)
binds the source and report. Raw trace tensors remain local with hashes in the
report; unrelated later runtime-source changes are not misrepresented as changes
to these historical trace artifacts.

```sh
.venv/bin/python tools/benchmark_guard.py --name 'Offline KV rounding and matched-input attention attribution' -- .venv/bin/python build/native-cpu/numerical-diagnostics/analyze_kv_attention_attribution_v1.py --output build/native-cpu/kv-attention-attribution-v2.json
```

This command repeats diagnostic subkernel arithmetic only, with a fresh output;
it performs no network inference, build or timing. The completed v1 report is
preserved without overwriting it.

## Completed all63 observation: native path divergence begins at projection

The new [native capture](../reports/native-cpu-layer0-all63-native-capture.json)
records all 63 rows of layer 0, with 11 phases per row. Its batch and sequential
paths each match pristine build20 full-vocabulary outputs at contexts 63 and 64,
across both repeats; all recorded phase bytes also repeat exactly. The separate
[independent oracle capture](../reports/native-cpu-layer0-all63-oracle-capture.json)
records the same 693 vectors and exactly matches its persisted original final
logits. These are **instrumentation identity checks**, not new passes of the
native absolute numerical gate. The existing 63/64 failures remain open.

The [native batch-versus-sequential analysis](../reports/native-cpu-layer0-all63-native-analysis.json)
finds identical embedding and input-normalized vectors across every row. The
first observed divergence is the matrix projection stage. Over all rows, K
projection RMS difference is 3.224e-7; after K normalization it is 1.401e-5, and
after RoPE it remains 1.401e-5. Canonical FP16 K RMS difference is 4.935e-5,
with 220 of 64512 elements differing. V projection RMS difference is 9.545e-8;
canonical V differs in 330 elements, with RMS 2.675e-6. All four native K/V casts
across the full history exactly reproduce their captured canonical values.

These observations isolate the first native-path divergence to projection on
identical normalized inputs. They show normalization and rounding change the
magnitude and sparsity of those differences; they do not establish which path
is closer to the independent oracle, nor explain the final-logit failure by
themselves. The prepared matched native/oracle analysis will compare all phases
and identify numerically different FP16 bins, including exact adjacent-bin
midpoints. No further arithmetic candidate is justified merely by these local
RMS changes. Earlier projection and normalization fixes that failed 2048 remain
rejected.

The [provenance inventory](../reports/native-cpu-layer0-all63-provenance.json)
archives source/build records and executed helper snapshots. Capture summaries
omit raw tensor files and repetitive tensor-hash inventories; they retain the
SHA256 of each complete local execution report. The native analysis preserves
all per-position metrics. This work changed no serving code, oracle tolerance,
release status, or performance claim.

## Independent-oracle comparison: normalization differs before projection

The completed [all-row oracle comparison](../reports/native-cpu-layer0-all63-oracle-analysis.json)
adds an important distinction: native batch and sequential share identical
normalized inputs, but those inputs already differ slightly from the independent
oracle. Their embedding vectors are exact; input-normalized RMS difference from
the oracle is 1.446e-8. Thus projection is the first **native-path** divergence,
not the first native-versus-oracle divergence.

| All63 layer0 tensor | Batch–oracle RMS | Sequential–oracle RMS |
|---|---:|---:|
| K projection | 3.120e-7 | 8.159e-8 |
| K post-normalization | 1.354e-5 | 3.430e-6 |
| K after RoPE | 1.355e-5 | 3.447e-6 |
| Canonical K | 4.918e-5 | 2.753e-5 |
| V projection | 9.142e-8 | 3.869e-8 |
| Canonical V | 2.081e-6 | 2.119e-6 |

All six complete-history FP16 casts match their own captured canonical arrays.
Batch K differs from the oracle in 204 elements, including 183 adjacent-bin
changes; sequential K differs in 120, including 106 adjacent-bin changes. For V,
the counts are 298/284 and 226/225 respectively. These are numerically different
bins; signed-zero bit-only differences are separately retained in phase metrics.
The full report retains every differing element, not just examples.

One concrete batch K example is position0/head0/channel51. Its precast value is
2.980232e-8 above the exact adjacent-bin midpoint, while the oracle value is
6.854534e-7 below it. Both conversions correctly choose their respective bins.
This demonstrates rounding sensitivity to upstream values, not a faulty cast.
Sequential V has fewer differing bins but slightly higher canonical RMS than
batch V, so neither bin count nor projection RMS guarantees a better final result.

The next controlled diagnostic should replay **normalization on identical
captured inputs**, before proposing another network change. Use all63 embedding
rows and every projected Q/K head from each captured path. Evaluate the unchanged
native normalization and unchanged NumPy normalization on each same input and
same FP32 norm weights. First require native own-input results to reproduce
captured native outputs and NumPy own-input results to reproduce captured oracle
outputs. Then use a fixed two-factor comparison of input origin and normalization
implementation to separate input perturbation from operation differences.

If that reproduction succeeds, separately expose rounded-square sum, denominator,
and output scaling. The native implementation sums FP32-rounded squares in FP64,
rounds the sum to FP32, and multiplies by a rounded reciprocal; the oracle uses
its existing FP32 mean and direct division. A predeclared diagnostic comparison
of these reduction/scaling choices may explain the difference, but cannot serve
as a new acceptance oracle or justify a network patch on its own. The previous
direct-division candidate's 2048 failure remains decisive. No additional network
variant or timing test follows from these results automatically.

The [analysis helper snapshot](../tools/experiments/analyze-layer0-all63-oracle-v1.py.txt)
and [binding record](../reports/native-cpu-layer0-all63-oracle-analysis-provenance.json)
preserve the executed source and report hashes. Both this analysis and its
predecessor are offline observations with unchanged max-absolute0.01/RMS0.002
acceptance limits; neither clears the existing native numerical failures.

## Same-input normalization replay completed

The [controlled replay](../reports/native-cpu-same-input-norm-replay-v1.json)
passes all nine own-input reproduction checks: input normalization and Q/K head
normalization, each on batch, sequential, and oracle inputs. The extracted native
function exactly reproduces captured native outputs; the unchanged NumPy expression
exactly reproduces captured oracle outputs. FP32 normalization weights are also
checked against the original checkpoint's widened BF16 values.

On every captured input, supplying the installed NumPy FP32 sum to compiled
**direct division** reproduces the NumPy normalization output exactly. Neither
changing only the reduction nor changing only reciprocal multiplication suffices.
For identical embedding inputs, original native-versus-NumPy RMS is 1.446e-8;
native reduction with division reduces it to 7.713e-9, NumPy sum with reciprocal
to 1.184e-8, and NumPy sum with division to zero. Twenty of 63 rounded sums and
12 of 63 denominators differ between the original reductions.

However, normalizing native batch projections with the NumPy expression barely
changes their discrepancy from the *captured oracle post-normalization outputs*:

| Stage | Native norm on batch input | NumPy norm on that same batch input |
|---|---:|---:|
| Q post-normalization RMS versus captured oracle | 1.51048e-6 | 1.51547e-6 |
| K post-normalization RMS versus captured oracle | 1.35414e-5 | 1.35727e-5 |

These slight increases are retained. Alignment of the local normalization
operation does not remove the upstream projection discrepancy and is not a
network accuracy fix. Sequential K improves modestly, from 3.42955e-6 to
3.32723e-6, but the conclusion remains local and fixture-specific.

The next useful controlled test is **projection input attribution across all63
rows**: run the unchanged NumPy FP32 Q/K/V matvec on both native-normalized and
oracle-normalized hidden inputs, using the same verified matrix values. Require
its own oracle-input projections to exactly reproduce the captured oracle arrays.
Then distinguish the effect of changed normalized inputs from native-versus-NumPy
projection reduction on the same inputs. Retain all output rows and both native
batch/sequential paths. This extends the earlier target-row matched projection
experiment without changing the network or inventing a different acceptance
reference. An FP64 exact-product diagnostic, if later needed, remains explicitly
separate from the NumPy network oracle.

The installed NumPy wrapper confirms FP32 reduction followed by FP32 division;
this experiment uses its actual compiled sum. It does not prove a particular
hand-written pairwise tree matches that implementation. No new network rewrite
is justified by the local zero-error combination alone. Any later arithmetic
candidate still requires fresh63/64, the old28 positions including2048/3968,
and remaining fine-boundary gates at unchanged0.01/0.002/rtol0 limits before timing.
The [source/build bindings](../reports/native-cpu-same-input-norm-replay-v1-provenance.json)
include the executed standalone helper and preregistered diagnostic plan.
