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
