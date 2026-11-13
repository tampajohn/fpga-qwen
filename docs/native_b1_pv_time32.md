# Ordinary B1 PV time32: rejected

The candidate extends the tested query-row kernel with a separate ordinary-B1 PV loop: a probability-division prepass followed by fixed 32-key/32-channel register tiles. It changes no persistent memory allocation. Source review preserved ascending per-output FMAs; compiled inspection and relative gates followed. The [decision inventory](../reports/native-cpu-queryrow-b1-pv-time32-decision.json) links source/build/assembly metadata and every completed gate.

All 100 repeated full-logit arrays passed: 56 original, 12 prefill time boundaries, 8 actual63/64, 24 explicit step boundaries (including 12 actualSTEP outputs at 31/32/33/63/64/65). API36, prefix28, suffix35 plus seven lifecycle cases, CLI5, seeded-stream controls, and frozen 9/18 quality suites passed relatively. Actual quality scores remain 4/9 and 10/18, exactly matching tuned llama; unchanged wrong answers remain failures. The inherited absolute 63/64 NumPy failures are not cleared.

The [standard report](../reports/native-cpu-queryrow-b1-pv-time32-vs-queryrow-standard-confirmation.json) and [analysis](../reports/native-cpu-queryrow-b1-pv-time32-vs-queryrow-standard-analysis.json) contain 20 alternating pairs with 32 output tokens. The [extended report](../reports/native-cpu-queryrow-b1-pv-time32-vs-queryrow-extended-confirmation.json) and [analysis](../reports/native-cpu-queryrow-b1-pv-time32-vs-queryrow-extended-analysis.json) contain 10 pairs with 128 outputs. Both use one warmup, decode12/prefill6, identical complete trajectories, and the **tested query-row prefill-only candidate** as reference. They are not a direct build19 or llama comparison.

Each cell is median paired speedup [95% paired bootstrap interval]; higher than 1 favors B1PV. Every pair/outlier remains archived.

| Prompt tokens | Decode | TTFT | Complete request |
|---:|---:|---:|---:|
| 85 | 0.9696 [0.8784, 1.0286] | 1.0161 [0.9801, 1.0451] | 0.9766 [0.9161, 1.0201] |
| 452 | 1.0134 [0.9348, 1.0569] | 0.9979 [0.9868, 1.0236] | 0.9987 [0.9735, 1.0520] |
| 1024 | 0.9022 [0.8495, 0.9584] | 0.9677 [0.9365, 1.0326] | 0.9450 [0.9104, 0.9694] |
| 2048 | 0.9667 [0.7565, 1.0368] | 1.0044 [0.9523, 1.0563] | 0.9944 [0.8692, 1.0401] |
| 3968 | 0.9441 [0.8770, 1.0068] | 0.9852 [0.9716, 1.0393] | 0.9796 [0.9591, 1.0393] |

B1PV is **rejected**: no metric establishes an incremental benefit, while the 1024-token decode and complete-request intervals favor the reference. All 2048/3968 intervals include 1, and the 20-pair short-context results establish no gain. These observations do not establish a universal slowdown. No samples are removed or substituted.

The [controller provenance record](../reports/native-cpu-queryrow-b1-pv-time32-controller-provenance.json) preserves a preparation mistake: an exact-gate shell filename was overwritten after its controller had launched. The executed version is archived separately with its SHA. The recorded difference is hash assertions and guard labels; model/compare arguments and runtime tools were unchanged. This is disclosed, not silently attributed to the later controller text.

The new B1 head-pair candidate is based on **plain query-row time32**, not this rejected B1PV change. Its evidence remains independent; no result transfers automatically. Live build19 is unchanged. Byte-preserved reproduction `.txt` helpers must be restored to their recorded local paths before guarded root execution.
