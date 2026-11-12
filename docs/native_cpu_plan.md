# Native CPU engine direction

The user-approved CPU-ENGINE-BRIEF.md supersedes the simulator optimization
plan. Target: Qwen3-0.6B on this Apple M5 Max, CPU-only, initially in the 16-bit
weight class against a tuned llama.cpp FP16 baseline. Primary metric is
single-stream decode; first-token latency at 452 prompt tokens is secondary.

`rtl/`, `sim/`, and the patched Verilator checkouts are frozen except for bug
fixes. Existing experimental code and artifacts are preserved; no pending
simulator experiment is a new release. Native implementation will live in
`engine/`, with existing tools, quality gates, server and browser reused.

## Phase 0

Keep llama.cpp source pinned to the clean local commit cb44fdb while comparing
build options, then tune decode and prefill thread counts separately. Build
candidates in `build/native-phase0/`; do not overwrite the live native build.
KleidiAI/LTO availability is not proof that a faster kernel ran. Check actual
runtime dispatch and model-buffer logs, alongside CPU-only build/runtime flags.

Use exploratory sweeps to select finalists, then at least ten alternating
paired measurements for performance claims. Save exact prompt/output token IDs,
respect EOS, disable prompt cache and speculation, and keep model loading and
warmup outside request timing. Measure 85- and 452-token prompts; report both
rather than selecting a winning context after the fact. The existing fixed
32-token workload remains a comparable starting point; longer sustained decode
can supplement it. Background OS activity remains a limitation and is recorded.

Profile separately from scoring. Sample stacks in the exact tuned binary; do
not equate worker CPU samples with critical-path wall time. Vocabulary and layer
projections can share symbols, so any callback-based attribution must report
its synchronization perturbation relative to callback-off execution.

## Phase 1 decision

John delegated remaining decisions with “use best judgement.” The working
budget is four hours of wall-clock implementation and evaluation, starting
when Phase 1 begins after Phase 0 establishes the target. Record the actual
start/deadline at that point. Apply the brief's stop rule: if the correct plain
engine is below 90% of tuned FP16 decode at expiry, stop and report the gap.
No Phase 1 implementation has started. Existing auto-push behavior continues.

Use an explicit numerical contract covering weights, activations, accumulation,
normalization and KV. BF16 and FP16 are not numerically interchangeable. Retain
token-identity gates and predeclared full-logit tolerances; diagnostic dense
logits remain available even if later vocabulary pruning is investigated.
A candidate-set optimization requires certified exclusion bounds and fallback,
not only successful recall on a finite corpus.

The worktree/branch inventory is in the local
`build/cpu-engine-worktree-inventory.txt`: 68 worktrees, 70 codex branches,
65 not ancestry-merged into main at the direction change. Cherry-picked work
may already be on main. Nothing was deleted.

## Initial baseline discovery

All four KleidiAI/LTO combinations compiled from the same clean cb44fdb source
with native CPU flags, Accelerate, OpenMP explicitly off, and GPU backends off.
Their build manifests are archived under `reports/native-phase0-*-build.json`.
The first FP16 build sweep completed two exploratory alternating rounds at
8 decode / 12 prefill threads; all configurations matched greedy IDs on the
85/452-token prompts and processed uncached prompts.

The [discovery report](../reports/native-phase0-f16-build-discovery.json) shows
no established universal winner: the untuned build had the highest observed
short decode median, and KleidiAI+LTO the highest observed long decode median.
Two observations are insufficient for a performance claim. Thread discovery,
numerical checks, profiling and ten-pair confirmation remain pending.

## Numerical contract before native implementation

Phase 0 calibration used the independent NumPy network with FP16-rounded matrix
weights, original FP32 one-dimensional normalization weights, and FP16-rounded
current/stored K/V widened for FP32 attention. Four teacher-forced positions at
contexts 1–4 had maximum absolute logit error 0.001997 or less and RMS error
0.000472 or less. Four predictions at contexts 452–455 had maximum absolute
error 0.004605 or less and RMS error 0.001069 or less. Every tested greedy ID
matched. Calibration is descriptive, not a retrospectively passed gate.

Before Phase 1 code exists, declare its numerical gate: maximum absolute error
<=0.01, RMS error <=0.002, relative tolerance zero, all full-vocabulary logits
finite, and greedy identity. These limits allow roughly twice the observed
baseline error at the target long context while retaining an absolute bound on
every logit. An independent short-chat trajectory will validate the gate on the
native baseline before applying it to the new engine. The existing nine-case
conversation suite remains a separate token-identity/regression requirement;
matching a wrong baseline answer does not establish answer correctness.

The baseline calibration reports are `native-phase0-f16-oracle-calibration.json`
and `native-phase0-f16-long-oracle-calibration.json` under `reports/`. The initial
four-position native build/callback comparison was bit-identical; its first
diagnostic client did not attach persistent pools, so its timings are excluded
from profiling conclusions. A corrected client attaches persistent CPU pools.

## Held-out gate and final baseline selection

The independently held-out short-chat trajectory passed the predeclared gate
at all eight positions (contexts 85–92): maximum absolute error 0.007021,
maximum position RMS error 0.001554, and zero greedy mismatches. See
[the gate report](../reports/native-phase0-f16-short-oracle-gate.json).
This establishes baseline compatibility with the chosen tolerance on these
positions; it is not yet a result for the unimplemented native engine.

The final two-round discovery remains exploratory and shows substantial
run-to-run scheduling variation, especially above twelve decode threads.
Ten-round confirmation compares KleidiAI+LTO at 10/6, 12/6, and 12/4
decode/prefill threads against the KleidiAI-off/LTO-off 8/6 control.
Normal priority and polling 50 are retained. A priority-1 probe failed preflight
with macOS permission denied; no elevated scheduling is claimed or required.
