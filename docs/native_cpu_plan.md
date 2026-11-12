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

## FP16 confirmation result

Ten alternating paired rounds completed, with uncached identical 85/452-token
prompts and 32 identical greedy output tokens per request. The primary target
configuration is KleidiAI ON, LTO ON, twelve decode threads, six prefill threads,
normal priority and polling 50. It produced median decode rates of 144.01 and
135.88 tokens/s, with median first-token times 0.215 and 0.588 seconds.
The ten-thread candidate was close at 142.49/135.38 tokens/s; these observations
do not establish a meaningful twelve-versus-ten-thread advantage.

Against the OFF/OFF 8/6 control, the primary configuration's median paired
decode speedups were 1.153x short (95% paired-bootstrap interval 1.056–1.423)
and 1.122x long (1.048–1.240). These compare complete configurations, not the
isolated effect of KleidiAI or LTO. Background scheduling variation remains
substantial. Future engine comparisons must rerun this baseline interleaved
with the candidate, rather than compare against these historical medians alone.

All four configurations matched exact output IDs, stop behavior, text and
grades on the existing nine-case conversation suite. Each scored 4/9 correct;
this is a no-regression result, not a claim of strong model answer quality.
Numerical checks cover the declared held-out positions, not every suite logit.

Evidence: [confirmation](../reports/native-phase0-f16-confirmation.json),
[paired uncertainty](../reports/native-phase0-f16-confirmation-analysis.json),
and [conversation gate](../reports/native-phase0-f16-confirmation-quality.json).
Phase 0 profiling and separate FP32/Q4 measurements remain in progress.
The four-hour Phase 1 implementation clock has not started.

## Q4 discovery boundary

The initial eight-configuration Q4_K_M discovery stopped at the exact output-ID
preflight gate, before any scored visits. All four KleidiAI-OFF configurations
(LTO OFF/ON, six/twelve decode threads, six prefill threads) matched on both
85/452-token prompts. The first KleidiAI-ON/LTO-OFF 6/6 configuration failed on
the **85-token short prompt**, not the long prompt. Its failed output IDs were
not persisted, so the first divergent token is unknown. The
[sanitized failure report](../reports/native-phase0-q4-discovery-failed.json)
preserves this negative result; startup/warmup timings are not scored evidence.
No speed or model-quality conclusion follows from this failed discovery.

The pinned local llama.cpp source provides a plausible arithmetic explanation,
not proof of the offending operation: `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`
accepts Q4_K and Q6_K matrix multiplications, and its Q6_K repacking path
dequantizes weights then requantizes each row to symmetric int8 using
`max_abs / 127`. The generic CPU Q6_K path instead uses
`ggml_vec_dot_q6_K_q8_K` (`ggml/src/ggml-cpu/ggml-cpu.c`). Identical GGUF storage
therefore does not guarantee identical effective arithmetic across these
backends. Separate fixed-prefix logit diagnostics would be needed to attribute
the divergence; changed greedy tokens alone do not establish worse answers.

The next constrained Q4 confirmation keeps the exact-ID gate unchanged and
compares the four KleidiAI-OFF configurations above over ten paired rounds,
with FP16 KV, normal priority, polling 50, and uncached identical prompts.
KleidiAI-ON is excluded from this comparison. The scope is a matched
KleidiAI-OFF Q4 baseline, not the globally fastest Q4 backend, nor equivalence
between Q4 and FP16 model quality. Confirmation and conversation-quality results
must be reported separately from this failed discovery.

## FP16 server CPU sampling diagnostic

Time Profiler attached to the warmed, exact ON/ON 12/6 FP16 server for a
25.95-second trace during thirty uncached 452-token requests, each generating
32 tokens. The export contains 186,335 running-thread samples from the target
PID; 29 complete decode windows fit inside the recording. Filtering by recorded
HTTP first-token/last-token windows and trimming two milliseconds at each edge
leaves 86,958 decode samples, with no missing user stacks.

| Decode stack category | CPU sample-weight share |
| --- | ---: |
| Matrix multiplication / packing path | 62.03% |
| Unresolved graph-worker loop | 24.11% |
| Attention / softmax / RoPE | 13.21% |
| Other unclassified | 0.39% |
| Other neural / tensor operations | 0.19% |
| Sampling | 0.07% |

These are sampled CPU-work shares across threads, **not fractions of request
wall time**. Shared matrix symbols do not distinguish vocabulary projection
from layer projections; attention ancestors are classified separately when
visible. Worker-loop leaf frames may contain inlined work, polling or
coordination, so their share is not a measured removable overhead.

The trimmed prefill interval contains 96,493 samples: 65.05% unresolved worker
loop, 15.25% attention/softmax/RoPE and 13.80% matrix/packing. It has 188 missing
user stacks (0.195%). The FP16-to-FP32 conversion leaf alone accounts for
3.82% of prefill samples. Its observed 3,625 unique operating-system thread IDs
are **not simultaneous workers** (decode observes thirty). The BLAS backend
source launches asynchronous weight-conversion tasks, a possible explanation
for churn that this analysis has not yet established by grouping TIDs/stacks.

Clock alignment uses the trace start timestamp and request UTC/performance-clock
anchors; HTTP boundaries and millisecond trace-start precision limit phase
accuracy. The report retains untrimmed results for comparison. This single
profile is diagnostic evidence, not another scored throughput confirmation.
See the [sanitized sample report](../reports/native-phase0-f16-profile-samples.json)
and [reusable parser](../tools/analyze_native_profile_samples.py). Raw profiler
artifacts stay local because their TOC can contain inherited environment values.

## Vocabulary-head boundary diagnostic

Ten alternating callback-off/on pairs used the persistent-pool public-API
client at twelve decode / six prefill threads, 452 prompt tokens and 32 greedy
outputs. All runs kept identical output IDs. The measured final-normalization
to-output interval was a median 17.51% of instrumented decode time (range
16.14–18.35%). It includes dispatch/synchronization around the vocabulary
projection; it is not a pure kernel duration or a direct server wall-time share.

Callback-on/off decode duration ratios had median 0.958 and range 0.859–1.310.
That variation prevents treating callback overhead as a known constant. The
measurement suggests prioritizing the whole projection pipeline and persistent
worker execution over vocabulary-only approximations; it does not certify a
speedup ceiling. Evidence: [paired diagnostic](../reports/native-phase0-f16-head-diagnostic.json).

## Secondary FP32 and constrained Q4 confirmation

Separate ten-round confirmations completed for FP32 and Q4_K_M, with identical
uncached prompt IDs and 32 greedy output IDs within each precision. These are
secondary baselines, not evidence that quantization preserves FP16 quality.
Normal priority, polling 50, FP16 KV, and six prefill threads apply throughout.

The FP32 primary decode target is KleidiAI OFF / LTO OFF, six decode threads:
median decode was 87.41/81.88 tokens/s at 85/452 prompt tokens. Eight and twelve
decode threads did not establish a decode advantage; their paired intervals
include 1. KleidiAI ON / LTO ON at six decode threads is a separate latency
option: median first-token time was 0.0536/0.2902 seconds versus
0.1795/0.4765 seconds for the decode target. Its paired TTFT speedups were
3.413x (95% interval 3.096–3.844) and 1.601x (1.486–1.747), while long-context
decode slowed to 75.76 tokens/s, a paired ratio of 0.923x (0.894–0.994).
There is no universal FP32 winner across decode and request latency.

The constrained Q4 target is KleidiAI OFF / LTO ON, twelve decode threads:
median decode was 269.57/230.78 tokens/s, with first-token times 0.0909/0.4941
seconds. Relative to the OFF/OFF six-thread control, paired decode speedups
were 1.186x (1.075–1.221) and 1.208x (1.164–1.273). The OFF/OFF twelve-thread
configuration had the highest observed short-context median, 278.10 tokens/s,
but its paired improvement interval against the control included 1. These
results do not establish that it beats the selected twelve-thread LTO build;
we retain one target rather than infer a context threshold from those medians.
KleidiAI-ON remains excluded following the identity failure documented above.
This is not a globally best-Q4 claim.

Both precision-specific conversation checks completed and passed: all four
configurations in each group matched exact IDs, stopping behavior, text, and
grades on all nine cases, with no new regressions. Every configuration scored
4/9 correct. Matching these baseline errors is not a quality improvement, and
these checks establish neither cross-precision identity nor a Q4 full-logit
oracle pass. Background scheduling variation and the limited workload remain
material limits; paired bootstrap intervals are not simultaneous intervals
corrected for selecting among configurations.

Evidence: [FP32 thread discovery](../reports/native-phase0-f32-thread-discovery.json),
[FP32 confirmation](../reports/native-phase0-f32-confirmation.json),
[FP32 uncertainty](../reports/native-phase0-f32-confirmation-analysis.json),
[FP32 quality](../reports/native-phase0-f32-confirmation-quality.json),
[Q4 confirmation](../reports/native-phase0-q4-off-confirmation.json),
[Q4 uncertainty](../reports/native-phase0-q4-off-confirmation-analysis.json), and
[Q4 quality](../reports/native-phase0-q4-off-confirmation-quality.json).
Archives preserve raw-report SHA256 provenance and anonymized host CPU
percentages. Analysis reports retain their conservative automatic claim flags;
the interpretation above is scoped to the separately reviewed quality results.
