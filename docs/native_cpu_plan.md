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
Phase 1 start/deadline are recorded in `engine/phase1-window.json`. Existing auto-push behavior continues.

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
The four-hour Phase 1 implementation clock had not started when these FP16 results were recorded.

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

## Phase 1 started

Start: 2026-11-12T05:25:15.063226+00:00; deadline: 2026-11-12T09:25:15.063226+00:00.
Four-hour wall-clock implementation/evaluation budget. The FP16 target is
ON/ON 12 decode / 6 prefill, with the frozen numerical and conversation gates.
Phase 0 is complete; constrained Q4 exclusions and profiling limits remain
explicit. No native engine performance result exists at this start point.

## First plain engine correctness evidence

Build 02 is original C++/NEON plus Accelerate with no llama/ggml neural runtime,
Torch or GPU dependency. It stores FP16 matrices and KV, uses FP32 activations
and normalization, and eagerly widens layer matrices for batched prefill
(approximately 1.76 GB additional cache, excluding mapped weights and KV).
The first build attempt failed because the direct Xcode compiler did not locate
C++ headers; using the system compiler shim produced the verified build.

All sixteen selected full-vocabulary positions passed the predeclared NumPy
gate: contexts 1–4, 85–92 and 452–455, maximum absolute error 0.008078 and
maximum RMS 0.001455, with all greedy IDs matching. Each captured trajectory
was repeated with identical logits. See the build-02 tiny/short/long gate
reports in `reports/`. These are finite-position checks, not universal quality.

Actual process tests passed seeded sampling repeatability, reset, in-flight
cancellation and process reaping, clean restart, context overflow and invalid
token handling. Existing nine-case conversation and matched speed tests remain
pending. The browser backend is registered as experimental and unavailable
until explicit binary promotion; the live server has not been restarted.

## Original engine build-03 confirmation

The first correct original engine, build-02, passed the declared NumPy gates at
16 positions and the nine-case exact-trajectory quality gate (including terminal
EOS). Its two-round discovery observed 59.93/52.29 decode tokens/s against
141.95/134.20 for tuned llama. The initial quality harness omitted terminal EOS
from native returned IDs; quality-v2 fixes that transport representation without
normalizing baseline output or relaxing the identity gate.

Build-03 changes worker scheduling/completion and scratch-buffer reuse. All 32
full-logit arrays from repeated tiny, short, and long captures matched build-02
bit-for-bit. The nine-case suite also passed exact IDs, EOS, text, and grading;
both engines scored 4/9, with no new regressions and no answer-quality gain.

Ten alternating paired rounds compared build-03 against tuned KleidiAI+LTO FP16,
with both at twelve decode/six prefill threads. At 85/452 prompt tokens, native
median decode was 125.74/101.93 tokens/s versus llama's 135.27/131.69. Median
within-round native/llama ratios were **0.838x** (95% paired-bootstrap interval
0.793–0.957) short and **0.754x** (0.645–0.834) long. These ratios are not the
quotients of independent medians. The engine remains slower and has not met
the 90% target. No live backend is promoted.

Scores include differing subprocess/HTTP transports, while the native engine
also keeps widened FP32 prefill weights. Background scheduling remains
uncontrolled. Two-round discovery results are retained as exploratory evidence,
not substituted for confirmation or used for a causal historical speedup claim.
Build-04's 32-array exact check has passed; its additional evaluation is ongoing.
Build-05 SIMD source has no performance result reported here.

Evidence: [build-02 quality-v2](../reports/native-cpu-build-02-quality-v2.json),
[build-02 discovery](../reports/native-cpu-build-02-discovery.json),
[build-03 exact gate](../reports/native-cpu-build-03-exact-gate-v2.json),
[build-03 quality](../reports/native-cpu-build-03-quality.json),
[build-03 discovery](../reports/native-cpu-build-03-discovery.json),
[build-03 confirmation](../reports/native-cpu-build-03-confirmation.json), and
[build-03 paired uncertainty](../reports/native-cpu-build-03-confirmation-analysis.json).
Sanitized archives retain raw-report hashes and anonymized host CPU percentages.

## First actual browser validation

Build 06 passed the real Chrome browser suite against an isolated localhost:8766
server: multi-turn recall, native controls, zero prefix reuse, benchmark polling
and server409, cancellation followed by a fresh request, and native/llama/native
backend switching. No browser JavaScript errors occurred. The live localhost:8765
server and stable native binary path were unchanged. See
[browser evidence](../reports/native-cpu-build-06-browser.json).

## Build-08 competitive point estimates, no established decode win

Build-07 (vector tanh) and build-08 (vector exp) each passed the unchanged
full-logit limits at all 16 tiny/short/long positions and the exact nine-case
conversation gate. Both native and reference remain 4/9 correct. Vector-library
arithmetic is checked numerically; these changes do not claim bit identity.
Build-07 discovery remains exploratory.

Build-08 ten-pair confirmation observed native median decode 148.35/136.93
tokens/s against tuned llama's 141.35/130.55 at 85/452 prompt tokens. Median
paired ratios were **1.014x [0.812, 1.114]** and **1.055x [0.949, 1.196]**
(95% paired-bootstrap intervals). Both include 1: no established decode win.
These are within-round paired ratios, not ratios of separate medians. Native
TTFT medians were 0.100/0.603 seconds versus 0.160/0.559: short-context latency
improved, but long-context latency remains behind. Retain transport, widened
prefill-cache memory, and uncontrolled OS scheduling caveats from prior runs.

The synthetic 513-token ChatProtocol fixture crosses the native 512-token
prefill chunk boundary. Both tuned llama and build-08 passed the unchanged
0.01 maximum absolute / 0.002 RMS limits at positions 513 and 514, with all
greedy predictions matching NumPy. Baseline diagnostic capture used a physical
batch size of 1024 to fit the prompt; it is not a timing-protocol change.
The fixture is structural coverage, not evidence of real-user answer quality.
511/512-token boundary results are not published in this update.

Build-06's separate actual-browser smoke does not validate build-08 in-browser.
No live backend has been promoted. Candidate-09 lossless BF16 auxiliary-cache
source is pending evaluation, with no result claimed here.

Evidence: build-07 [tiny](../reports/native-cpu-build-07-tiny-gate.json),
[short](../reports/native-cpu-build-07-short-gate.json),
[long](../reports/native-cpu-build-07-long-gate.json),
[quality](../reports/native-cpu-build-07-quality.json),
[discovery](../reports/native-cpu-build-07-discovery.json);
build-08 [tiny](../reports/native-cpu-build-08-tiny-gate.json),
[short](../reports/native-cpu-build-08-short-gate.json),
[long](../reports/native-cpu-build-08-long-gate.json),
[quality](../reports/native-cpu-build-08-quality.json),
[confirmation](../reports/native-cpu-build-08-confirmation.json),
[paired uncertainty](../reports/native-cpu-build-08-confirmation-analysis.json),
and [boundary 513/514 gate](../reports/native-cpu-boundary-513-oracle-gate.json).

## Prefill boundary coverage completed

The 511- and 512-token synthetic fixtures also passed unchanged NumPy tolerances
for both tuned llama and build-08, at contexts 511/512 and 512/513 respectively.
Together with the 513/514 result this covers the two sides of the 512-token
prefill split, including a carried chunk and subsequent decode. Each engine
used identical prescribed continuations; all tested greedy predictions matched.
The baseline diagnostic physical batch remains 1024, with no timing claim or
change to the scored workload. This finite structural coverage does not prove
all context lengths or real-user quality.

See [511 gate](../reports/native-cpu-boundary-511-oracle-gate.json),
[512 gate](../reports/native-cpu-boundary-512-oracle-gate.json),
[513 gate](../reports/native-cpu-boundary-513-oracle-gate.json), and
[fixture metadata, messages, IDs and raw hashes](../reports/native-cpu-boundary-fixtures.json).

## Candidate-09 auxiliary cache: correctness and memory cost

Candidate-09 constructs a BF16 auxiliary representation only when every value
in a loaded FP16 matrix widens to exactly the same FP32 bit pattern as the BF16
representation. It does not round weights again; any inexact value disables the
cache for that entire tensor. The canonical model's actual startup reported
**1,191,968,768 additional cache bytes** (about 1.19 GB decimal), beyond mapped
FP16 weights, widened FP32 prefill weights, KV, and other working memory. This
extra memory is a cost, not a compression of the existing resident model.

The exact-logit check and nine-case conversation check passed. The explicit
fallback experiment cloned the pack and changed only the embedding's first
FP16 word to `0x3c01` (1.0009765625), a value not exactly representable in BF16.
The entire 311,164,928-byte embedding auxiliary cache disappeared, leaving
880,803,840 bytes. Build-08 and candidate-09 returned bit-identical full logits
and greedy IDs on the changed fixture. Row-zero's first logit changed from
5.9522008896 to 13.1209831238 versus the normal capture, demonstrating that the
modified weight was exercised. The canonical pack hash remained unchanged.
This is fallback correctness evidence, not quality evidence for an altered model.

Candidate-09 is **not adopted**. A direct paired build-08/build-09 comparison is
pending; no speed advantage or justification for the extra cache is claimed.
Candidate-10 scheduling work starts from the build-08 arithmetic/cache baseline
and reports zero auxiliary BF16 cache bytes. The live backend remains unchanged.

Evidence: [exact logits](../reports/native-cpu-build-09-exact-gate.json),
[conversation quality](../reports/native-cpu-build-09-quality.json), and
[real changed-weight fallback](../reports/native-cpu-build-09-fallback.json).

## Candidate-09 rejected after direct comparison

Ten alternating paired rounds compared build-09 directly with build-08 using
the same original-engine transport, FP16 pack, twelve decode/six prefill threads,
and three warmups per context per visit. Exact output-ID preflight remained
required. The auxiliary cache gave a short-context median paired decode ratio
of **0.92553x** (95% paired-bootstrap interval **0.72308–0.98871**), indicating
a regression. The long-context ratio was **1.00727x** (**0.84068–1.62179**),
with no established gain. TTFT was approximately unchanged.

Reject the auxiliary cache for the current candidate: its measured additional
1,191,968,768 bytes are not justified by these results. Correctness and real
fallback evidence remain valid, but do not make the optimization worthwhile.
This direct comparison evaluates the cache change; it is not a performance win
against tuned llama. Candidate-10 retains build-08's FP16 path and zero auxiliary
BF16 cache, adding only balanced prefill attention work assignment. Its evaluation
is ongoing, and the live backend remains unchanged.

Evidence: [direct paired report](../reports/native-cpu-build-09-versus08-confirmation.json)
and [paired uncertainty](../reports/native-cpu-build-09-versus08-analysis.json).

## Build-10 confirmation: competitive decode and lower first-token latency

Build-10 uses the proven build-08 FP16 arithmetic path, with causal-attention
work assigned by cumulative context length during batched prefill. It retains
zero auxiliary BF16 cache bytes. Thirty alternating paired rounds compared it
with the tuned KleidiAI-ON/LTO-ON llama baseline, both configured for twelve
decode and six prefill workers, using identical uncached 85/452-token prompts
and 32 greedy output tokens. Each visit warmed both contexts three times.
The original engine's Accelerate prefill implementation controls its own BLAS
threads; the six-worker setting describes its custom pool.

| Context | Median paired decode ratio | 95% paired-bootstrap interval | Median paired TTFT speedup |
| --- | ---: | ---: | ---: |
| 85 tokens | 1.02835x | 0.95439–1.07877 | 1.64681x |
| 452 tokens | 0.98020x | 0.90520–1.02219 | 1.14327x |

These results meet the within-10% decode target on the two tested workloads,
including the lower bounds of the reported intervals. They **do not establish
a decode speed advantage**: both decode intervals include 1.0. The TTFT
speedup intervals were 1.56292–1.91121 short and 1.09918–1.24135 long. Median
native throughput was 135.32/129.50 tokens/s versus 134.42/129.18 for llama;
these marginal medians are descriptive, while the paired ratios above account
for visit pairing. Native median TTFT was 0.124/0.532 seconds versus
0.216/0.628 seconds. Scheduling variation remains substantial, and this is a
comparison of complete CPU inference paths and their transports on this host,
not a universal kernel or hardware speed claim.

Correctness evidence is deliberately separate. Build-10 matched build-08
bit-for-bit across 44 complete FP32 vocabulary arrays: 22 evaluated positions
with repeated captures, including 511/512/513-token boundary prompts and their
continuations. Those build-08 captures passed the independent NumPy gates at
the unchanged declared limits. The build-10 numerical conclusion is therefore
transitive on those exact trajectories; it is not an exhaustive model proof.
The nine-case conversation suite matched all reference output IDs, stopping
behavior and grades; both models answered 4/9 correctly. This establishes no
new regression on the suite, not strong general answer quality.

The runtime robustness gate also passed: worker counts 1/64 produced identical
full logits across repeated resets and 0/2/50-ms idle intervals; parked workers
shut down cleanly; seven malformed header/index fixtures were rejected without
changing canonical weights. The interface gate passed seeded-chat repeatability,
in-flight cancellation/restart, small-context capacity and invalid-ID handling.
These checks do not establish numerical accuracy at the full 4096-token limit.

Evidence: [immutable build manifest](../reports/native-cpu-build-10-manifest.json),
[exact-array gate](../reports/native-cpu-build-10-exact-gate.json),
[conversation gate](../reports/native-cpu-build-10-quality.json),
[30-round confirmation](../reports/native-cpu-build-10-confirmation.json),
[paired uncertainty](../reports/native-cpu-build-10-confirmation-analysis.json),
[runtime robustness](../reports/native-cpu-build-10-robustness.json), and
[interface checks](../reports/native-cpu-build-10-interface.json).
Raw report hashes are retained in each sanitized archive's provenance; embedded
artifact hashes continue to identify the original immutable inputs.


## Local build-10 release

The stable native executable now resolves to the immutable build-10 binary.
The localhost:8765 browser runs `tools/chat_server.py --native-only --port 8765`:
it needs no simulator binary or imports and selects the latest native build
first. Available llama.cpp comparison options remain selectable. Full-context
processing is explicit; prefix reuse and suffix drafting remain unsupported.

The [live browser report](../reports/native-cpu-build-10-browser-live.json)
records successful default selection after reload, two-turn recall, cancellation
and fresh generation, native/llama/native switching, and the polled benchmark
banner with server-side HTTP409 rejection while locked. This is interface
validation, not an additional speed measurement. The benchmark lock is released
and the live server is ready for user testing.


## Follow-on work: conversation latency and specialization

The user approved the next path after build-10: safe prefix reuse, 1K/2K/near-4K
and sustained-decode measurements, profiling this original engine, certified
vocabulary-head candidate pruning, and then measured suffix drafting. Expand
conversation-quality coverage in parallel. These are separate deliverables;
prefix caching alone does not complete this follow-on goal.

Candidate build-11 adds KV rewind and restores sampler presence from retained
consumed IDs. Browser reuse is session-specific and limited to completed
canonical 512-token prefill chunks. Partial tails and decoded-token KV are not
reused; each request recomputes a nonempty tail. This preserves fresh-prefill
batch shapes. Old binaries continue with fresh prefill via capability detection.

The candidate preserved 44 baseline full-vocabulary arrays exactly. Its new
cache checker passed 28 cached/fresh transitions across boundary lengths,
identical/edited prompts, session changes, greedy and seeded sampling, plus
cancellation/restart and rewind-error checks. Every observed prediction has an
exact full-logit comparison, not only an answer-text check. These synthetic
fixtures do not establish broad answer quality or a timing benefit. Performance
and browser release checks remain pending; build-10 remains the stable binary.

The extended matched runner uses complete tokenized messages of exactly 1024,
2048, and 3968 tokens, with configurable generation up to 128 tokens. It retains
natural EOS, identical-output preflight, fresh prompts, CPU-only settings and
alternating pairs. Cached-versus-fresh measurements use a separate runner so
cache savings cannot be presented as a dense-engine win against uncached llama.

## Build-11 prefix-reuse correctness candidate

Build-11 adds validated rewind and consumed-token sampler-history restoration.
The browser adapter reuses only complete 512-token canonical prefill blocks from
the same session. Reuse is bounded by the actual common consumed prefix and
retains at least one prompt token for recomputation. Decoded KV and partial tails
are not eligible. Cancellation, errors, restart, and a new session invalidate
reuse metadata; unsupported older binaries continue with fresh prefill.

The candidate passed 44 full-logit array comparisons with exact predicted and
consumed trajectories on the supplied repeated captures. The unchanged nine-case
suite also passed exact tokens/stops/text/grading against tuned llama; both
engines still scored 4/9. This demonstrates implementation compatibility on the
tested fixtures, not improved answer quality.

The real prefix checker passed all 28 synthetic cases across boundary lengths,
identical/edited prompts, and session changes, with greedy and seed-42 sampled
chat. Cached/fresh predictions and checked full logits matched exactly.
Cancellation/reap/restart and direct CLI rewind checks passed separately. These
results support the restricted canonical-prefix policy; they are neither a
browser end-to-end result for build-11 nor a measured prefix speedup.
Build-11 remains a candidate, with no performance or live-promotion claim here.

The initial extended counting prompt failed fixed-128-token preflight because
it naturally ended after 51 returned tokens. There were zero scored visits.
This failed report is retained; forcing past EOS or silently shortening it would
change the protocol. A revised tutorial prompt is a separate extended workload
and must pass its own identity/EOS preflight before scoring. Its ongoing results
are not published in this update.

Evidence: [build manifest](../reports/native-cpu-build-11-build.json),
[44-array exact gate](../reports/native-cpu-build-11-exact-gate.json),
[nine-case quality](../reports/native-cpu-build-11-quality.json),
[28-case prefix and lifecycle checks](../reports/native-cpu-build-11-prefix-check.json),
and [failed counting preflight](../reports/native-cpu-build-11-extended-counting-preflight-failed.json).

## Extended tutorial discovery completed

The revised tutorial workload passed fixed-128-token exact-ID/EOS preflight at
1024, 2048, and 3968 prompt tokens, followed by two alternating exploratory
rounds. Full prepared messages and token IDs are archived separately. The failed
counting preflight remains retained; it was not silently converted into a pass.

Observed medians, for planning only:

| Prompt tokens | Native decode tokens/s | Llama decode tokens/s | Native TTFT seconds | Llama TTFT seconds | Native request seconds | Llama request seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 115.81 | 88.28 | 1.445 | 1.494 | 2.542 | 2.935 |
| 2048 | 90.45 | 65.05 | 4.153 | 3.576 | 5.559 | 5.768 |
| 3968 | 60.85 | 57.08 | 13.660 | 8.612 | 15.752 | 10.837 |

Two pairs do not establish statistical significance or a performance win.
These observations motivate investigating longer-context prefill: at 3968
prompt tokens the native request was slower despite a somewhat higher observed
decode median. They are not a substitute for a predeclared paired confirmation
or independent numerical coverage of every position in these long prompts.
No new deployment or browser claim follows from this discovery.

Evidence: [exploratory report](../reports/native-cpu-build-11-extended-tutorial-discovery.json)
and [immutable prompt IDs/messages](../reports/native-cpu-build-11-extended-tutorial-fixtures.json).

## Extended conversation probes and long-context profile

The separate predetermined 18-case suite completed with exact native/llama
greedy trajectories and stopping behavior, no new case/check regressions, and
**10/18 strict rubric passes for each engine**. This does not change the frozen
nine-case suite or establish broad model accuracy. All three reported-loop
recovery probes and all three recall/quote probes passed these fixtures.

Eight strict failures were retained. Latest-question focus returned “Empty.”
when asked for the opposite of empty; multi-step arithmetic returned 21 instead
of 13. Six other failures had the requested answer but violated exact-output
instructions: corrected destination/owner added punctuation (“York.”, “Eli.”),
two arithmetic responses added working/explanation, and two missing-evidence
responses added explanation or punctuation instead of exactly “unknown”. This
is descriptive diagnosis, not post-hoc regrading: the reported score stays
10/18, and baseline agreement does not turn any failure into a pass.

A separate warmed build-11 macOS `sample` diagnostic at a 3968-token prompt
recorded 60,628 top-of-stack observations in the chunk attention worker thunk,
compared with 654 in the ordinary projection worker thunk and 1,492 in VVEXPF.
Waiting dominated the aggregate: `__psynch_cvwait` had 181,072 observations.
These are counts across sampled threads, including sleeping workers, **not CPU
percentages or wall-time fractions**. Aggregate samples lack timestamps for
separating prefill from decode, and shared projection symbols do not separate
vocabulary from layer work. The profile motivates investigating attention; it
does not quantify the critical-path payoff of an optimization. Attach/overlap
start was not independently confirmed, and the raw profile remains local.

Evidence: [extended conversation results](../reports/native-cpu-build-11-extended-quality.json)
and [sanitized 3968-token profile summary](../reports/native-cpu-build-11-profile3968-summary.json).

### Candidate 12: paired GQA prefill and measured prefix reuse

Candidate 12 shares K/V conversion between the two query heads belonging to each KV head during batched prefill. Its [immutable build record](../reports/native-cpu-build-12-build.json), [56-array exact comparison](../reports/native-cpu-build-12-exact-gate.json), [nine-case quality suite](../reports/native-cpu-build-12-quality.json), and [28-case prefix checker](../reports/native-cpu-build-12-prefix-check.json) are archived. The compared full-logit arrays match exactly; quality trajectories match the tuned llama baseline, with both scoring 4/9 under the unchanged rubric. Prefix checks include greedy and seeded chat, edits, session changes, cancellation and CLI behavior. These are scoped correctness results, not a general model-quality claim.

The [two-pair discovery against build 11](../reports/native-cpu-build-12-vs11-extended-discovery.json) remains exploratory. It does not establish a performance improvement. Separately, the [ten-pair cache-on/off confirmation](../reports/native-cpu-build-12-prefix-confirmation.json) measures the same build 12 binary on primed prompts with an edited tail, with priming outside the measured request. Output IDs and stop behavior match between configurations. The cache reuses only completed canonical 512-token prefill blocks and recomputes the remaining tail.

| Prompt tokens | Median TTFT fresh → reused (seconds) | Median paired TTFT speedup | 95% paired bootstrap interval |
|---|---|---|---|
| 1,024 | 1.335 → 0.803 | 1.632× | 1.575–1.730× |
| 2,048 | 3.728 → 1.397 | 2.672× | 2.557–2.819× |
| 3,968 | 11.719 → 1.974 | 5.744× | 5.512–6.512× |

These paired ratios are calculated per round, not from ratios of independent medians. All three decode-speed confidence intervals include 1, so there is no established decode improvement from reuse. The TTFT finding applies to this primed, edited-tail workload; it does not describe a fresh prompt or arbitrary conversation reuse. The browser benchmark guard excludes chat contention, but unrelated host activity was not controlled. Standard candidate-12 versus llama confirmation and browser promotion are separate steps; this entry does not claim deployment.

The [short-context build-11 profiling summary](../reports/native-cpu-build-11-profile85-summary.json) is also archived: 85 prompt tokens, 32-token requests, 48 fresh requests, 12 decode/6 prefill threads. Its 239,921 thread-stack observations include 134,510 waiting observations (`__psynch_cvwait`) and 27,503 ordinary projection-worker observations. These counts include sleeping threads and are not CPU percentages or wall-time attribution. The aggregate profile has no phase filter and does not isolate vocabulary-head projection from other projections; attach-start confirmation is also limited as recorded in the report.


### Build 12 local release and uncached confirmation

At the user's explicit request, the local stable executable now points to build 12 and the native-only localhost browser service has been restarted with it. Build 12 enables canonical 512-token prefix reuse and paired-query K/V conversion during batched prefill. The immutable build-10 artifact remains preserved. [Live browser verification](../reports/native-cpu-build-12-browser-live.json) passed default/reload, a focused dropdown across repeated status polls, changed-question replies with prefix reuse, cancellation/new chat, native→llama→native switching, and the polled benchmark banner with generation rejected while guarded. No heavy benchmark work runs during the user's testing window.

The completed [standard ten-pair confirmation](../reports/native-cpu-build-12-standard-confirmation.json) and [paired analysis](../reports/native-cpu-build-12-standard-confirmation-analysis.json) are preserved, including their unfavorable result. For uncached 85/452-token prompts, median paired native/llama decode ratios were 0.687 and 0.772, with 95% bootstrap intervals 0.487–0.914 and 0.658–0.923. This run does not establish a fresh-request performance improvement. The user reported heavy concurrent machine activity; its contribution is unresolved. The browser guard excludes chat contention but does not isolate the machine, so neither background activity nor a code change has been established as the cause. These observations must not be replaced by build-10's earlier competitive 30-pair result, which remains historical evidence for that older binary.

The separate ten-pair cache confirmation above supports TTFT improvement only for its primed edited-tail workloads; its decode intervals all include 1. Deployment makes the new functionality available for testing and does not turn that scoped cache benefit into a general inference-speed claim.


## Remaining acceptance work after build-12 exposure

Build-12 is exposed at the user's request and passed the live browser/cache
checks. Heavy runs are idle while the user tests it. The user reported other
substantial machine activity during the slower standard comparison; that run
is retained, and its cause remains unresolved. A direct immutable build-10 vs
build-12 comparison is still required before attributing the difference to code.

The head-only SDOT prototype has independent source review, including its
exact-rescore threshold, tie handling, full FPCR guard, and explicit FP32-range
enclosure. It has not yet been compiled or timed. Its driver validates all eight
saved short-context snapshots before timing predeclared steps 0/3/7. Offline
snapshot bounds alone do not establish runtime pruning safety or speed.

Experimental BLAS-attention build-14 remains preserved locally, but its source
change was reverted from the default engine pending unchanged NumPy and quality
gates. No performance claim is made for that experiment.

Suffix work is an unused two-position verification API with a separate exact
sequential-step checker. Shared weight loads must preserve each position's B1
FMA order and separate vector nonlinear calls. Pending speculative KV must not
become committed history or canonical prefix-cache state. There is no browser
suffix capability or default enablement yet. Numerical, rollback/EOS, browser,
and paired latency/acceptance gates remain required.
