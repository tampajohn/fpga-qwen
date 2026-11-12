# Greedy B2 verification protocol (experimental)

The CLI advertises `verify_greedy_b2: true` and `max_verified_draft: 1`.
`suffix_drafting: false` remains the CLI's broad capability label; the adapter
separately recognizes the narrowly scoped greedy B2 handshake. Requests opt in
through `max_draft: 1`; the default remains zero. No performance claim follows
from exposing this command. Chat sampling is explicitly unsupported.

`verify-greedy PENDING DRAFT` evaluates exactly two input positions. PENDING
has already been emitted but has not entered KV/history. DRAFT is one observed
ngram proposal. The caller must have at least two output-budget slots and two
KV slots available; otherwise use the ordinary `step` command. The CLI validates
both IDs, its greedy mode, capacity and consumed-history consistency.

The response contains `tokens` (including terminal EOS), `accepted` (0 or 1),
`committed_inputs` (count 1 or 2), resulting `position`, `verified_positions: 2`,
`finish_reason` (`"eos"` or null), and diagnostic `seconds`.

- First prediction differs from DRAFT: emit correction, commit PENDING only.
- First prediction matches non-EOS DRAFT: emit it and the second prediction;
  commit PENDING and DRAFT.
- First prediction is EOS: emit only EOS and commit PENDING, even if EOS matches
  DRAFT. `accepted` still records the match; it is independent of commit count.
- Second prediction is EOS: emit both predictions and commit both inputs.

The correction or bonus prediction remains unconsumed until the next round.
Commit restores the corresponding full logits and normalized hidden vector.
Only committed input IDs enter consumed history and sampler presence. The CLI
uses no RNG draws in greedy mode; any error resets model/history/presence using
the existing error contract. The adapter retains the canonical prefill
cache ceiling: verified/decoded KV cannot become canonical prefix-cache chunks.

`tools/check_native_suffix_cli.py` prepares real-model protocol checks against
ordinary sequential steps, including first rejection, all accepted, EOS at the
first/bonus output, exact committed logits/hidden state, malformed commands and
chat rejection. Its prompt must naturally reach EOS within 64 steps; these
explicit correctness proposals are not a production drafter or timing workload.
The adapter is responsible for output-budget/capacity dispatch gates and for
keeping suffix drafting disabled in chat until a separately verified sampler
implementation exists.

## Numerical API validation

The full direct-API matrix passed **36 prefix/commit cases**: prefixes
0, 1, 511, 512, 513 and 3966, committing one or two inputs, with 1, 12 and
64 decode workers (six prefill workers). Both verification rows matched two
ordinary sequential steps bit-for-bit across all 151,936 logits and all 1024
normalized hidden elements. Different subsequent input tokens checked that
commit-one rollback did not expose rejected speculative KV. Public base-state
snapshots remained unchanged while verification was pending.

The same gate checked capacity rejection, preservation of pending state after
validation errors, and abort through reset/truncate. Execution-failure reset
behavior was source reviewed; allocation/hardware failures were not injected.
These were predetermined synthetic token prefixes and fixed verification IDs,
not conversation fixtures. The result establishes exact API behavior on this
matrix, not suffix acceptance rates, browser correctness or speed. CLI/adapter
integration, EOS behavior and measured benefit require separate gates.

Evidence: [sanitized full B2 API gate](../reports/native-cpu-b2-api-gate.json).

## Candidate-15 CLI and adapter correctness

Candidate-15 passed the real CLI gate using a greeting prompt that naturally
reached EOS. The five cases covered first rejection, acceptance plus bonus,
first-position EOS matching/rejecting the proposal, and bonus EOS. Every
committed full-logit and hidden-vector state matched the ordinary sequential
path. Malformed commands and chat-mode verification were rejected with the
specified full reset.

The first attempt used a prompt that did not reach EOS within the checker's
64-step discovery cap. That run never reached the EOS-specific verification
cases; it is a **test-fixture limitation**, not evidence of an engine failure.
Both the incomplete attempt and the successful greeting-fixture result are
preserved.

Across 35 real-adapter fixtures, frozen build12, candidate15 with suffix off,
and candidate15 with greedy suffix on produced identical visible token IDs,
text and natural stop/output-cap outcomes. These included the fixed nine- and
eighteen-case suites plus repetitive and budget fixtures. The enabled path
made **63 proposals and accepted 32**. Those counts demonstrate exercised
acceptance/rejection paths on these fixtures; they are not a general acceptance
rate estimate or a performance result. Unscored incidental timing fields were
omitted from the published correctness archive.

Candidate15 is not the live release; build12 remains live. No suffix speedup
or default-enablement claim is made. Chat drafting remains unsupported.

Evidence: [CLI gate](../reports/native-cpu-build-15-suffix-cli.json),
[original fixture limitation](../reports/native-cpu-build-15-suffix-cli-fixture-limit.json),
and [35 adapter trajectories](../reports/native-cpu-build-15-suffix-trajectories.json).

The candidate15 ordinary path also matched frozen build12 bit-for-bit on
**32 full-vocabulary arrays**: two repeated captures of four tiny positions,
eight short-prompt positions, two 513-token boundary positions and two
3968-token-context positions. This independently checks the default path
against the prior binary, complementing the direct API gate that compares
verification with sequential steps inside the new core. It supports carrying
forward prior numerical evidence on these matched trajectories, without
claiming exhaustive equivalence.
See the [frozen-build exact gate](../reports/native-cpu-build-15-exact-gate.json).

## Isolated candidate15 browser check

The candidate15 browser test passed on localhost:8766, separately from the live
server. It confirmed capability-aware greedy controls, switching to chat
unchecking/disabling drafting, a real chat POST carrying `max_draft: 0`, and
returning to greedy without restoring the earlier checkbox selection. It also
passed canonical 512-token prefix reuse, focused backend-selector stability
during polling, visible benchmark blocking, cancellation/restart and backend
switching. No browser console errors were recorded.

**Every recorded browser request used `max_draft: 0`.** The checkbox was checked
and then switched to chat before submitting. This test validates mode controls
and safe request construction; it does not demonstrate browser-submitted
suffix-on inference. The separate 35-trajectory real-adapter gate exercised
suffix-on acceptance/rejection. A future browser suffix-on request remains a
separate coverage item. The isolated server was stopped after testing; this
result does not deploy candidate15 or change the live build12 release.

Evidence: [isolated browser report](../reports/native-cpu-build-15-browser-isolated.json).


## Paired suffix off/on outcome: keep drafting off by default

The first fixed repeat prompt, “Write the exact sequence red blue green twenty times, without commentary.”, returned **“Red, Blue, Green, Twenty.”** and made zero suffix proposals. The workload gate correctly stopped before scoring: **zero paired rounds** ran. This is preserved in the [failed preflight archive](../reports/native-cpu-build-15-suffix-preflight-failed.json), including its actual output. It provides no suffix-benefit timing evidence.

Before any new timed rounds, an explicit `--repeat-fixture counting` selected “Count from 1 to 40, separated by commas, without commentary.” This prompt was already present in the 35-case correctness suite. The original default and failed evidence were retained. The second fixture remained the complete frozen 85-token memory-bandwidth prompt. The [completed confirmation archive](../reports/native-cpu-build-15-suffix-confirmation-v2.json) contains ten alternating off/on pairs, two symmetric warmups per configuration/profile, the same candidate15 binary and 12/6 threads, fresh prefixes with reuse disabled, and a 64-token cap that honors natural EOS. Every warmup and measured request matched exact IDs, text and stop behavior. Both fixtures reached the cap in this run.

| Fixture | Median paired request-time speedup, off/on (95% interval) | Median paired decode-throughput ratio, on/off (95% interval) |
|---|---|---|
| Counting, 90 prompt tokens | 0.968107 (0.850626–1.112122) | 0.993300 (0.837174–1.156923) |
| Low-repetition prose, 85 prompt tokens | 0.971805 (0.848502–1.013266) | 0.971677 (0.827671–1.001238) |

Each interval is a deterministic paired bootstrap over its own ten rounds; no workloads are pooled. Ratios above 1 favor suffix drafting. **Neither workload establishes a request-time or decode improvement.** The intervals do not establish equivalence, either. Full precision ratios and every paired measurement remain in the compact archive; only repeated output arrays and unscored warmup timings were omitted.

Per 64-token counting request, drafting proposed **27** tokens and accepted **9**, with **18 rejected**. It evaluated 54 verification positions, committed 36 verified inputs, and left 18 computed verification positions uncommitted. Neural command count fell from 64 to 55, including prefill. The prose request proposed **1**, accepted **0**, and left one of two computed verification positions uncommitted; command count stayed 64. These work counters explain what the experiment exercised, but are not a direct allocation of elapsed time or a general acceptance estimate.

The observed paired TTFT ratios were 0.906694 (0.893014–0.957502) for counting and 0.962948 (0.942146–1.091532) for prose. B2 verification starts **after the first emitted token**. Therefore these TTFT differences are not established algorithmic B2 overhead; scheduling, address placement and background activity remain uncontrolled. Whole-request time is the primary end-to-end measure; decode-rate timing also has batching/EOS-tail limitations and is reported separately.

The evidence supports keeping suffix drafting **off by default**. Build12 remains the live release at this report's publication; candidate15 browser verification is a separate step and no deployment or speedup is claimed here.
