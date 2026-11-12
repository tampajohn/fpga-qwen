# Greedy B2 verification protocol (experimental)

The CLI advertises `verify_greedy_b2: true` and `max_verified_draft: 1`.
`suffix_drafting: false` remains the broader browser-facing capability until
adapter integration and independent gates pass. No performance claim follows
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
the existing error contract. A future adapter must retain the canonical prefill
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
