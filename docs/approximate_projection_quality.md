# Held-out distribution probe for approximate RTL projections

`tools/evaluate_projection_quality.py` compares the current exact SV executable
with a candidate using the same original checkpoint and authored reference
continuations. It tokenizes, transports tokens and scores returned logits on the
host; all neural inference remains in the SV simulator. It does not use PyTorch
or GPU inference. This is a descriptive evaluation, not an automatic quality
approval rule.

The versioned corpus contains sixteen locally authored cases across code, prose,
arithmetic and multi-turn dialogue. It is separate from the local PGO training
prompts and the existing nine-case objective suite. An exact whitespace-normalized
comparison against recorded PGO prompts/responses rejects duplicate texts. This
check is narrow: it does not prove absence of paraphrases or upstream model
pretraining overlap, which is unknown. Never use this evaluation corpus to choose
quantization/calibration scales, fit profiles, or tune approximation parameters.
Freeze an additional unseen evaluation split if repeated iteration starts using
these results to make design choices.

For each example, both engines receive identical prompt IDs and reset logical
KV. Each scored next-token distribution is conditioned on the same prefix and
previous **authored target tokens**, never either engine's generated prediction.
The generation prefix comes from the existing ChatProtocol. The target is encoded
separately without added special tokens; EOS is not appended. The report records
these tokenization choices and all IDs. Prefill block size is explicit for each
engine (1 or 4), including tail blocks, and position checks guard alignment.

Every prediction is dumped as exactly 151,936 finite binary64 logits. Stable
log-softmax subtracts the maximum before exponentiation. Per-token output includes
reference/candidate negative log likelihood in nats, candidate-minus-reference
loss delta, top-1 agreement, target top-1 indicators, KL(reference||candidate),
and maximum raw-logit difference. Constant logit shifts leave probabilities/NLL/
KL unchanged while changing raw-logit errors, so the latter are only diagnostics.
Tiny negative computed KL values from floating-point roundoff are preserved,
not interpreted as an improvement or silently clamped. Nonfinite/wrong-shaped
outputs, invalid token IDs, stale dump/greedy disagreement and position mismatch
fail evaluation rather than disappearing from averages.

The primary aggregate is total loss divided by the number of scored tokens,
**not** an unweighted mean of example means. Per-example scores, category-weighted
aggregates, token counts, truncation, selected IDs and total corpus token coverage
remain visible. A separately labeled mean of example mean deltas is supplementary.
Perplexity is exp(mean NLL); overflow is recorded as null without discarding the
underlying finite loss. Probability metrics are host scoring, not neural execution.

The report streams a SHA256 hash of the original checkpoint, records its size,
and hashes config, tokenizer and chat-template files where present. It explicitly
states that both engines use that checkpoint. Binary and corpus hashes, scorer/
chat-protocol source hashes and raw token IDs are also retained. Candidate-specific
external quantized/calibration artifacts are not automatically discovered; those
will need a separate manifest if the candidate loads them. A binary hash alone
never identifies model weights.

Example (paths refer to independently built local executables):

```sh
uv run --locked python -m unittest tests.test_projection_quality -v
uv run --locked tools/evaluate_projection_quality.py \
  --baseline /path/to/exact/Vqwen_behavioral \
  --candidate /path/to/candidate/Vqwen_behavioral \
  --baseline-block-size 4 --candidate-block-size 4 \
  --output build/heldout-quality.json
```

`--case-ids` and `--max-target-tokens` explicitly select a smoke subset; the report
must not be mistaken for full corpus coverage. `--require-exact` is useful only
for an exact-engine sanity check, not as an approximation acceptance threshold.

This small corpus cannot establish broad maintained quality. It measures
probability assigned to particular chosen references, not every valid alternative
answer, free-running error accumulation, task success, safety, or long-context
capability. Loss deltas and KL need to be considered alongside code execution,
objective task checks, generated-text comparisons and a larger independently
held-out corpus. No arbitrary pass/fail loss or KL threshold is supplied.

Initial unit checks use tiny analytic distributions and stub transport only:
identity, constant shifts, known KL, extreme logits, finite/shape/token checks,
unequal-length weighting, authored-token advancement, block tails and alignment.
These validate the harness mathematics, not an approximate model.


## Harness validation

Twelve lightweight analytic/transport/tokenizer/provenance tests pass without
model execution. A separate real smoke compared the independently built exact
non-grouped B4 baseline with the current grouped B4 executable: two authored
target tokens in one dialogue example had bit-identical full-vocabulary logits,
zero loss delta and KL, and complete top-1 agreement. The raw-tokenizers API was
verified using `.encode(...).ids`; no Transformers inference is involved.

[Smoke provenance and metrics](../reports/projection-quality-exact-smoke.json)
explicitly record the small selected coverage. This is a harness identity check,
not evidence that an approximate candidate maintains quality. The full corpus
contains 292 target tokens across 16 cases; a full exact identity run is separate
from this initial smoke.
