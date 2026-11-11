# Archived ARM native and PGO experiments

Neither native CPU targeting nor this bounded PGO profile established a useful
consistent improvement. Production defaults and the active builder are unchanged.
The archive retains measured results and a patch for optional reproduction.

Exact baseline commit: `1751fa2d6cffbd60c86f22b758c7da9f456c5c6f`.
Experimental commit: `f2eac63` on `codex/arm-tuning`.

Apply the accompanying `arm-pgo-reproduction.patch` only in a separate worktree
checked out at that exact baseline. It adds explicit profile-generation/use
options, the bounded trainer, and a reference-block-size benchmark option. No
weights, executables, raw profiles or profdata are included in the archive.

```sh
git worktree add ../arm-pgo-reproduce 1751fa2d6cffbd60c86f22b758c7da9f456c5c6f
# From the repository containing this archive:
cp tools/experiments/arm-pgo-reproduction.patch ../arm-pgo-reproduce/
cd ../arm-pgo-reproduce
git apply --check arm-pgo-reproduction.patch
git apply arm-pgo-reproduction.patch
```

The native-only experiment uses the baseline's existing `--cpu native` option;
it needs no PGO patch. The applied PGO patch adds `--pgo-generate` and `--pgo-use`.
Use the B4 flags and commands documented in
[the experiment report](../../docs/arm_compiler_tuning.md). Generate new training
profiles locally; the reported historical profile hashes identify the measured
artifacts and do not promise bit-identical profiles on a new training run.

Reserve an uninterrupted training window. The measured run used one worker to
avoid contended instrumentation counters and was paused for275.1 seconds; the
report separates that pause from instrumentation cost and does not treat
training metrics as inference speed. Held-out evaluation used eight workers.
Do not modify production transport timeouts for training.

The archived patch is not a proposal to enable PGO in production. Any further
variant needs fresh exactness/quality gates and independently paired measurements.
