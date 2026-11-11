# ARM compiler-target and PGO experiments

These rejected experiments are archived only. The production builder does not
include their PGO options or trainer. To reproduce on the exact historical base,
apply the [archived reproduction patch](../tools/experiments/arm-pgo-reproduction.md)
in a separate worktree before using the PGO commands below.

## Native target: measured, not promoted

Apple Clang 21 defaults to target CPU `apple-m1` on this host. `-mcpu=native`
selects `apple-m4` and enables BF16, I8MM, SME, SME2 and SME FP64 features. This
identifies the compiler's selected target, not a new hardware identification.
The installed toolchain was used without global installs.

The native-only build retained every selected B4 option and all source hashes;
only the CPU compiler target changed. Generic binary SHA256:
`64a806a5d2237364b85c890919b7c1c762a4d0eab2384a31648fdddc4ddb6e92`.
Native binary SHA256:
`f3b5a414e95926f41e246b8088b83d9b5c0a493ec765864ddee4de355399e101`.
Both use BF16 stored weights and SystemVerilog FP64 arithmetic/KV.

Actual linked-worker disassembly contains FP64 NEON `fmla.2d`, with 40 static
instances, and no FP32 `fmla.4s`. Searches found no SME matrix/ZA/streaming
instructions. Enabling SME target features did **not** make this model use SME.
ThinLTO vectorization remarks were captured by a diagnostic-only relink into a
separate executable, preserving the scored binary. The remarks include both
successful vectorization and misses; source locations are unavailable without
debug line information. Static instruction/remark counts are not runtime shares.

Three alternating pairs, both processes resident, eight workers, B4 prefill for
both sides, 32 identical greedy output tokens per request:

| Context | Median prefill ratio | Median TTFT ratio | Median decode ratio |
| --- | ---: | ---: | ---: |
| 85 prompt tokens | 1.020 | 1.018 | 1.030 |
| 452 prompt tokens | 0.986 | 0.986 | 1.006 |

Ratios above 1 favor native. Short decode gains were 2.8–4.3%; long decode ranged
from 6.2% slower to 3.0% faster. The modest short-context benefit did not establish a
consistent improvement across workloads, so the generic target remains selected.
Full-vocabulary logits were bit-identical at eight continuation positions after
both 85- and 452-token prompts. Short block/sampling/cache/cancellation/validation
gates also passed. The expensive 4096-position gate was not repeated for this
compiler-only candidate; the existing generic gate remains separately recorded.

[Paired measurements](../reports/arm-native-paired.json) and
[codegen/source provenance](../reports/arm-native-codegen.json) retain exact
settings and hashes. The historical `serial` key in the paired report denotes
the baseline engine; `reference_block_size=4` explicitly records that both sides
used B4. No inference/compilation from other agents overlapped scored timings.

## Bounded generic-target PGO attempt

The next isolated attempt keeps the generic CPU target and numerical flags.
Instrumentation is explicit in both compilation and linking, and normal process
exit flushes profiles; terminating the simulator would discard them. A profile
use build requires a matching source/toolchain manifest and profile hash and
rejects out-of-date instrumentation profiles. Profile artifacts stay in ignored
`build/` directories; source manifests and reproducible commands document them.

Training uses one projection worker with non-atomic counters to avoid contended
atomic updates in inner loops. This exercises the same SV projection/attention
and B4 arithmetic but does not represent eight-worker scheduler frequencies.
Three small prompts, distinct from held-out scoring prompts, cover greedy token
decode, B4 prefill, longer attention context and sampled suffix verification.
Training elapsed time is not a performance claim. Held-out 85/452-token exactness
and paired timings decide whether PGO is useful; no fast-math or host neural math
is introduced.

### PGO outcome: measured, not promoted

The bounded profile used 87/89/216 prompt tokens and 24/7/16 output tokens across
three training requests. The sampled suffix case proposed 6 tokens, accepted 3,
and evaluated 11 target predictions for 7 outputs. The training simulator exited
normally and `xcrun llvm-profdata` merged a frontend profile with 461 functions and
1,981 blocks. Training took 547.6 seconds wall time including a 275.1-second pause
for another agent's benchmark (~272.4 seconds excluding that pause). These are
instrumentation costs, not model performance. Future runs should reserve an
uninterrupted training window or pause between completed requests; production's
300-second per-command timeout was not changed.

The profile-use build emitted no profile mismatch warnings or errors, and its
final executable contains no instrumentation runtime/counter symbols. Profile
SHA256 is
`94dd40c14e35c50df51e1a60c7c1447561008d0da1d81fc611fe9509f5cb113a`;
profile-use binary SHA256 is
`5f85978468a92b5d25f5cb38dd4f8fa494d39a3bdffd69eb40fe0187532a7022`.
All held-out 85/452-token prompts plus eight continuation positions had bit-exact
full logits. Greedy trajectories matched, and independent block sizes/tails,
sampling, cache, cancellation and malformed-input tests passed. The full capacity
test was not repeated for this compiler-only candidate.

Three alternating held-out pairs, both sides using B4 prefill and eight workers:

| Context | Median prefill ratio | Median TTFT ratio | Median decode ratio |
| --- | ---: | ---: | ---: |
| 85 prompt tokens | 0.977 | 0.976 | 0.990 |
| 452 prompt tokens | 1.000 | 1.000 | 1.011 |

Ratios favoring PGO exceed 1. Short decode ranged 12.5% slower to 3.0% faster; long
decode ranged 2.3% slower to 2.9% faster. This single bounded training profile did
not establish a useful overall improvement. Keep the generic, non-PGO default.
This does not rule out all PGO strategies: one-worker training does not represent
eight-worker scheduler frequencies, and only one small training mixture was
tried. No training prompt was reused in held-out scoring.

[Held-out pairs](../reports/arm-pgo-paired.json),
[training/profile manifest](../reports/arm-pgo-training.json), and
[build provenance](../reports/arm-pgo-build.json) record the experiment. Both ARM
experiments started from main commit `1751fa2`; production neural sources were
unchanged. The native report uses the helper before the optional PGO arguments
were added. Raw profiles, profdata and binaries remain ignored build artifacts.

Reproduce using the selected B4 flag set, with generic CPU target:

```sh
python tools/build_projection_variant.py --lto --unchecked --masked-activation \
  --unsigned-indices --fast-attention --parallel-attention --batched-prefill \
  --block-unroll --compact-projection --pgo-generate "$PWD/build/pgo/raw"
# Use the printed instrumentation binary path as TRAIN_BINARY:
python tools/train_pgo.py --binary "$TRAIN_BINARY" --output build/pgo/workload.profdata
python tools/build_projection_variant.py --lto --unchecked --masked-activation \
  --unsigned-indices --fast-attention --parallel-attention --batched-prefill \
  --block-unroll --compact-projection --pgo-use build/pgo/workload.profdata
```

Each profile-use artifact is tagged with its profile digest. Source, compiler,
Verilator and profile hashes are checked before compilation, and source/profile
changes during the build are rejected. The generated profile is tied to the
source snapshot; do not reuse this historical profile for newer grouped kernels
without retraining and validation.
