# CPU-only locally streaming generated-worker experiment

This isolated experiment does not change SV arithmetic, handwritten host math,
GPU configuration, or the live profile. The helper generates the worker C++, adds
exactly one `__arm_locally_streaming` declaration keyword to its arithmetic hot
function, then compiles it with the installed Apple Clang21 and ThinLTO under
`--cpu native`. The build manifest records the generated filename, symbol and
before/after hashes. Removing that keyword is asserted to recover the original
source bytes exactly. SV remains the authoritative neural implementation.

The attribute asks Clang to execute a function body in SME streaming mode while
preserving its ordinary interface. The compiler manages transitions; calls into
non-streaming code can require leaving and re-entering that mode. See the
[official Clang attribute reference](https://clang.llvm.org/docs/AttributeReference.html#arm-locally-streaming).
No manual mode-switch instructions, custom tensor intrinsics, or matrix kernel
were inserted.

## Feasibility and generated instructions

The relevant generated arithmetic function is
`Vprojection_worker___024root___nba_sequent__TOP__0`. The similarly named
`eval_nba` is only a trigger/dispatch wrapper; annotating that wrapper did not
produce useful streaming arithmetic in the final linked program. The reproducible
helper targets the arithmetic function directly and rejects ambiguous matches.

Actual hot-function candidate SHA256:
`45403d4d1e02a21d28107de836a2625b21869afee9bb67b3fe252c60e33d29a4`.
The compile succeeded without changing any mathematical statement. Disassembly
shows26 static streaming `fmla z*.d` instructions, using `p1.d, vl2`: two active
FP64 lanes. This is streaming SVE code, not SME matrix/ZA outer-product code, and
it does not establish a wider arithmetic width than the previous two-lane NEON
path.

The checked/unchecked weight transport callbacks inline in projection paths;
there is no call or streaming-mode transition per weight. However, attention's
key/value callbacks remain calls in the same worker function. Each callback is
surrounded by `smstop sm` and `smstart sm`; softmax's `exp` call also switches.
These callback sites sit inside the head/time/dimension loops. Consequently this
whole-worker annotation is not a viable default for attention, regardless of any
projection-only result. No full-model throughput run is warranted on it.

A future projection-only variant would have to isolate the projection body while
keeping attention non-streaming. A pure SV task extraction with explicit inputs
and outputs might provide a generated function boundary. Verilator's
[`no_inline_task` restriction](https://verilator.org/guide/latest/extensions.html)
requires the task and its callees to avoid references to variables outside the
task. Any extraction must satisfy that restriction and retain all arithmetic in SV. This experiment does
not implement that extraction or annotate unrelated callbacks to conceal the
switching cost.

## Archived reproduction only

The rejected builder option and microbenchmark driver are not installed in the
active tree. Restore them only in a separate worktree at the recorded base using
[the archive instructions](../tools/experiments/streaming-worker-reproduction.md).
The commands below apply inside that reproduction worktree after patching.

```sh
python3 tools/build_projection_variant.py --lto --cpu native --streaming-worker \
  --unchecked --masked-activation --unsigned-indices --fast-attention \
  --parallel-attention --batched-prefill --block-unroll --compact-projection
python -m tools.benchmark_streaming_projection --binary CANDIDATE --reference NATIVE_BASELINE
```

The optional builder rejects streaming unless native CPU tuning and ThinLTO are
selected. Normal builds are unchanged. It regenerates a previously annotated
source before a repeat build, avoiding cumulative edits. Inspect the emitted
instructions on each toolchain; source annotations alone are not evidence of
which instructions actually execute.

## Decisive projection-only result: reject

The native-only baseline SHA256 is
`f3b5a414e95926f41e246b8088b83d9b5c0a493ec765864ddee4de355399e101`.
Both executables used eight persistent workers, eight reduction lanes, the same
weights and existing `bench` command. Each sample averaged five projections;
one warmup preceded alternating baseline/candidate then candidate/baseline pairs.
No other agent inference, training or compilation overlapped scored samples.

| Projection | Native baseline pair times | Streaming pair times | Approximate slowdown |
| --- | ---: | ---: | ---: |
| Q,2048×1024 | 0.307/0.365 ms | 3.702/3.978 ms | 11–12× |
| Down,1024×3072 | 0.390/0.407 ms | 5.459/4.974 ms | 12–14× |
| Vocabulary,151936×1024 | 11.977/12.144 ms | 250.369/248.112 ms | 20–21× |

Every exposed first/last output value matched. The existing microbench does not
expose interior values; this is not a full numerical-equivalence or model-quality
claim. All benchmark processes completed normally on the actual CPU. The
slowdown is decisive for rejecting this candidate, despite the small sample;
its precise microarchitectural cause was not isolated.

This result provides no realistic upside for extracting the same two-lane
streaming projection code into a separate SV task. It does **not** show that SME
matrix kernels are inherently slow: this executable contains streaming FP64 SVE
FMA, with a two-lane predicate, and no SME ZA matrix outer-product operations.
A mode flag must not be presented as matrix acceleration. No further tuning or
broad model run was performed, and no main/live profile was changed.

Raw measurements: [streaming-projection.json](../reports/streaming-projection.json).
Code-generation evidence: [streaming-codegen.json](../reports/streaming-codegen.json).
