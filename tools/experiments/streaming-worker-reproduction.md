# Archived streaming-worker reproduction

This is a rejected CPU compiler experiment, not an active build option. The main
builder is intentionally untouched so later grouped-projection and PGO work do
not inherit or conflict with it.

Exact base: `8f48862b8c093587c33dbc366e8fce4bdb43b07f` (Enable validated block prefill and verified suffix drafting
in browser chat). Experiment commit: `d0c5f82624cb5a7a64db1a066b4a55a2396b6b15`. The accompanying patch contains
only the builder annotation hook and its projection microbenchmark driver from
that experiment. It does not modify SV or handwritten model arithmetic.

Create a separate worktree from the exact base, then apply the patch there:

```sh
# Run from the repository root; choose an unused reproduction directory.
archive_patch="$PWD/tools/experiments/streaming-worker-reproduction.patch"
git worktree add --detach /tmp/fpga-qwen-streaming-repro 8f48862
git -C /tmp/fpga-qwen-streaming-repro apply --check "$archive_patch"
git -C /tmp/fpga-qwen-streaming-repro apply "$archive_patch"
cd /tmp/fpga-qwen-streaming-repro
python3 tools/build_projection_variant.py --lto --cpu native --streaming-worker \
  --unchecked --masked-activation --unsigned-indices --fast-attention \
  --parallel-attention --batched-prefill --block-unroll --compact-projection
```

The recorded run used the installed Apple Clang21 and SME-capable ARM CPU.
Inspect the resulting manifest and disassembly before running the projection
probe; another compiler may emit different code. Build the comparison artifact
using identical flags except `--streaming-worker`, then run the archived helper:

```sh
python -m tools.benchmark_streaming_projection --binary STREAMING_BINARY --reference NATIVE_BASELINE
```

Do not apply this patch onto current main: its base predates later builder
options, and the experiment was decisively rejected. Do not promote its whole
worker annotation to the server. The measurements were11–21× slower for tested
projections, and attention incurred mode switches around individual KV reads.
The emitted instructions were two-lane streaming FP64 SVE, not SME ZA matrix
outer products. The active production profile remains unchanged.

See [the findings](../../docs/streaming-worker-experiment.md),
[code-generation evidence](../../reports/streaming-codegen.json), and
[raw projection measurements](../../reports/streaming-projection.json).
