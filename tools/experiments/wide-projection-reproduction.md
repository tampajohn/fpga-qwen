# Reproduce the rejected lane-major projection probe

The rejected kernel is archived as a patch, outside active runtime sources.
Apply only in a separate checkout at base `28e2f6cbf08c5909e8b9f31b0d4e04ef9bd9c07b`. The complete original
experiment is commit `5972eda`; its results reject this layout, not B8 generally.

```sh
git worktree add --detach ../fpga-qwen-wide-repro 28e2f6cbf08c5909e8b9f31b0d4e04ef9bd9c07b
cd ../fpga-qwen-wide-repro
git apply ../fpga-qwen/tools/experiments/wide-projection-reproduction.patch
uv run --locked python tools/probe_wide_projection.py --batch 4 --repetitions 100
uv run --locked python tools/probe_wide_projection.py --batch 8 --repetitions 100
```

The checkpoint, Verilator and compiler must be available locally. Probe timing
excludes dispatch and is not model throughput. No GPU or PyTorch is used.
