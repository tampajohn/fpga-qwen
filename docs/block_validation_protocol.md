# Independent RTL block validation

The block interface is `block count logits sample_flag penalty tokens[count]
uniforms[count]`, with count 1–4. It returns one next-token prediction, committed
position and diagnostic logit per input position. `tools/benchmark_blocks.py`
implements an independent raw adapter instead of reusing the production batching
or suffix orchestration. This tests the actual simulator protocol and arithmetic.

`tests/test_block_model.py` covers sizes 1, 2, 3, 4 and tails; greedy and fixed-uniform
sampling; future-token causal exclusion; prefix truncation/replacement, reset and
restart; invalid counts/tokens/uniforms; and cancellation that discards the process.
Full 151,936-element binary logit dumps provide the numerical gate. Legacy scalar
wire diagnostics print six significant digits, while block diagnostics print 17;
those rounded text fields are not used as the numerical oracle. Exact bit identity
is default. An explicit `SV_BLOCK_ATOL`/`--logit-atol` may request at most 1e-6 only
when justified compiler reassociation is documented; the harness never relaxes it
on failure.

Set `SV_BLOCK_BINARY` to the actual candidate and optionally `SV_BLOCK_REFERENCE`
to a separate reference executable. `RUN_BLOCK_CAPACITY=1` enables the expensive
real 4096-position capacity test: fill 4092, commit 4 through 4096, truncate 4095 and
reject a crossing 2-token block. Ordinary short test runs explicitly skip it and
must not be reported as proving the capacity boundary.

The timing harness compares serial prefill against true layer-first RTL blocks,
then runs the same single-token decode path. It measures prefill throughput,
launch-to-ready, prompt-to-first-token and subsequent decode separately. Both
processes remain resident but requests run sequentially in alternating AB/BA
order. Each request resets logical KV, uses identical prompt IDs, and requires
identical output IDs with no early EOS. Numerical gates evaluate every logit
following the prompt plus teacher-forced continuations. The long profile has 452
prompt tokens. Two modes are supported: same candidate executable for serial and
block paths, or an independently selected `--reference-binary`.

```sh
SV_BLOCK_BINARY=/absolute/path/to/candidate \
  uv run --locked python -m unittest tests.test_block_model -v
uv run --locked tools/benchmark_blocks.py --binary /absolute/path/to/candidate \
  --profiles short,long --block-size 4 --pairs 3 --tokens 32 --gate-tokens 8 \
  --output build/block-benchmark.json
```

Real suffix verification is an additional gate owned by the chat implementation:
suffix on/off must preserve text and fixed-seed sampling while reporting actual
proposed, verified and accepted token counts. Rejected speculative KV must be
truncated to the committed history. Prefix reuse, cancellation, EOS, context
limits and backend switches need real browser coverage. Merely reducing IPC
calls does not establish a neural-math speedup; implementation inspection must
confirm layer-first block projections and shared weight traversal.

## First optimized B4 result

The optimized layer-first B4 executable (`d576cde12f0fc64499b35647878fe2ddea21b9fdd3c5459f14dd96cf58a36272`)
was compared with the independently built live parallel-attention reference
(`d6828bf12adbbe1338be325074ca2f92efb291683e07a39d3b8c6a863e089106`).
The short tests passed with exact binary logits, matching greedy and sampled IDs,
and no tolerance relaxation. Both 85-token and 452-token prompts followed by eight
teacher-forced continuation positions had all 151,936 logits bit-identical, max
absolute error 0. All warm/timed 32-token greedy continuations matched.

Two alternating AB/BA pairs, empty logical KV for every request:

| Prompt | Pair | Serial TTFT | B4 TTFT | Serial prefill tok/s | B4 prefill tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 85 | 1 | 1.596s | 1.131s | 53.55 | 75.78 |
| 85 | 2 | 1.763s | 1.233s | 48.28 | 69.63 |
| 452 | 1 | 10.199s | 7.408s | 44.36 | 61.13 |
| 452 | 2 | 10.767s | 7.684s | 42.02 | 58.98 |

TTFT improved 1.41–1.43× short and 1.38–1.40× long. The prefill denominator excludes
the final prompt token, whose forward pass produces the first output token.
The separate decode measurements were not uniformly neutral: serial→candidate
short 40.56→41.31 and 41.04→34.72 tok/s; long 33.64→32.52 and 30.33→26.42. These two
pairs show variability and possible decode regression, not a decode speedup.
Both binaries used the same serial token interface for this decode phase.

No other agent inference/compilation ran during scored timings. This remains a
small software experiment, not an FPGA estimate or a win over native llama.cpp.
The historical FP16 native 85-token TTFT (~0.235s) is still substantially lower,
and was not rerun in this experiment. Source inspection, rather than IPC command
count, establishes that seven dense layer matrices execute in batches with
shared weight traversal; vocabulary heads still run per position.

Full requests, token IDs, hashes, numerical gates and separate timings are in
[block-prefill-paired.json](../reports/block-prefill-paired.json). Actual 4096-position
boundary validation was pending when these timing results were recorded and is
reported separately when completed.

### Actual capacity gate completed

The frozen optimized executable subsequently passed the real 4096-position test:
4,092 input positions were actually evaluated, then a four-token block committed
positions 4,093 through 4,096. Truncating to 4,095 and submitting two more tokens
was rejected with `Invalid block request`. No artificial position-setting API
was introduced. The standalone test took 263.6 seconds while other agents ran
correctness checks; that elapsed time is not a scored performance result.
Malformed count/token/uniform checks were also rerun with explicit process exit
code 1 and `Invalid block` diagnostics, distinguishing validation from a crash.
Both runs passed. Ordinary short test invocation still skips the costly capacity
case unless `RUN_BLOCK_CAPACITY=1` is supplied.
