# Experimental four-token layer-first prefill and verification

Build independently, or use `make chat` for the validated fast B4 profile.
`make chat-reference` retains the single-token profile:

```sh
python3 tools/build_projection_variant.py --lto --unchecked --masked-activation --unsigned-indices --fast-attention --compact-projection --parallel-attention --batched-prefill --block-unroll
```

`--batched-prefill` adds the block protocol and layer-first engine.
`--block-unroll` additionally uses explicit eight-column chunks for the lane8
projection path; it requires batched prefill. Both require scalar weight transport
and parallel-attention support. Workers0 remains supported through an SV scalar
fallback. Existing token/sample/reset/truncate commands keep working.

## Protocol

```text
block count logits sample_flag penalty token0 ... tokenN uniform0 ... uniformN
block count next_id0 position0 best_logit0 ... next_idN positionN best_logitN
```

Count is1..4. There are exactly count tokens and count uniforms; uniforms are
supplied even when sampling is disabled. Boolean flags are0/1; penalty0..2,
uniforms0<=u<1. The harness validates the entire command, IDs and context capacity
before starting computation. The response contains one triple per input token:
its next-token prediction, resulting prefix length, and selected logit. With
logits0 the ID/logit fields are0 and only positions are meaningful. With logits1,
`dump` exposes the last input's full logits. Block diagnostic logits print17
significant digits; legacy token commands still print6, so numerical tests must
compare binary dumps, not rounded text diagnostics.

## Execution and cache semantics

The engine embeds all active inputs, then processes all four hidden states through
each layer before advancing to the next. Q/K/V, attention output, gate/up and down
matrices each reuse one weight load across the active token block. Projection
inputs are feature-major (`feature*4+token`), with partial sums indexed
`partial[lane][token]`. The unrolled lane8 path keeps each token's original
ascending-column partial sums and ascending-lane final reduction. Inactive lanes
may perform arithmetic on ignored staging values, but never write token history,
KV, output triples, or alter active-token reductions.

Existing SV normalization, RoPE, attention and sampling tasks are reused. Within
a layer, token t writes its own K/V and attends only through initial_position+t.
Existing prefix KV are preserved; future block inputs cannot affect earlier
queries. Sampling sees token history length initial_position+t+1, even though
all block IDs have already been staged. The host supplies uniforms but performs
no sampling probabilities or neural arithmetic.

Vocabulary projection and sampling currently run once per input position after
the layer stack; those large projections are not batched. Verification therefore
still pays a vocabulary projection for every proposed input. This is a complete
layer-first transformer path, not a single projection microbenchmark, but it is
not a finished high-throughput vocabulary verification kernel.

The command is synchronous. Position temporarily tracks the query being processed
inside SV, but no command acknowledgment or externally usable cache advancement
occurs until all layers and requested predictions finish. The final committed
position advances by count. Existing truncate can discard any suffix, including
part of a prior block; subsequent writes replace stale KV. Cancellation kills the
simulator and clears the Python-side cache, as with token inference; there is no
in-process rollback claim for an interrupted block.

## Evidence and remaining gates

The initial dynamic-lane implementation passed independent exact-logit tests for
counts1/2/3/4 and tails, greedy/sampled predictions, future exclusion, truncation,
replacement/reset/restart, malformed commands and cancellation. The4096-token
capacity test was not run in that initial short suite. Generated assembly showed
per-MAC indexed stack accesses for those dynamic partial sums, motivating the
separate unrolled option. The unrolled build generates vector `fmla.2d` operations
across tokens with partial vectors in registers.

The selected unrolled binary SHA256 is
`d576cde12f0fc64499b35647878fe2ddea21b9fdd3c5459f14dd96cf58a36272`.
Independent tests against the separately built live attention reference passed
exact short-suite logits and85/452-token prompts plus8 continuation positions
(maximum error0). Two paired runs measured TTFT reductions from1.596/1.763s to
1.131/1.233s at85 tokens and10.199/10.767s to7.408/7.684s at452 tokens, roughly
1.4× faster. These results establish a modest prefill benefit on this CPU.
Steady decode was13–15% slower in some runs, so this is not a decode improvement.
The full4096-token capacity/crossing-boundary gate subsequently passed, as did
optimized sampled suffix parity and browser checks. See the independent
[block report](../reports/block-correctness.json) and
[suffix report](../reports/suffix-optimized-correctness.json).
The main-workspace rebuild has SHA256
`64a806a5d2237364b85c890919b7c1c762a4d0eab2384a31648fdddc4ddb6e92`;
its source hashes and build options match the scored artifact exactly
([equivalence record](../reports/block-integration-equivalence.json)).
No broad conversational-quality claim follows from these gates.
