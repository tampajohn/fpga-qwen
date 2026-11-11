# Independent attention dot-lane review

The optional eight-lane QK reduction introduces no correctness blocker found in this review. Each lane accumulates columns lane, lane+8, through127; all eight partials reset for each causal key position and query head. The final ordered lane sum changes association, while scaling, softmax and value reduction remain unchanged. Head/query indices remain0..2047; KV indexing and worker lifetime are unchanged. The build option requires parallel attention and is off by default.

The partition/cache fixture covers workers1,3,8,17, including empty head partitions, sampled IDs, full finite logits within1e-6, truncate/replacement and reset/restart replay. Longer-context numerical and292-target checks were performed separately by the root; this review does not substitute those tests with source inspection.

The [independent free-running report](../reports/attention-dot-conversations.json) uses explicit prefill8 on both engines, verification capacity4, greedy decoding, no suffix draft, identical promptIDs and fresh per-case sessions. All nine complete output sequences and texts match. Both pass4/9 strict objective cases. This supports no observed regression, not strong conversational quality or universal sampling equivalence.

Reviewed/measured baseline SHA256 `2a77181e4b03777d1d602405f1f44ba83ac03e320fdff02c6fb38877c27d8b21`; candidate `c09071439f6fc9705c1fcceb807af0acbe830259fa1276c8502506f39f94551b`. Correctness jobs may run concurrently; no timings from this suite support performance claims. Reassociation can alter low bits and potentially decisions at exact ties despite the sampled cases agreeing.

Reproduce with `uv run --locked python tools/evaluate_prefill_conversations.py --reference BASELINE --candidate CANDIDATE --reference-prefill-size 8 --output build/attention-dot-conversations.json`.
