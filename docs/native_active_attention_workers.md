# Fixed attention worker partition experiment

The isolated candidate keeps build20's physical12-worker decode pool and6-worker prefill pool. Ordinary single-token attention assigns its16 query heads to8 useful workers; projections and vocabulary scoring retain12. Other workers still acknowledge completion and briefly spin. This changes task partitioning, not affinity or core ownership, and does not guarantee a shorter critical path: the largest partition is still two heads. No context threshold is selected from earlier8-versus12 results. Live build20 is unchanged.

The first build failed because the build helper compiles every `.cpp` in the source directory and the standalone pool stress test added a second `main`. The [failed manifest](../reports/native-cpu-active8-v1-failed-build.json) and [linker log](../tools/experiments/native-active8/v1-failed-build.log) are retained. Revision2 separates the test from the build source; all three runtime source files remain byte-identical to revision1.

Root-run [pool stress](../reports/native-cpu-active8-pool-stress.json) passed3000 partition jobs across1/6/8/12/64 physical workers, plus caller/worker exceptions, reuse, empty/uneven partitions, invalid active counts and idle wakeups. The [post-run binding](../reports/native-cpu-active8-pool-binding.json) verifies the exact extracted Pool class; it is not a compiler attestation captured at compile time.

The candidate passed100 full-array relative checks: [56 historical arrays](../reports/native-cpu-active8-exact-gate.json), [12 time boundaries](../reports/native-cpu-active8-timeboundary-exact-gate.json), [8 actual63/64 controls](../reports/native-cpu-active8-newboundary-relative-exact-gate.json), and [24 fresh build20 STEP-edge arrays](../reports/native-cpu-active8-step-edge-exact-gate.json). Historical references are explicitly bound through build20's completed release evidence. The known absolute63/64 oracle failures remain failures; this establishes nonregression on these fixtures.

[Compiled inspection](../reports/native-cpu-active8-assembly.json) found the ordinary B1 attention callback has483 instructions in both binaries and the same154 floating-point instruction/operand sequence, including FP16 widening, two-accumulator QK reduction and direct division. No reciprocal-estimate instruction appears in that callback. This is a targeted assembly check, not a proof about every executable path or a performance result.

The [preregistered protocol](../tools/experiments/native-active8/design.md) starts with only3 exploratory pairs at five held-out contexts192/768/1536/2560/3328,128 outputs, physical12/6 in both arms, complete frozen chat prompts and full-logit preflight. Only the fixed exploratory triage followed by full functional gates can justify a fresh20-pair confirmation. The earlier20-pair-first proposal is preserved as a pre-execution draft. The initial screen below is exploratory; no promotion is authorized by these checks.

The [source patch](../tools/experiments/native-active8/candidate.patch), standalone test, root-run controllers and [archive inventory](../reports/native-cpu-active8-source-validation-archive.json) preserve reproducibility. Source-only concurrency review found no publication/lifetime blocker: active count is published with the existing release/acquire generation, and all physical workers complete before callback storage or scratch can change.

## Three-pair held-out screen

The [complete exploratory report](../reports/native-cpu-active8-heldout-discovery.json) retains every raw pair and full-logit preflight; the [decision record](../reports/native-cpu-active8-heldout-screen-decision.json) applies the previously fixed budget triage. The equal-weight geometric mean of context median decode ratios was1.02321. All median request ratios exceeded0.95, so the rule permits spending time on further qualification. It does not establish a speed improvement.

| Context | Median decode ratio | Median whole-request ratio |
|---|---:|---:|
|192|1.0942|1.0809|
|768|1.1324|1.0412|
|1536|0.8410|0.9575|
|2560|1.1128|1.0244|
|3328|0.9672|1.0001|

Ratios above1 favor the candidate. The1536-token decode regression and3328-token median slowdown remain visible; three pairs cannot support context thresholds or a broad performance claim. No contexts or outliers were removed. The [complete messages and token fixtures](../reports/native-cpu-active8-heldout-fixtures.json) were frozen before measurements.

The [full36-case B2 API gate](../reports/native-cpu-active8-verify-api.json) also passed with1/12/64 physical workers. The additional qualification gates below have now passed; independent20-pair confirmation remains pending. The policy remains fixed operation-aware partitioning, not an adaptive context switch. Live build20 remains unchanged.

## Completed relative functional qualification

The [bound aggregate](../reports/native-cpu-active8-qualification.json) now includes the prior100-array/API36 evidence and all six additional checks against pristine build20: [canonical prefix28](../reports/native-cpu-active8-qualification-prefix.json), [suffix35 plus7 cached cases and cancellation/restart](../reports/native-cpu-active8-qualification-suffix.json), [CLI5](../reports/native-cpu-active8-qualification-cli.json), [nine-case quality](../reports/native-cpu-active8-qualification-quality.json), [eighteen-case quality](../reports/native-cpu-active8-qualification-extended.json), and [seeded sampled streams](../reports/native-cpu-active8-qualification-seeded.json). The archives retain all token arrays, case outcomes and source/binary bindings.

Both engines scored4/9 and10/18 respectively. Passing means exact trajectory agreement and no new regressions on these finite suites, not better answer accuracy or universal correctness. Seeds42/1337 and repeated runs retain the same sampled trajectories. The known absolute63/64 numerical failures remain disclosed.

The [qualification archive inventory](../reports/native-cpu-active8-qualification-archive.json) includes byte-identical executed controller/checker and native-reference eighteen-case helper snapshots. Confirmation, original-context regression controls and any release decision remain separate; these results do not promote the candidate or alter live build20.
