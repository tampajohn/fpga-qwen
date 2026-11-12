Explicit FMA32 compiler prototype
================================

Status: bounded compiler milestone validated and independently reviewed. For
internal use only. This is a nonportable extension to the isolated Verilator shortreal fork,
not IEEE SystemVerilog syntax supported by ordinary simulators or synthesizers.
Compiler base: c607db496a5cb5207a14cfa70e1639edb742a4b1; extension commit
47ce0bd9dd0b263f913389a7380e7ad3784ceb2f. Existing compiler and
production artifacts are unchanged.

``$verilator_fma32(a,b,c)`` requires three scalar shortreal operands and returns
shortreal. Explicit narrowing is required at the call site for other types.
Finite operands specify exact product plus exact addition followed by one
binary32 rounding, nearest/ties-to-even, with gradual underflow. Signed zeros,
infinities and NaNs follow fused arithmetic classification; NaN payload/sign
preservation and observable floating-point exception flags are not promised.
The supported runtime requires the default nearest rounding mode with flush to
zero disabled; no host floating environment changes are introduced.

The emitter uses explicit ``__builtin_fmaf``. Ordinary AddF/MulF, DOUBLE math and
strict ``-fno-fast-math -ffp-contract=off`` compilation remain unchanged. There
is no host DPI arithmetic or generated-source substitution. This is primitive
compiler lowering of explicitly written SV math, comparable to a math builtin.
GNU/Clang builtin availability is required; other toolchains are not yet
supported. Actual optimized runtime IR contains explicit llvm.fma.f32; its linked ARM64
binary contains scalar FP32 fmadd, while the ordinary counterexample retains
separate fmul/fadd. This is not SIMD or kernel-throughput evidence.

Required constant evaluation is unsupported, as are the inherited shortreal
trace/save/VPI/SystemC modes, ports/functions and aggregate operations. There
must be explicit diagnostics rather than accepting unsupported code with a
warning or generating invalid C++. Negative fixtures run individually so one
error cannot mask another unsupported case.

Validation and remaining kernel gates
-------------------------------------

* Generate negative-test golden files with the regression harness, then run
  runtime O0/O3, each argument/type failure, array, parameter/range and trace
  rejection tests. Add other required-constant contexts if review finds holes.
* Run all existing shortreal/real64/cast regressions unchanged. Assert the
  existing fused-vs-separate example still returns zero via ordinary operations.
* Compare actual generated FMA outputs with the independent exact-rational
  ``fused_multiply_add`` oracle, including cancellation, ties, gradual underflow,
  signed zero and intermediate-overflow cancellation. NaNs are compared by
  classification only, not canonical payload equality.
* Inspect strict-flags optimized IR for explicit float FMA and ordinary separate
  float multiply/add, then disassemble the final linked artifact. Reject a speed
  claim if the relevant operation is scalarized or invokes a per-element library
  call; neither instruction choice is assumed from source.
* Only afterward benchmark a tiny SV projection against the same separate-rounded
  kernel, including conversion/transport costs. Full model integration remains
  out of scope until numerical quality and same-route cache/suffix gates pass.

Reproduction
------------

Apply ``verilator-fma32-extension.patch`` in a separate compiler checkout at the
exact base above. A temporary-index apply check passed against that base. Build
with the same isolated CMake procedure described in the shortreal milestone;
do not install over the existing compiler. Generated GNUmake retains strict
floating flags. Run the new ``t_shortreal_fma*.py`` tests alongside the eleven
existing shortreal/real64/cast regression drivers. The final recorded run passed
24 tests with zero failures; initial test-harness setup corrections are recorded
in ``reports/fma32-milestone.json``.

For each normal/optimized runtime ``vlt_sim.log``, run::

  python tools/check_fma32_log.py PATH/TO/vlt_sim.log --output REPORT.json

Both recorded runs passed all 274 actual outputs (274 distinct input triples).
These comprise 18 edge cases and 256 bounded deterministic finite cases, not
uniform sampling of all IEEE encodings. Oracle and checker hashes, actual logs,
compiler/patch hashes, IR excerpt and linked assembly excerpt are archived in
``reports/fma32-*``. Independent review approved the bounded milestone; no full
compiler conformance, projection acceleration or model-quality claim is made.
