# Native FP 32 SystemVerilog simulation: compiler feasibility

A subsequent isolated compiler milestone now passes bounded native-shortreal runtime and rejection tests. See [the milestone, exact-base patch and validation evidence](shortreal_compiler_milestone.md). It is not a general compiler release or a production model precision change. The assessment below records the original read-only investigation.

A correct `shortreal` implementation is possible in principle, but it is a
cross-cutting Verilator language/compiler feature, not a small C++ code-generation
switch. It is not recommended as the next short performance experiment. Replacing
`double` text in generated files would not implement SystemVerilog semantics and
would break type/ABI assumptions.

This read-only review inspected official Verilator tag `v5.054`, commit
`bbb60a403932a9f695cb6ff001ceb73de6ad9234`, cloned under ignored
`build/verilator-shortreal-review`. At the time of this initial assessment, no compiler fork was modified or built, and
nothing was installed globally. The current simulator still uses FP 64 arithmetic.

## Evidence in the official compiler

- [Parser](https://github.com/verilator/verilator/blob/v5.054/src/verilog.y): line 2034
  maps `shortreal` directly to `VBasicDTypeKwd::DOUBLE`; lines 4452/4545 map
  `$bitstoshortreal`/`$shortrealtobits` onto the real/64-bit nodes and issue
  `SHORTREAL`. The original type distinction is already lost before codegen.
- [Type vocabulary](https://github.com/verilator/verilator/blob/v5.054/src/V3AstAttr.h):
  `VBasicDTypeKwd` has `DOUBLE`, with no binary 32 floating type. Its DPI table emits
  `double`; `isDouble()` identifies the floating type.
- [Width/type propagation](https://github.com/verilator/verilator/blob/v5.054/src/V3Width.cpp):
  casting, mixed expressions, assignment conversion, ports and arrays repeatedly
  distinguish `isDouble()` from integral types. For example, lines 10125–10145
  accept matching doubles or insert real/integer conversions; lines 10370 onward
  implement conversions specifically to/from double. A new 32-bit type cannot
  safely fall through existing integral-width paths.
- [Expression nodes](https://github.com/verilator/verilator/blob/v5.054/src/V3AstNodeExpr.h):
  `AstAddD` and `AstMulD` force `dtypeSetDouble()`. `AstBitsToRealD` emits the 64-bit
  helper `VL_CVT_D_Q`. Arithmetic and bit reinterpretation require real type-aware
  lowering, not just altered variable declarations.
- [Constant representation](https://github.com/verilator/verilator/blob/v5.054/src/V3Number.h):
  `V3NumberDataType` contains logic, double and string, with no float variant.
  Constant folding must round at the same specified boundaries as runtime code.
- [C++ type emission](https://github.com/verilator/verilator/blob/v5.054/src/V3AstNodes.cpp):
  the basic floating type maps to `double`. DPI wrappers in
  [V3Task.cpp](https://github.com/verilator/verilator/blob/v5.054/src/V3Task.cpp)
  derive signatures and conversions from these types.
- Existing [DPI array regression](https://github.com/verilator/verilator/blob/v5.054/test_regress/t/t_dpi_arg_output_unpack.cpp)
  contains native float cases but disables them with `NO_SHORTREAL` under
  `VERILATOR`. Existing test presence therefore does not demonstrate support.
- The [language guide](https://github.com/verilator/verilator/blob/v5.054/docs/guide/languages.rst)
  explicitly documents promotion to real. The
  [internals guide](https://github.com/verilator/verilator/blob/v5.054/docs/internals.rst)
  lists IEEE 1800-2023 section 6.12 shortreal among features not anticipated to be
  implemented. This is an upstream roadmap statement, not proof of impossibility.

## Minimum credible implementation scope

1. Preserve a distinct 32-bit floating dtype in parsing and elaboration, including
   typedefs, arrays, function return values, casts, `$bits`, and both shortreal bit
   conversion builtins. Keep `real`/`realtime` at 64 bits.
2. Implement SystemVerilog shortreal expression sizing, mixed real/shortreal/int
   conversions and assignment rounding explicitly. The precise expression rules
   need validation against the applicable IEEE standard and an independent
   compliant simulator; C++ promotion rules cannot be assumed equivalent.
3. Add or generalize floating expression/conversion nodes and constant folding.
   Ensure optimized and unoptimized paths agree on binary 32 rounding, signed zero,
   subnormals, overflow, infinities and NaNs. Control contraction separately:
   fused multiply-add can differ from separately rounded multiply then add.
4. Emit actual C++ `float` storage and operations where required, with binary 32
   temporaries and correctly typed constants. Preserve mixed-precision/system
   function semantics: blindly switching every math call to its `f` variant is
   not automatically correct.
5. Emit DPI float ABI for scalar input/output/inout/return values and fixed/open
   arrays. Update array element size/stride metadata, port copies, model storage,
   tracing/VPI/save support as applicable. Unsupported features must fail clearly
   in an experimental subset rather than silently produce wrong results.
6. Add new parser/type/constant/runtime/DPI regression coverage, and run existing
   real/integer/DPI suites to show that the new type did not change old behavior.

A bounded research milestone would support explicitly typed shortreal scalar
add/multiply and scalar DPI in an isolated compiler branch, with unsupported
constructs rejected. That alone would not run this model: projection arrays,
conversion boundaries, reductions and eventually KV/main-model operations still
need coverage. No amount of declaration-only substitution is evidence of native
FP 32 model arithmetic.

## Validation before any model benchmark

Test 32-bit layout and ABI in generated declarations and compiled fixtures; both
bitcasts must round-trip 32 bits. Exercise halfway rounding, mantissa boundaries
around 2^24, cancellation, smallest subnormals, overflow, signed zero, infinities
and NaN classification. Compare constant versus runtime results, mixed-type
expressions, casts, function boundaries and array/DPI strides. Use an independent
binary 32 oracle with explicit rounding boundaries, plus a standards-compliant SV
simulator where available. Check contraction on/off independently.

Then run a small dot-product fixture and inspect native assembly to establish
actual FP 32 SIMD. Only after those pass should an opt-in model build be compared
against the current FP 64 reference using full logits, greedy choices, sampled
conversation fixtures and objective quality cases. FP 32 model results will not
generally be bit-identical to FP 64; tolerances and quality thresholds must be
specified before comparing performance.

The rejected FP 32 *weight cache* experiment is unrelated: it expanded BF16 storage
while retaining FP 64 neural arithmetic. It supplies no evidence for or against
the performance of a correctly implemented native FP 32 arithmetic backend.
