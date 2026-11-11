# Native SystemVerilog shortreal compiler milestone

An isolated Verilator fork now emits genuine FP32 storage and arithmetic for a bounded SystemVerilog subset. Eleven compiler/runtime regressions pass, including unchanged real64/integer/cast tests. Independent review confirmed that all19 unsupported-case fixtures fail explicitly without internal compiler errors. This is approved for a projection microkernel experiment, not a general-purpose compiler release or model precision change.

The exact-base patch is [verilator-shortreal-subset.patch](../tools/experiments/verilator-shortreal-subset.patch), against upstream v5.054 commit `bbb60a403932a9f695cb6ff001ceb73de6ad9234`. It contains the compiler changes and all new regression sources/drivers/goldens. The isolated compiler commit is `c607db4`; complete hashes and test names are in [shortreal-milestone.json](../reports/shortreal-milestone.json). Nothing is installed globally; the production model still uses its original compiler and FP64 math.

## Supported and rejected boundaries

The fork preserves a distinct FLOAT datatype while leaving DOUBLE as real/realtime. It supports internal shortreal scalars, typedefs and fixed unpacked arrays, scalar element assignment, explicit shortreal/real casts, 32-bit bitcasts, and add/subtract/multiply. Mixed real operands promote to real64; otherwise shortreal arithmetic has FP32 results. The language rule follows IEEE1800 section11.3.1 and the [official Accellera proposal](https://accellera.org/images/eda/sv-bc/att-6692/1035_D3a.htm). Existing real literals retain their real64 type until explicitly narrowed.

The tested subset rejects shortreal ports/function signatures/DPI, parameters and required constant evaluation, integral-to-shortreal conversions wider than32bits, bitcast inputs wider than32bits, unary/conditional/comparison/division/power/bitwise/modulus operations, whole-array copies, streaming and formatting. Trace/save/VPI/SystemC and public shortreal variables are also excluded. This list is a milestone boundary, not a claim that every SystemVerilog construct has been exhaustively audited.

Wide integer conversion is deliberately rejected: routing integer64 through double can double-round. For example, `0x4000004000000001` should convert directly to FP32 bits`5e800001`, but integer→double→float gives`5e800000`. The fork does not approximate this unsupported conversion.

New FP32 AST nodes are not constant-folded because the existing V3Number representation only stores real64. Required constant contexts reject instead of silently changing NaN payloads or emitting invalid C++. Runtime assignments from constants work. Integral localparams containing shortreal bitcasts have a dedicated regression: checking only the parameter's declared type had initially missed that case. Streaming/AND/modulus each have standalone negative regressions under `-Wno-fatal`, preventing unrelated errors from hiding acceptance.

## Validation and evidence

The O0/O3 tests cover dynamic cross-cycle array indexing, opposite array bounds, casts, `$bits`, ties near2^24, ties at the subnormal boundary, signed zero, subnormals, overflow, quiet/signaling-NaN bitcast payload roundtrips, separate multiply/add rounding, mixed-real arithmetic and a small array projection. Existing `t_math_real`, `t_math_real_round`, `t_cast` and `t_math_signed` pass unchanged. [Raw regression output](../reports/shortreal-regressions.txt) and [independent negative review](../reports/shortreal-independent-review.json) are retained. No commercial standards-compliant SV simulator was available for crosschecking.

Generated storage includes `VlUnpacked<float,3>`. The optimized array projection contains scalar FP32 `fmul`/`fadd` and a two-lane `fmul.2s`; see [assembly excerpt](../reports/shortreal-projection-assembly.txt). These instructions come from typed SV lowering, not replacing generated double declarations or adding a host neural kernel. This is not a throughput benchmark or full-model quality result.

The validated generated build path is GNUmake. When FLOAT appears, its makefile appends `-fno-fast-math -ffp-contract=off` after user flags. Those flags affect embedded real64 operations too. A fair comparison must build the deployed FP64 worker separately with its original compiler and flags; a shared strict-FP binary could artificially handicap the FP64 comparator. Other generators/manual builds require equivalent flags and are outside this milestone's validation.

## Reproduce in an isolated compiler checkout

Create a separate checkout of upstream Verilator at the exact base above and apply the patch with `git apply --check` followed by `git apply`. On the tested macOS host:

```sh
cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release \
  -DFLEX_INCLUDE_DIR=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include
cmake --build build-local -j4 --target verilatorRelease
ln -s ../build-local/include/verilated.mk include/verilated.mk
ln -s ../build-local/include/verilated_config.h include/verilated_config.h
python3 -m venv build-local/test-venv
build-local/test-venv/bin/pip install distro==1.9.0
```

Set `VERILATOR_ROOT` to that checkout, `VERILATOR_BIN` to its `build-local/src/verilator_bin`, and prepend its `build-local/test-venv/bin` to PATH. From `test_regress`, run:

```sh
python driver.py --vlt -j2 \
  t/t_shortreal_subset.py t/t_shortreal_subset_opt.py \
  t/t_shortreal_subset_unsup.py t/t_shortreal_stream_unsup.py \
  t/t_shortreal_and_unsup.py t/t_shortreal_mod_unsup.py t/t_shortreal_const_unsup.py \
  t/t_math_real.py t/t_math_real_round.py t/t_cast.py t/t_math_signed.py
```

The first build is a compiler build, not a global installation. Build paths/toolchain versions can change artifact hashes; the patch/base and semantic tests establish what is reproduced. Do not replace the production compiler until a later review explicitly promotes it.
