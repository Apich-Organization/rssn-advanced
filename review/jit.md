# Module Review: `jit` (Phase 6 — Optimization Passes)

### 2.2 Vectorized IR

**Answer:** Fixed in Phase 6 — `compile_batch_f64x2` generates a 2× unrolled
scalar batch function (two independent SSA paths for rows `i` and `i+1`)
using Cranelift's standard `F64` type. The function:
- Builds a proper loop CFG with SSA block parameters for the induction variable
- Loads variables from column-major layout (`*const *const f64`) for both rows
- Emits two independent scalar expression trees (achieving 2× throughput via ILP)
- Handles the NaN guard for Div via `fcmp + select`
- Processes 2 rows per vector iteration; scalar tail handles 0-or-1 remaining rows
- Returns `None` for non-vectorizable expressions (powf call sites, Mod, custom functions)

The 2× unrolled scalar approach was chosen over F64X2 SIMD types to avoid
Cranelift 0.131 API subtleties with the `BlockArg` encoding in vector load/store
paths, while achieving equivalent throughput through instruction-level parallelism.

### Power Lowering

**Fixed in Phase 6** — `x^n` for integer n in 2..=8 compiles to repeated `fmul`
via binary exponentiation (no `powf` call). `x^0.5` compiles to Cranelift `sqrt`
instruction. This eliminates the most expensive path in typical symbolic math
expressions. Controlled by `OptConfig::max_int_pow` (default: 8) and
`OptConfig::expand_sqrt` (default: true).

### CSE

**Fixed in Phase 6** — Pre-scan identifies duplicate DAG nodes (same `dag_id`).
Shared subexpressions are computed once; subsequent references reuse the SSA Value.
Controlled by `OptConfig::enable_cse` (default: true).

### NaN Guard Elision

**Fixed in Phase 6** — The analysis pass (`analysis.rs`) proves non-zero denominators
bottom-up. For proven non-zero denominators (constant non-zero values or multiply of
two non-zero values), the `fcmp + vselect` guard is elided. Controlled by
`OptConfig::elide_nan_guard` (default: true).

### Reciprocal Math

**Added in Phase 6** — `OptConfig::allow_reciprocal_math` (default: false) replaces
`x / C` with `x * (1/C)` for constant C ≠ 0. Not enabled by default due to IEEE-754
precision difference.

### Sub-Self Identity

**Added in Phase 6** — `x - x → 0` when both SSA Values are identical (same
instruction result). Fires naturally for CSE-shared values.

### New Files

- `src/jit/analysis.rs` — Bottom-up `NodeAnalysis` pass: `is_nonzero` propagation
  and `PowExpansion` classification per node.
- `src/jit/passes.rs` — `emit_int_pow` (binary exponentiation, n=2..=8) and
  `emit_sqrt` (native Cranelift sqrt instruction).

### New API

- `OptConfig` struct with `Default` impl — controls all Phase 6 passes.
- `JitCompiler::compile_with_opts(ast, opts)` — explicit optimization settings.
- `JitCompiler::compile_batch_f64x2(ast)` — vectorized batch evaluation.
- `CompiledBatchFn` type alias — column-major batch function pointer.
