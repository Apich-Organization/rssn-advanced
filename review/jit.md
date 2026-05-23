# Module Review: `jit` (Phase 7 — Abstract Interpretation & True SIMD)

## §1 Abstract Interpretation Analysis Pass (`src/jit/analysis.rs`)

**Rewritten in Phase 7** from a shallow `is_nonzero: bool` flag to a full
abstract interpretation framework.

### New `NodeAnalysis` struct

| Field | Type | Meaning |
|---|---|---|
| `lower_bound` | `Option<f64>` | Proven lower bound (value ≥ lb) |
| `upper_bound` | `Option<f64>` | Proven upper bound (value ≤ ub) |
| `is_nonnegative` | `bool` | Provably ≥ 0.0 |
| `is_positive` | `bool` | Provably > 0.0 (implies nonzero AND nonneg) |
| `no_nan` | `bool` | Cannot produce NaN |
| `pow_expansion` | `PowExpansion` | How to lower Pow without powf |

The `is_nonzero()` method derives non-zero from multiple sources:
- `is_positive` is set
- `lower_bound > 0`
- `upper_bound < 0`

### Abstract Interpretation Rules (bottom-up, reverse pre-order)

- **Constant(v)**: lb = ub = v; `is_nonneg = v >= 0`; `is_positive = v > 0`; `no_nan = !nan && !inf`
- **Variable**: all unknown
- **Neg(x)**: flips bounds; nonneg if x.ub ≤ 0; positive if x.ub < 0
- **Add(a, b)**: lb = a.lb + b.lb; ub = a.ub + b.ub; nonneg if both nonneg; positive if one is positive and other is nonneg
- **Sub(a, b)**: lb = a.lb - b.ub; ub = a.ub - b.lb; sign derived from bounds
- **Mul(a, b)**: conservative bounds when both nonneg; nonneg if both nonneg; positive if both positive
- **Div(a, b)**: nonneg if a nonneg and b positive; positive if both positive; no_nan if b nonzero
- **Pow(x, even-int)**: nonneg always; positive if base nonzero; lower_bound = 0.0
- **Pow(x, 0.5)**: nonneg always; no_nan if base nonneg
- **Pow(x, neg-int)**: positive if base nonzero (for even n) or base positive (odd n)

### Key correctness benefit: `x^2 + 1`

- `Pow(x, 2)` → `is_nonneg = true`, `lower_bound = Some(0.0)`
- `Constant(1)` → `is_positive = true`, `lower_bound = Some(1.0)`
- `Add` → `is_positive = true` (nonneg + positive rule)
- `x / (x^2 + 1)` — denominator proven positive → NaN guard elided

### Tight-bound exponent detection

The analysis detects `Pow(x, Neg(Constant(1.0)))` as `NegIntPow(1)` by checking
if the exponent's analysis has `lower_bound == upper_bound` (tight bounds). This
handles parser output where `-1` is `Neg(1)` not `Constant(-1)`.

## §2 Extended `PowExpansion` Enum

```
PowExpansion::NegIntPow(u32)   // x^(-n) → 1 / x^n, for n in 1..=4
```

`classify_exponent(-1.0)` → `NegIntPow(1)`, `-2.0` → `NegIntPow(2)`, etc.

## §3 NaN Guard Elision Fix (`src/jit/compiler.rs`)

**Bug fixed in Phase 7**: The Div NaN guard check previously used the Div
node's own analysis (`parent_nonzero`), which only proves nonzero when BOTH
numerator AND denominator are nonzero. This is a weaker condition than needed
for guard elision (we only need the denominator to be nonzero).

**Fix**: `emit_operator` now receives `child_analyses: &[Option<&NodeAnalysis>]`
instead of a single `node_analysis`. For `Div`, it checks `child_analyses[1]`
(the denominator's analysis) directly:

```rust
let rhs_nonzero = child_analyses.get(1).and_then(|a| *a)
    .map_or(false, |a| a.is_nonzero());
let skip_guard = opts.elide_nan_guard && (rhs_nonzero || rhs_is_const_nonzero);
```

The Pow node's own analysis is passed as slot `[2]` for `PowExpansion` lookup.

## §4 New Peephole Identities

| Pattern | Before | After |
|---|---|---|
| `0 - x` | `fsub(0, x)` | `fneg(x)` |
| `(a*b) - c` | `fmul(a,b); fsub(..., c)` | `fma(a, b, fneg(c))` |
| `x * -1` | `fmul(x, -1)` | `fneg(x)` |
| `-1 * x` | `fmul(-1, x)` | `fneg(x)` |

The `a*b - c → fma` peephole mirrors the existing `a*b + c → fma` optimization.

## §5 `NegIntPow` Code Generation

In `emit_operator` for `Pow` with `NegIntPow(n)`:

```
x^(-n) = 1 / x^n
```

1. Emit `x^n` via `emit_int_pow(builder, base, n)`
2. Emit `f64const(1.0)`
3. Check `child_analyses[0].is_nonzero()`: if provably nonzero, skip guard
4. Otherwise: `select(x^n == 0, NaN, 1 / x^n)`

`passes::emit_neg_int_pow` provides the scalar (F64) version without guard.

## §6 True F64X2 SIMD Batch Evaluation

**Replaced in Phase 7** from 2× unrolled scalar to genuine F64X2 SIMD.

### `compile_batch_f64x2` vec_body

**Old**: Two independent scalar expression trees for rows i and i+1 via ILP.

**New**: True F64X2 loads, F64X2 arithmetic, F64X2 stores:

1. For each variable column, load 16 bytes = `F64X2` = [row_i, row_i+1]
2. Evaluate entire expression tree on `F64X2` values via `emit_ast_simd_f64x2`
3. Store `F64X2` result = 16 bytes = [out_i, out_i+1]

This halves the number of load/store instructions and halves the expression
tree evaluation depth vs. 2× scalar.

### `f64x2_const(builder, v)` helper

Creates a `vconst` (F64X2) with both lanes set to `v`:

```rust
let data: [u8; 16] = [lo_f64_bits..., lo_f64_bits...];  // same value both lanes
let handle = builder.func.dfg.constants.insert(ConstantData::from(&data[..]));
builder.ins().vconst(types::F64X2, handle)
```

### `emit_ast_simd_f64x2`

Iterative post-order walker emitting F64X2 instructions. Key differences from
the scalar walker:

- Constants → `f64x2_const(builder, v)` (splat both lanes)
- Variables → pre-loaded `F64X2` values from `var_vals_vec`
- All arithmetic (fadd, fsub, fmul, fdiv, fneg, fma, sqrt) is polymorphic
  in Cranelift — the same instructions work for F64X2 operands
- `emit_int_pow(builder, base_vec, n)` works because it uses `fmul` (polymorphic)
- NaN guard for Div: `fcmp + bitcast(F64X2) + bitselect` pattern

### `emit_operator_simd_f64x2`

F64X2 version of `emit_operator`. The NaN guard uses:

```rust
let is_zero_bv = builder.ins().fcmp(FloatCC::Equal, rhs, zero_vec);
// fcmp on F64X2 returns a boolean vector mask
let is_zero_mask = builder.ins().bitcast(F64X2, MemFlags::new(), is_zero_bv);
// bitselect: select nan_vec where is_zero_mask is set, else div_result
builder.ins().bitselect(is_zero_mask, nan_vec, div_result)
```

### Vectorizability check update

`is_vectorizable_ast` now also accepts `NegIntPow` as vectorizable (was only
`Sqrt`, `IntPow`). The SIMD walker returns `MalformedNode` for `Mod` and
`Function` nodes (excluded by vectorizability check).

## §7 `src/jit/passes.rs` Additions

- `emit_neg_int_pow(builder, lhs, n)` — scalar `lhs^(-n) = 1/lhs^n`
  (no guard; caller's responsibility)
- Updated docs: `emit_sqrt` and `emit_int_pow` both work polymorphically on
  F64X2 values since Cranelift's `sqrt` and `fmul` are type-polymorphic

## §8 New Tests

| Test | What it verifies |
|---|---|
| `test_neg_int_pow_x_inv` | `x^(-1)` gives 1/3 for x=3; NaN for x=0 |
| `test_analysis_x_squared_plus_1_is_positive` | Analysis correctly marks `x^2+1` as `is_positive` |
| `test_nan_guard_elision_x_sq_plus_1_denominator` | `x/(x^2+1)` compiles without NaN guard; correct for x=0 |
| `test_batch_f64x2_true_simd` | True SIMD batch: correct results for 5 rows (4 SIMD + 1 scalar tail) |
| `test_zero_minus_x_is_fneg` | `0 - x` for x=3 gives -3 (fneg peephole) |
