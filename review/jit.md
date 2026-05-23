# Module Review: `jit` (Phase 5 Audit)

### 2.2 Vectorized IR
`batch_eval` still iterates over rows.
- **Sharp Question:** If we have the metadata to know an expression's width, why are we still generating scalar code? Could we generate SIMD-vectorized IR that processes 4 or 8 `f64` values at a time, or is the overhead of "splatting" variables into vectors too high for the typical expression?
