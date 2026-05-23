# Module Review: `jit` (Phase 3 Audit)

## 1. Performance & Numerics

### 1.1 The "Splat" Bottleneck
`batch_eval` iterates over input rows and calls the JIT function one by one.
- **Sharp Question:** We have a JIT and we have SIMD presets. Why are we not emitting vectorized IR in the JIT to process 4 or 8 rows at a time? Is our "batch evaluation" just a loop wrapper over a scalar function, and if so, where is the "advanced" part?

## 2. Extensibility

### 2.1 The Unary Function Prison
`register_custom_function` is hardcoded to `fn(f64) -> f64`.
- **Sharp Question:** In a world of `pow(x, y)`, `atan2(y, x)`, and `min(a, b)`, why is our custom function registry limited to unary operators? How does a user implement a binary custom function without rewriting the `codegen.rs` logic?

## 3. Dead Code

### 3.1 The `CustomRule` Ghost
`src/jit/custom.rs` defines a `CustomRule` struct that is never used by the compiler.
- **Sharp Question:** Is this a forgotten feature or an abandoned design? Why is dead code sitting in the middle of our performance-critical JIT pipeline?
