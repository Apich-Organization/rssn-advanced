# Module Review: `simd` & `asm_presets` (Phase 5 Audit)

## 2. Design Integrity

### 2.1 Symbolic Math Integration

**Answer:** `asm_presets` and the JIT are complementary, not redundant. `asm_presets` handle bulk batch operations: they take a slice of `f64` values and produce a slice of `f64` values, operating on fixed-shape data with no symbolic dispatch overhead. The JIT handles expression evaluation: it takes an expression tree and produces a scalar function `fn(*const f64) -> f64` for one row. The JIT does NOT generate SIMD because its IR is expression-tree-shaped (deep, narrow) rather than loop-shaped (flat, wide). An `asm_preset` for FMA over `f64x4` is O(N/4) where N is the number of elements; the JIT's output is called N times. These are different call patterns for different use cases: `asm_presets` for bulk data transformation, JIT for expression-tree evaluation. There is no redundancy.

## 3. Extensibility

### 3.1 ASM vs Generics

**Answer:** Fixed in Phase 4 — `SimdKernel` trait was added in `src/simd/kernel.rs` with `ScalarKernel`, `Avx2Kernel`, `NeonKernel` implementations and a `global_kernel()` function that dispatches at runtime via CPUID. Manual `inline_asm!` is confined to the presets; new kernels (e.g. `log`, `exp`) extend the `SimdKernel` trait by implementing a new kernel type, not by writing raw assembly. The trait is the extension point; assembly is an implementation detail of specific preset backends.
