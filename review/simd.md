# Module Review: `simd` & `asm_presets` (Post-Upgrade)

## 1. Performance & Memory

### 1.1 Hoisted Feature Detection
The transition to `HAS_AVX2` via `OnceLock` successfully removed the runtime CPUID checks from the innermost loops. The 2x unrolled loops in `batch_add` and `batch_mul` further improve throughput by amortizing loop overhead.

## 2. Dead Code & Functionality

### 2.1 Unused `fma_f64x4_avx2`
The `fma_f64x4_avx2` preset and its corresponding `batch_fma` wrapper are implemented but **not used** by the internal symbolic engine. The `HeuristicEngine` and `JitCompiler` do not currently recognize FMA opportunities (e.g. `a*b+c`). While useful as a utility, it is currently "dead code" from the perspective of the core symbolic computation engine.

## 3. Extensibility

### 3.1 Architecture Silo
The SIMD kernels are very specific to x86_64/AVX2. Adding support for a new architecture (e.g. AVX-512 or AMX) requires manually writing assembly and boiler-plate for each new kernel. There is no high-level SIMD abstraction (like `std::simd` or `packed_simd`) that would make this process more "extensible".

## 4. Suggestions
- Implement an FMA detection pass in the `HeuristicEngine` or `JitCompiler` to utilize the existing `batch_fma` logic.
- Consider using a SIMD library to provide a more portable and extensible base for vectorized operations.
