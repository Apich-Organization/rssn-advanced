# Module Review: `simd` & `asm_presets` (Phase 3 Audit)

## 1. Performance

### 1.1 The Missing FMA
We have a high-performance `vfmadd231pd` kernel in `asm_presets`, but the engine never uses it.
- **Sharp Question:** Is our assembly library just for show? Why doesn't the JIT or the Heuristic engine identify `a*b+c` patterns and map them to our `batch_fma` or a fused IR instruction?

## 2. Design Consistency

### 2.1 The Scalar Fallback Loop
The batch wrappers (e.g., `batch_add`) implement a manual scalar loop for the tail and for cases where SIMD is missing.
- **Sharp Question:** We have `OpKind` logic everywhere else. Why are we duplicating basic arithmetic logic (`a + b`) inside the `simd` module? Shouldn't the "scalar path" just be a dispatch to a central evaluation unit?

## 3. Extensibility

### 3.1 ASM vs Portability
The kernels are hand-written for AVX2, NEON, and RVV.
- **Sharp Question:** If a new architecture arrives (e.g., AVX-512 with masking), do we really want to hand-write 50 more files? Why are we not using a portable SIMD abstraction if our current kernels are performing such basic operations?
