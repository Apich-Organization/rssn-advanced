# Module Review: `simd` & `asm_presets` (Phase 5 Audit)

## 2. Design Integrity

### 2.1 Symbolic Math Integration
- **Sharp Question:** Why do we maintain a separate `asm_presets/fma_f64x4_avx2.rs` if the JIT is capable of generating the same instruction natively? Is our manual assembly layer becoming a "museum of presets" that are redundant with our JIT's output?

## 3. Extensibility

### 3.1 ASM vs Generics
We are still hand-writing assembly for every architecture.
- **Sharp Question:** As we add more kernels (e.g. `log`, `exp`), will we really hand-write them for 4 different SIMD instruction sets? Is it time to transition to a generic SIMD trait that compiles down to these presets, or are we committed to manual `inline_asm!` forever?
