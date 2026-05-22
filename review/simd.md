# Module Review: `simd` & `asm_presets`

## 1. Performance Issues (High Severity)

### 1.1 Innermost-Loop Feature Detection
Every SIMD kernel (e.g., `add_f64x4_avx2::apply`) performs runtime CPU feature detection (e.g., `is_x86_feature_detected!("avx2")`) on **every call**. Since the batch wrappers process data in 4-element chunks, a 1-million element slice triggers 250,000 feature detection checks. This overhead significantly outweighs the gains from SIMD vectorization.
- **Fix:** Perform feature detection once at the start of the batch operation and use function pointers or a specialized loop.

### 1.2 High Call Overhead for Small Width
The kernels are designed for 4-lane (256-bit) widths. The overhead of calling a function, checking lengths, and performing feature detection for just 4 lanes of work is high. Modern compilers can often auto-vectorize better than this manual "chunked" approach if given the right hints, especially when the manual approach adds so much branchy overhead.

### 1.3 Suboptimal Batch Hashing
`batch_hash` calls the AES-NI kernel per element, mixing a key with its own rotation. This uses the 128-bit (2-lane) AES-NI pipeline inefficiently.
- **Fix:** Process two keys at a time in the AES-NI pipeline to double the hashing throughput.

## 2. Engineering Standards

### 2.1 Silent Failure in Kernels
The kernels in `asm_presets` return silently if the input slice lengths are not exactly 4:
```rust
if lhs.len() != 4 || rhs.len() != 4 || out.len() != 4 {
    return;
}
```
If a caller (internal or external) makes a mistake, the operation simply doesn't happen, leading to extremely hard-to-debug "ghost" bugs where results are just old values.

### 2.2 Brittle Assembly Constraints
The assembly uses `vmovupd` and `ymmword ptr [{lhs}]`. While `vmovupd` handles unaligned access, the reliance on `in(reg) lhs.as_ptr()` instead of letting the compiler handle the memory operand can sometimes lead to suboptimal register pressure or missed optimizations in the surrounding code.

## 3. Deviations from Plan

### 3.1 "SIMD Fallback" vs Primary Path
The plan suggests SIMD as a "fallback" or "preset" for common scenarios. However, the current implementation is very rigid and doesn't integrate well with the JIT or the DAG storage (which uses 80-byte nodes, making batch SIMD on nodes impossible without first copying to flat arrays).

## 4. Suggestions
- Use the `multiversion` crate or similar pattern to perform feature detection at a higher level.
- Increase the batch size or unroll the loops in `simd/arithmetic.rs` to amortize the call overhead.
- Fix kernel silent failures; they should at least `debug_assert!` the lengths.
- Optimize the `batch_hash` to process 2 elements per AES-NI round.
