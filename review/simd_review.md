# RSSN-Advanced Review: `src/simd`

## **1. Alignment with `plan.md` & Instructions**

### **Naked/Inline ASM Presets**
- **Status**: **CRITICAL FAIL**
- **Issue**: The user explicitly instructed: "shall use more naked asm/inline asm presets for complete suites of presets".
- **Evidence**: `src/simd/arithmetic.rs` and `src/simd/hash.rs` use standard Rust loops. They rely on the compiler's auto-vectorizer rather than the requested `naked_asm` or `inline_asm` presets.
- **Impact**: Auto-vectorization is brittle. It may fail to vectorize under certain compiler versions or flag combinations, leading to a silent and massive performance drop.

### **Hardware Level Optimization**
- **Status**: **PASS (Partial)**
- **Observation**: Correctly uses `std::is_x86_feature_detected!` for runtime dispatch.
- **Issue**: The "SIMD Fallback" is implemented by simply repeating the same Rust loop in both the `if has_avx2()` and `else` branches. This provides zero benefit if the compiler fails to vectorize.

---

## **2. Performance Issues**

### **Suboptimal Vectorization**
- **Issue**: The loops in `batch_add`, `batch_mul`, etc., are extremely basic.
- **Recommendation**: To truly fulfill the "hardware level optimization" goal (§4.2 of plan), use explicit SIMD intrinsics (via `std::arch`) or the requested `inline_asm` to ensure the use of AVX2/FMA instructions.

### **Incomplete Suite**
- **Observation**: Currently only `add`, `mul`, and `hash` are implemented.
- **Recommendation**: Expand to include common symbolic kernels like batch exponentiation, batch comparison, and batch coefficient merging as requested by the "complete suites of presets" instruction.

---

## **3. Safety & Security**

### **Bounds Checking**
- **Issue**: Uses `assert_eq!` for slice lengths.
- **Observation**: This is correct for safety, but in a high-performance SIMD loop, one should ensure that the compiler can eliminate bounds checks inside the loop (which it usually can for simple `0..n` loops).

---

## **4. Error Handling**

### **Macro Non-Compliance**
- **Issue**: Does not use the requested cold-path error macro.
- **Recommendation**: Any failure in feature detection or length verification should use the cold-path macro.
