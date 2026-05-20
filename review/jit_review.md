# RSSN-Advanced Review: `src/jit`

## **1. Alignment with `plan.md` & Instructions**

### **Naked/Inline ASM Presets**
- **Status**: **CRITICAL FAIL**
- **Issue**: The user explicitly instructed: "shall use more naked asm/inline asm presets for complete suites of presets".
- **Evidence**: The JIT module relies entirely on Cranelift's high-level IR. There is no use of `naked_asm` or `inline_asm` for optimized presets (e.g., for specialized symbolic kernels).

### **Explicit Prefetching**
- **Status**: **FAIL**
- **Issue**: `plan.md` (§4.2) states "JIT generated instruction stream MUST contain prefetch".
- **Evidence**: `src/jit/codegen.rs` contains an empty `emit_prefetch_hint` function. This function is **never called** within `compile_node` in `compiler.rs`.
- **Impact**: Significant performance loss for large expressions that exceed L1/L2 cache.

### **Memory Ordering (Acquire/Release)**
- **Status**: **FAIL**
- **Issue**: `plan.md` (§4.2) requires "Strictly prohibit misuse of SeqCst ... use Acquire/Release".
- **Evidence**: There is no evidence of atomic operations or memory ordering controls in the current JIT compilation pipeline.

### **Coefficient Merging**
- **Status**: **FAIL**
- **Issue**: `plan.md` (§3.1) mentions "coefficient merging calculation" for multiplication.
- **Evidence**: `compile_node` simply emits an `fmul` instruction. It does not perform any algebraic merge of coefficients (e.g., merging `(3*x) * (2*y)` into `6*(x*y)`) at the IR level.

---

## **2. Performance Issues**

### **Recursive Codegen**
- **Issue**: `compile_node` is recursive.
- **Risk**: Large symbolic expressions (which are the target of this project) will cause a **Stack Overflow** during compilation.
- **Recommendation**: Implement an iterative IR generation pass.

### **Suboptimal Primitives**
- **Issue**: `src/jit/primitives.rs` contains Rust-level simplifications (e.g., `simplify_add` with identity checks), but these are **not used** to optimize the generated JIT code.
- **Observation**: The compiler always emits a full `fadd`/`fmul` even if one side is a constant `0.0` or `1.0`.

---

## **3. Zero-Copy & `bincode-next`**

### **JIT Cache Serialization**
- **Issue**: `JitCache` uses `HashMap<String, usize>`.
- **Observation**: If this cache were to be persisted (as implied by the project goals), it would not support the requested zero-copy feature of `bincode-next`.

---

## **4. Error Handling**

### **Macro Non-Compliance**
- **Issue**: JIT compilation errors return `Result<..., String>` or use `unwrap()`.
- **Recommendation**: Use the requested cold-path error macro for compilation failures and JIT traps.

---

## **5. Extensibility**

### **Custom Functions**
- **Issue**: `SymbolKind::Function` is explicitly not supported: `Err("JIT compilation of custom Functions is not yet supported".to_owned())`.
- **Impact**: Limits the ability for users to extend the symbolic engine with new mathematical functions.
