# Module Review: `jit` (Post-Upgrade)

## 1. Performance & Memory

### 1.1 Effective Resource Reuse
The `JitCompiler` now correctly reuses `work_stack` and `work_values` buffers, eliminating per-compilation allocation overhead for the iterative walker. The `RssnJitContext` in the FFI layer also addresses the expensive ISA initialization issue.

### 1.2 Branch-Free Division
The use of `select` in `emit_operator` for division by zero ensures that the generated code is branch-free and numerically consistent with standard floating-point behavior (returning `NaN`).

## 2. Dead Code & Functionality

### 2.1 Unfinished `CustomRule`
The `CustomRule` struct in `src/jit/custom.rs` is currently **dead code**. It is defined but not referenced anywhere in the `JitCompiler` or the `codegen` logic. The "User-defined pattern-rewrite derivation rules" mentioned in the module header are not implemented.

## 3. Extensibility

### 3.1 Limited Custom Function Support
`register_custom_function` only supports `extern "C" fn(f64) -> f64`. There is no support for functions with multiple arguments or specialized signatures (e.g. SIMD vectors) without modifying the `emit_one_node` logic.

### 3.2 Closed Peephole Pass
The `emit_operator` peephole simplifications are hardcoded. Users cannot define their own IR-level folding rules (e.g. `x * 2.0 -> x + x`).

## 4. Suggestions
- Implement the `CustomRule` logic or remove the shell if it was a discarded idea.
- Provide a way to register custom IR emission handlers for specific `SymbolKind::Function` IDs to enable true extensibility.
- Expand the peephole pass to be more comprehensive or programmable.
