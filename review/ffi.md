# Module Review: `ffi` (Post-Upgrade)

## 1. Performance & Memory

### 1.1 JIT Context Persistence
The introduction of `RssnJitContext` and `rssn_dag_compile_with_ctx` is a major performance win for FFI users, as it amortizes the high cost of Cranelift initialization across many compilation calls.

## 2. Dead Code & Unfinished Updates

### 2.1 Persistent Use-After-Free in v1
The `rssn_dag_simplify_async` (v1) function still captures a raw pointer and returns immediately. While `v2` provides a safe handle approach, the presence of the unsafe v1 variant without sufficient protection (e.g. `Arc`) remains a security risk for legacy integrations.

### 2.2 Incomplete V2 Transition
Multiple functions (`rssn_dag_compile`, `rssn_dag_add`, etc.) still exist in their v1 forms which return sentinels like `u32::MAX`. While `v2` variants are being added, the migration is incomplete, leading to an inconsistent API surface.

## 3. Extensibility

### 3.1 Closed Handle System
The FFI is strictly limited to the `DagBuilder` and `JitContext`. There is no way for users to plug in their own handles or extend the FFI with custom types without writing significant Rust glue.

## 4. Suggestions
- Complete the V2 transition for all FFI entry points and consider deprecating the V1 functions that use sentinel error values.
- Use `Arc<DagBuilder>` in the async API to provide genuine safety for the v1 variant, or remove it entirely in favor of the joinable handle.
