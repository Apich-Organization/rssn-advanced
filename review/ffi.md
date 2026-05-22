# Module Review: `ffi`

## 1. Performance Issues (High Severity)

### 1.1 Excessive Compiler Initialization
In `rssn_dag_compile`, a new `JitCompiler` is instantiated for every single call:
```rust
let mut compiler = crate::jit::compiler::JitCompiler::new();
```
`JitCompiler::new()` is an expensive operation that involves hardware feature detection and Cranelift target ISA initialization. This makes the FFI compilation path orders of magnitude slower than necessary.

### 1.2 Redundant AST Conversion
`rssn_dag_compile` always converts the DAG to an AST before compiling:
```rust
let ast = crate::ast::convert::dag_to_ast(builder_ref.arena(), root_id);
```
Given the identified performance issues in the AST conversion (per-node allocations), this adds significant latency to the FFI boundary.

## 2. Security & Safety Issues

### 2.1 Use-After-Free in Async API
`rssn_dag_simplify_async` captures a raw pointer to `DagBuilder` and dereferences it on a background fiber. There is no mechanism to prevent the C caller from freeing the `DagBuilder` (via `rssn_dag_free`) while the async task is still running, leading to a guaranteed Use-After-Free (UAF) and potential memory corruption or process crash.

### 2.2 Unsafe Pointer Dereferencing
Multiple functions (e.g., `rssn_dag_variable`) use `unsafe { &mut *builder }` and `unsafe { CStr::from_ptr(name) }` without validating that the pointers are valid beyond a simple null check. While expected in C FFI, the lack of safety documentation for the C caller is a concern.

## 3. Engineering Standards

### 3.1 Inconsistent v2 API Coverage
The "v2" status-returning API is only partially implemented. Critical functions like `rssn_dag_compile` and `rssn_dag_execute` do not have v2 equivalents, leading to an inconsistent experience for C developers who must mix-and-match error handling styles (sentinel values vs status codes).

## 4. Suggestions
- Provide a persistent `JitContext` or `Compiler` handle in the C API to avoid re-initializing the Cranelift environment on every call.
- Implement reference counting or a "join" mechanism for the async API to ensure the `DagBuilder` remains valid until all tasks are complete.
- Complete the v2 API coverage for all FFI functions.
- Consider providing a way to compile directly from the DAG or a more efficient intermediate form.
