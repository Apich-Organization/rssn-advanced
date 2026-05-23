# Module Review: `ffi` (Phase 4 Audit)

## 1. Design Integrity

### 1.1 Compiler Re-initialization
`rssn_dag_compile` still instantiates a new `JitCompiler` for every call.
- **Sharp Question:** We have `jit_context.rs` with a persistent context. Why is the main `c_api.rs` entry point still ignoring it? Are we forcing C users to learn two different ways to compile just to get decent performance?

## 2. Sharp Questions

### 2.1 Error Propagation
- **Sharp Question:** We have a rich `RssnStatus` enum(which need to be incorperated with the error mudule way), but our internal `error` module uses 7 different enums with `cold_*` constructors. Is the FFI status code a "lossy conversion" of our internal error state, and if so, how does a C developer debug a `CompilationError` without the internal context?
