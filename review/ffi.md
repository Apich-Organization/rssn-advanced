# Module Review: `ffi` (Phase 5 Audit)

## 2. Design Integrity

### 2.1 Compiler Re-initialization
- **Sharp Question:** `rssn_dag_compile` still instantiates a new `JitCompiler` for every call, ignoring the `RssnJitContext`. Why are we keeping a known inefficient entry point in our public API? Is it purely for users who "don't care about performance," and if so, is that the right audience for a library called `rssn-advanced`?

## 3. Sharp Questions

### 3.1 Error Context
- **Sharp Question:** We have a rich `RssnStatus` enum, but our internal `error` module uses 7 different enums with `cold_*` constructors. Is the FFI status code a "lossy conversion" of our internal error state, and if so, how does a C developer debug a `CompilationError` without the internal context?
