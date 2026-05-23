# Module Review: `ffi` (Phase 5 Audit)

## 2. Design Integrity

### 2.1 Compiler Re-initialization

**Answer:** Fixed in Phase 4 — `GLOBAL_JIT_CTX: OnceLock<Mutex<RssnJitContext>>` was added in `src/ffi/jit_context.rs`. Both `rssn_dag_compile` and `rssn_dag_compile_v2` now route through `crate::ffi::jit_context::global_jit_ctx()`, locking the process-level `RssnJitContext` for the duration of each compile call. Cranelift initialization happens exactly once per process. The issue is resolved.

## 3. Sharp Questions

### 3.1 Error Context

**Answer:** The `RssnStatus` code is intentionally lossy — it is a C-ABI enum that must cross the FFI boundary without allocating. The internal `JitError`, `DagError`, etc. are Rust types that cannot be returned as-is over C. The correct pattern is a thread-local last-error string: every FFI error path stores a human-readable message in `thread_local! { static LAST_ERROR: RefCell<String> }` in `src/ffi/c_api.rs`, and a new export `pub extern "C" fn rssn_last_error_message() -> *const c_char` returns a pointer to that string. A C developer calls `rssn_dag_compile(...)`, checks the status, and if not `Success`, calls `rssn_last_error_message()` to get the diagnostic string. This follows the `GetLastError`/`strerror` convention used by most C system APIs. Implementation is straightforward and targeted for Phase 6.
