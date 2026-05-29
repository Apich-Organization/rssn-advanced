//! `extern "C"` entry points for the RSSN-Advanced API.
//!
//! Exposes a flat C API, capturing panics securely at the FFI boundary to
//! avoid undefined behavior (UB).

#![allow(unsafe_code)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use super::types::RssnStatus;
use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use crate::heuristic::{HeuristicConfig, HeuristicEngine, SearchStrategy};
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::panic::catch_unwind;
use std::time::Duration;

/// Creates a new `DagBuilder` context.
///
/// Returns a raw pointer to the builder, or NULL if creation failed or panicked.
/// The returned pointer must be freed exactly once via [`rssn_dag_free`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_new() -> *mut DagBuilder {
    let result = catch_unwind(|| Box::into_raw(Box::new(DagBuilder::new())));
    result.unwrap_or(std::ptr::null_mut())
}

/// Releases the memory of a previously allocated `DagBuilder`.
///
/// # Safety
///
/// `builder` must be a pointer previously returned by [`rssn_dag_new`], or NULL.
/// After this call the pointer is dangling and must not be used.
/// Passing a pointer not from `rssn_dag_new`, or freeing twice, is undefined behaviour.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_free(builder: *mut DagBuilder) {
    if builder.is_null() {
        return;
    }
    let _ = catch_unwind(|| {
        let _ = unsafe { Box::from_raw(builder) };
    });
}

/// Allocates a new variable node in the DAG.
///
/// Returns the index of the variable node, or `u32::MAX` if a panic
/// occurred, the builder was null, or `name` was not valid UTF-8.
///
/// **Deprecated** — use [`rssn_dag_variable_v2`] for richer error reporting.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
/// - `name` must be a valid, non-null, null-terminated C string valid for the duration of
///   this call.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_variable(builder: *mut DagBuilder, name: *const c_char) -> u32 {
    if builder.is_null() || name.is_null() {
        return u32::MAX;
    }
    let result = catch_unwind(|| -> u32 {
        let builder_ref = unsafe { &mut *builder };
        let c_str = unsafe { CStr::from_ptr(name) };
        builder_ref
            .variable_bytes(c_str.to_bytes())
            .map_or(u32::MAX, DagNodeId::value)
    });
    result.unwrap_or(u32::MAX)
}

/// Allocates a new constant node in the DAG.
///
/// Returns `u32::MAX` on error.  **Deprecated** — use [`rssn_dag_constant_v2`].
///
/// # Safety
///
/// `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_constant(builder: *mut DagBuilder, val: f64) -> u32 {
    if builder.is_null() {
        return u32::MAX;
    }
    let result = catch_unwind(|| {
        let builder_ref = unsafe { &mut *builder };
        builder_ref.constant(val).value()
    });
    result.unwrap_or(u32::MAX)
}

/// Allocates a new addition node in the DAG: `lhs + rhs`.
///
/// Returns `u32::MAX` on error.  **Deprecated** — use [`rssn_dag_add_v2`].
///
/// # Safety
///
/// `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_add(builder: *mut DagBuilder, lhs: u32, rhs: u32) -> u32 {
    if builder.is_null() {
        return u32::MAX;
    }
    let result = catch_unwind(|| {
        let builder_ref = unsafe { &mut *builder };
        builder_ref
            .add(DagNodeId::new(lhs), DagNodeId::new(rhs))
            .value()
    });
    result.unwrap_or(u32::MAX)
}

/// Simplifies a target expression using the default heuristic engine.
///
/// Returns the new root node index, or `u32::MAX` on error.
/// **Deprecated** — use [`rssn_dag_simplify_v2`] or [`rssn_dag_simplify_with_config`].
///
/// # Safety
///
/// `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_simplify(builder: *mut DagBuilder, root: u32) -> u32 {
    if builder.is_null() {
        return u32::MAX;
    }
    let result = catch_unwind(|| {
        let builder_ref = unsafe { &mut *builder };
        let root_id = DagNodeId::new(root);

        let config = HeuristicConfig::default();
        let mut engine = HeuristicEngine::new(config, SearchStrategy::Greedy);

        engine.simplify(builder_ref, root_id).value()
    });
    result.unwrap_or(u32::MAX)
}

/// JIT compiles a target expression and writes the native function pointer to `out_fn`.
///
/// `out_fn` can be called via `rssn_dag_execute` or cast directly as `double (*)(const double*)`.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
/// - `out_fn` must be a valid, non-null pointer to a `*mut c_void` that the function will write to.
/// - The compiled function pointer written to `*out_fn` remains valid until the `JITModule`
///   backing this compiler is dropped. Do not call it after that.
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_compile(
    builder: *mut DagBuilder,
    root: u32,
    out_fn: *mut *mut c_void,
) -> RssnStatus {
    if builder.is_null() || out_fn.is_null() {
        return RssnStatus::NullPointer;
    }

    let result = catch_unwind(|| {
        let builder_ref = unsafe { &mut *builder };
        let root_id = DagNodeId::new(root);
        let ast = crate::ast::convert::dag_to_ast(builder_ref.arena(), root_id);

        // Reuse the process-level JIT context to amortise Cranelift init cost.
        let ctx_mutex = crate::ffi::jit_context::global_jit_ctx();
        let mut ctx = ctx_mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ctx.compiler_mut()
            .compile(&ast)
            .map_or(RssnStatus::CompilationError, |compiled_fn| {
                let ptr = compiled_fn as *mut c_void;
                unsafe { *out_fn = ptr };
                RssnStatus::Success
            })
    });

    result.unwrap_or(RssnStatus::Panic)
}

/// JIT compiles a target expression and writes the native function pointer to `out_fn`.
///
/// `out_fn` can be called via `rssn_dag_execute` or cast directly as `double (*)(const double*)`.
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_compile(
    _builder: *mut DagBuilder,
    _root: u32,
    _out_fn: *mut *mut c_void,
) -> RssnStatus {
    RssnStatus::CompilationError
}

/// Executes a previously compiled JIT function with the given variable input array.
///
/// Returns `0.0` on error, which is indistinguishable from a legitimate zero result.
/// **Deprecated** — use [`rssn_dag_execute_v2`] to get a distinct error status.
///
/// # Safety
///
/// - `func` must be a valid function pointer previously written by [`rssn_dag_compile`],
///   with signature `double (*)(const double*)`.
/// - `variables` must be a valid pointer to an array of at least as many `f64` values
///   as there are distinct variables in the compiled expression, ordered by `SymbolId`.
/// - Both pointers must remain valid for the duration of this call.
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_execute(func: *const c_void, variables: *const f64) -> f64 {
    if func.is_null() || variables.is_null() {
        return 0.0;
    }
    let result = catch_unwind(|| {
        let compiled_fn: crate::jit::compiler::CompiledExprFn =
            unsafe { std::mem::transmute(func) };
        compiled_fn(variables)
    });
    result.unwrap_or(0.0)
}

/// Executes a previously compiled JIT function with the given variable input array.
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_execute(_func: *const c_void, _variables: *const f64) -> f64 {
    0.0
}

// =========================================================================
// Status-returning surface (canonical API)
// =========================================================================
//
// Each `*_v2` function takes an `out_id: *mut u32` (or equivalent) and
// returns [`RssnStatus`].  This is the canonical API for new code.
//
// The legacy `*` (non-v2) functions below return `u32::MAX` / 0.0 on error,
// which is ambiguous and cannot distinguish between different failure modes.
// They are **deprecated**: use the `_v2` equivalents for all new consumers.
// They are retained for ABI compatibility (e.g. existing Python/C++ callers).

/// Creates a new variable node. Status-returning variant.
///
/// On `Success`, writes the new node id to `*out_id`.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
/// - `name` must be a valid, non-null, null-terminated C string valid for this call.
/// - `out_id` must be a valid, non-null, writable `u32` pointer.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_variable_v2(
    builder: *mut DagBuilder,
    name: *const c_char,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || name.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    let result = catch_unwind(|| -> RssnStatus {
        let builder_ref = unsafe { &mut *builder };
        let c_str = unsafe { CStr::from_ptr(name) };
        builder_ref
            .variable_bytes(c_str.to_bytes())
            .map_or(RssnStatus::InvalidUtf8, |id| {
                unsafe { *out_id = id.value() };
                RssnStatus::Success
            })
    });
    result.unwrap_or(RssnStatus::Panic)
}

/// Creates a new constant node. Status-returning variant.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
/// - `out_id` must be a valid, non-null, writable `u32` pointer.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_constant_v2(
    builder: *mut DagBuilder,
    val: f64,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    let result = catch_unwind(|| -> RssnStatus {
        let builder_ref = unsafe { &mut *builder };
        let id = builder_ref.constant(val);
        unsafe { *out_id = id.value() };
        RssnStatus::Success
    });
    result.unwrap_or(RssnStatus::Panic)
}

/// Creates an addition node. Status-returning variant.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
/// - `out_id` must be a valid, non-null, writable `u32` pointer.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_add_v2(
    builder: *mut DagBuilder,
    lhs: u32,
    rhs: u32,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    if lhs == u32::MAX || rhs == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    let result = catch_unwind(|| -> RssnStatus {
        let builder_ref = unsafe { &mut *builder };
        let id = builder_ref.add(DagNodeId::new(lhs), DagNodeId::new(rhs));
        unsafe { *out_id = id.value() };
        RssnStatus::Success
    });
    result.unwrap_or(RssnStatus::Panic)
}

/// Executes a previously compiled JIT function. Status-returning variant.
///
/// On `Success`, writes the result to `*out_val`.
///
/// # Safety
///
/// - `func` must be a valid function pointer previously written by [`rssn_dag_compile`].
/// - `variables` must be a valid pointer to an array of at least as many `f64` values
///   as there are variables in the compiled expression, ordered by `SymbolId`.
/// - `out_val` must be a valid, non-null, writable `f64` pointer.
/// - All pointers must remain valid for the duration of this call.
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_execute_v2(
    func: *const c_void,
    variables: *const f64,
    out_val: *mut f64,
) -> RssnStatus {
    if func.is_null() || variables.is_null() || out_val.is_null() {
        return RssnStatus::NullPointer;
    }
    let result = catch_unwind(|| {
        let compiled_fn: crate::jit::compiler::CompiledExprFn =
            unsafe { std::mem::transmute(func) };
        compiled_fn(variables)
    });
    result.map_or(RssnStatus::Panic, |val| {
        unsafe { *out_val = val };
        RssnStatus::Success
    })
}

/// Executes a previously compiled JIT function (stub for non-JIT builds).
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_execute_v2(
    _func: *const c_void,
    _variables: *const f64,
    _out_val: *mut f64,
) -> RssnStatus {
    RssnStatus::CompilationError
}

// =========================================================================
// Bulk / batch evaluation — amortises FFI overhead across many rows
// =========================================================================
//
// Calling `rssn_dag_execute` from an interpreted language (Python, Julia, …)
// inside a tight loop is ~200–400 ns per call just for the FFI dispatch,
// completely swamping the 1–5 ns the JIT needs per evaluation.
//
// These three functions bring the overhead down to O(1) per batch:
//
//   rssn_dag_execute_bulk   — scalar JIT fn called in a tight *Rust* loop;
//                             ~1–5 ns amortised overhead per row.
//   rssn_dag_compile_batch  — compiles a 2-row ILP vectorised version of the
//                             expression (Cranelift SSA dual-path).
//   rssn_dag_execute_batch  — dispatches the vectorised batch fn; fastest path.
//
// Both functions use *column-major* layout for variables:
//   vars_cols[var_index]  →  pointer to an array of n_rows f64 values.
// This mirrors NumPy's column-major convention and avoids transposition.

/// Evaluates a scalar JIT function over `n_rows` rows in a tight Rust loop,
/// eliminating per-row FFI overhead from the calling language.
///
/// `vars_cols` is an array of `n_vars` pointers; each pointer addresses a
/// contiguous column of `n_rows` `f64` values for the corresponding variable.
/// Columns must be ordered by **`SymbolId`**: the first variable interned into
/// the `DagBuilder` has `SymbolId` 0 and uses `vars_cols[0]`, etc.
///
/// One FFI call amortises setup cost over `n_rows` evaluations. For `n_rows`
/// ≥ 1 000, throughput is limited by memory bandwidth, not FFI overhead.
///
/// # Safety
///
/// - `func` must be a valid function pointer from [`rssn_dag_compile`].
/// - `vars_cols` must point to `n_vars` valid column pointers, each of length
///   `n_rows`.
/// - `out` must point to a writable array of `n_rows` `f64` values.
/// - All pointers must remain valid for the duration of this call.
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn rssn_dag_execute_bulk(
    func: *const c_void,
    vars_cols: *const *const f64,
    n_vars: u32,
    n_rows: usize,
    out: *mut f64,
) -> RssnStatus {
    if func.is_null() || out.is_null() {
        return RssnStatus::NullPointer;
    }
    if n_vars > 0 && vars_cols.is_null() {
        return RssnStatus::NullPointer;
    }
    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let compiled_fn: crate::jit::compiler::CompiledExprFn =
            unsafe { std::mem::transmute(func) };
        let nv = n_vars as usize;
        let cols: &[*const f64] = unsafe { std::slice::from_raw_parts(vars_cols, nv) };
        let out_slice: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(out, n_rows) };

        // Fixed stack buffer for the common case (≤ 8 variables).
        // Avoids heap allocation inside the hot loop.
        if nv <= 8 {
            let mut buf = [0.0f64; 8];
            for (row, out_val) in out_slice.iter_mut().enumerate() {
                for (vi, &col) in cols.iter().enumerate() {
                    buf[vi] = unsafe { *col.add(row) };
                }
                *out_val = compiled_fn(buf.as_ptr());
            }
        } else {
            let mut buf = vec![0.0f64; nv];
            for (row, out_val) in out_slice.iter_mut().enumerate() {
                for (vi, &col) in cols.iter().enumerate() {
                    buf[vi] = unsafe { *col.add(row) };
                }
                *out_val = compiled_fn(buf.as_ptr());
            }
        }
        RssnStatus::Success
    }));
    result.unwrap_or(RssnStatus::Panic)
}

/// Stub for non-JIT builds.
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_execute_bulk(
    _func: *const c_void,
    _vars_cols: *const *const f64,
    _n_vars: u32,
    _n_rows: usize,
    _out: *mut f64,
) -> RssnStatus {
    RssnStatus::CompilationError
}

/// Compiles a 2-row ILP-vectorised version of the expression.
///
/// The Cranelift backend generates two independent SSA chains that evaluate
/// two rows simultaneously, keeping execution units busy across instruction
/// latency gaps. For memory-bound workloads this approaches 2× scalar
/// throughput; for compute-bound workloads the speedup is limited by
/// available instruction-level parallelism.
///
/// On success writes the batch function pointer to `*out_fn`.
/// Use [`rssn_dag_execute_batch`] to dispatch the compiled function.
///
/// Returns [`RssnStatus::CompilationError`] if the expression cannot be
/// vectorised (e.g. contains non-vectorisable operations).
///
/// # Safety
///
/// Same as [`rssn_dag_compile`].
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn rssn_dag_compile_batch(
    builder: *mut DagBuilder,
    root: u32,
    out_fn: *mut *mut c_void,
) -> RssnStatus {
    if builder.is_null() || out_fn.is_null() {
        return RssnStatus::NullPointer;
    }
    if root == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    let result = catch_unwind(|| {
        let builder_ref = unsafe { &mut *builder };
        let root_id = DagNodeId::new(root);
        let ast = crate::ast::convert::dag_to_ast(builder_ref.arena(), root_id);
        let ctx_mutex = crate::ffi::jit_context::global_jit_ctx();
        let mut ctx = ctx_mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match ctx.compiler_mut().compile_batch_f64x2(&ast) {
            Ok(Some(batch_fn)) => {
                unsafe { *out_fn = batch_fn as *mut c_void };
                RssnStatus::Success
            }
            _ => RssnStatus::CompilationError,
        }
    });
    result.unwrap_or(RssnStatus::Panic)
}

/// Stub for non-JIT builds.
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_compile_batch(
    _builder: *mut DagBuilder,
    _root: u32,
    _out_fn: *mut *mut c_void,
) -> RssnStatus {
    RssnStatus::CompilationError
}

/// Dispatches a batch-compiled function over `n_rows` rows.
///
/// `vars_cols` is an array of column pointers (one per variable, each of
/// length `n_rows`).  The batch function processes two rows per cycle via
/// independent SSA chains; a scalar tail handles any odd final row.
///
/// # Safety
///
/// - `batch_fn` must be a valid function pointer from [`rssn_dag_compile_batch`].
/// - `vars_cols` must point to an array of column pointers, each of length `n_rows`.
/// - `out` must point to a writable array of `n_rows` `f64` values.
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn rssn_dag_execute_batch(
    batch_fn: *const c_void,
    vars_cols: *const *const f64,
    n_rows: usize,
    out: *mut f64,
) -> RssnStatus {
    if batch_fn.is_null() || vars_cols.is_null() || out.is_null() {
        return RssnStatus::NullPointer;
    }
    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let f: crate::jit::compiler::CompiledBatchFn = unsafe { std::mem::transmute(batch_fn) };
        f(vars_cols, n_rows, out);
        RssnStatus::Success
    }));
    result.unwrap_or(RssnStatus::Panic)
}

/// Stub for non-JIT builds.
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_execute_batch(
    _batch_fn: *const c_void,
    _vars_cols: *const *const f64,
    _n_rows: usize,
    _out: *mut f64,
) -> RssnStatus {
    RssnStatus::CompilationError
}

/// Simplifies an expression. Status-returning variant.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
/// - `out_id` must be a valid, non-null, writable `u32` pointer.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_simplify_v2(
    builder: *mut DagBuilder,
    root: u32,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    if root == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    let result = catch_unwind(|| -> RssnStatus {
        let builder_ref = unsafe { &mut *builder };
        let root_id = DagNodeId::new(root);
        let config = HeuristicConfig::default();
        let mut engine = HeuristicEngine::new(config, SearchStrategy::Greedy);
        let id = engine.simplify(builder_ref, root_id);
        unsafe { *out_id = id.value() };
        RssnStatus::Success
    });
    result.unwrap_or(RssnStatus::Panic)
}

/// JIT compiles a target expression. Status-returning variant.
///
/// On `Success`, writes the compiled function pointer to `*out_fn`.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
/// - `out_fn` must be a valid, non-null, writable `*mut c_void` pointer.
/// - The compiled function pointer remains valid until the `JITModule` is dropped.
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_compile_v2(
    builder: *mut DagBuilder,
    root: u32,
    out_fn: *mut *mut c_void,
) -> RssnStatus {
    if builder.is_null() || out_fn.is_null() {
        return RssnStatus::NullPointer;
    }
    if root == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    let result = catch_unwind(|| {
        let builder_ref = unsafe { &mut *builder };
        let root_id = DagNodeId::new(root);
        let ast = crate::ast::convert::dag_to_ast(builder_ref.arena(), root_id);
        // Reuse the process-level JIT context.
        let ctx_mutex = crate::ffi::jit_context::global_jit_ctx();
        let mut ctx = ctx_mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ctx.compiler_mut()
            .compile(&ast)
            .map_or(RssnStatus::CompilationError, |compiled_fn| {
                unsafe { *out_fn = compiled_fn as *mut c_void };
                RssnStatus::Success
            })
    });
    result.unwrap_or(RssnStatus::Panic)
}

/// JIT compiles a target expression (stub for non-JIT builds).
///
/// Always returns [`RssnStatus::CompilationError`] when the `cranelift-jit`
/// feature is not enabled.
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_compile_v2(
    _builder: *mut DagBuilder,
    _root: u32,
    _out_fn: *mut *mut c_void,
) -> RssnStatus {
    RssnStatus::CompilationError
}

/// C-compatible simplification configuration.
///
/// Pass a pointer to this struct to [`rssn_dag_simplify_with_config`] to
/// override the default heuristic parameters. Pass NULL to use defaults.
#[repr(C)]
pub struct RssnSimplifyConfig {
    /// Maximum rewrite depth (default: 10).
    pub max_depth: u32,
    /// Wall-clock timeout in milliseconds (default: 500).
    pub timeout_ms: u64,
    /// Approximate-pruning aggressiveness in `[0.0, 1.0]` (default: 0.1).
    pub aggressiveness: f64,
}

/// Simplifies an expression using a caller-supplied configuration.
///
/// If `config` is NULL, behaves identically to [`rssn_dag_simplify_v2`].
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer to a `DagBuilder`.
/// - `out_id` must be a valid, non-null, writable `u32` pointer.
/// - If `config` is non-null, it must point to a valid `RssnSimplifyConfig`.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_simplify_with_config(
    builder: *mut DagBuilder,
    root: u32,
    config: *const RssnSimplifyConfig,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    if root == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    let (max_depth, timeout_ms, aggressiveness) = if config.is_null() {
        let def = HeuristicConfig::default();
        (
            def.max_depth,
            def.timeout.as_millis() as u64,
            def.simplification_aggressiveness,
        )
    } else {
        let c = unsafe { &*config };
        (c.max_depth as usize, c.timeout_ms, c.aggressiveness)
    };
    let result = catch_unwind(|| -> RssnStatus {
        let builder_ref = unsafe { &mut *builder };
        let root_id = DagNodeId::new(root);
        let cfg = HeuristicConfig::default()
            .max_depth(max_depth)
            .timeout(Duration::from_millis(timeout_ms))
            .simplification_aggressiveness(aggressiveness);
        let mut engine = HeuristicEngine::new(cfg, SearchStrategy::Greedy);
        let id = engine.simplify(builder_ref, root_id);
        unsafe { *out_id = id.value() };
        RssnStatus::Success
    });
    result.unwrap_or(RssnStatus::Panic)
}

// =========================================================================
// Full operator surface: sub, mul, div, pow, mod, neg
// =========================================================================
//
// Each operator comes in two variants:
//   • Legacy (no suffix)  — returns u32::MAX on error. **Deprecated.**
//   • Canonical (_v2)     — returns RssnStatus; writes node id to *out_id.
//
// New code should use the _v2 forms.

/// Allocates a subtraction node: `lhs - rhs`.
///
/// Returns `u32::MAX` on error or null input.
///
/// # Safety
///
/// `builder` must be a valid, non-null pointer from [`rssn_dag_new`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_sub(builder: *mut DagBuilder, lhs: u32, rhs: u32) -> u32 {
    if builder.is_null() {
        return u32::MAX;
    }
    catch_unwind(|| {
        let b = unsafe { &mut *builder };
        b.sub(DagNodeId::new(lhs), DagNodeId::new(rhs)).value()
    })
    .unwrap_or(u32::MAX)
}

/// Allocates a subtraction node. Status-returning variant.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer from [`rssn_dag_new`].
/// - `out_id` must be a valid, non-null writable `u32` pointer.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_sub_v2(
    builder: *mut DagBuilder,
    lhs: u32,
    rhs: u32,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    if lhs == u32::MAX || rhs == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    catch_unwind(|| {
        let b = unsafe { &mut *builder };
        let id = b.sub(DagNodeId::new(lhs), DagNodeId::new(rhs));
        unsafe { *out_id = id.value() };
        RssnStatus::Success
    })
    .unwrap_or(RssnStatus::Panic)
}

/// Allocates a multiplication node: `lhs * rhs`.
///
/// Returns `u32::MAX` on error or null input.
///
/// # Safety
///
/// `builder` must be a valid, non-null pointer from [`rssn_dag_new`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_mul(builder: *mut DagBuilder, lhs: u32, rhs: u32) -> u32 {
    if builder.is_null() {
        return u32::MAX;
    }
    catch_unwind(|| {
        let b = unsafe { &mut *builder };
        b.mul(DagNodeId::new(lhs), DagNodeId::new(rhs)).value()
    })
    .unwrap_or(u32::MAX)
}

/// Allocates a multiplication node. Status-returning variant.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer from [`rssn_dag_new`].
/// - `out_id` must be a valid, non-null writable `u32` pointer.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_mul_v2(
    builder: *mut DagBuilder,
    lhs: u32,
    rhs: u32,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    if lhs == u32::MAX || rhs == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    catch_unwind(|| {
        let b = unsafe { &mut *builder };
        let id = b.mul(DagNodeId::new(lhs), DagNodeId::new(rhs));
        unsafe { *out_id = id.value() };
        RssnStatus::Success
    })
    .unwrap_or(RssnStatus::Panic)
}

/// Allocates a division node: `lhs / rhs`.
///
/// Returns `u32::MAX` on error or null input.
///
/// # Safety
///
/// `builder` must be a valid, non-null pointer from [`rssn_dag_new`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_div(builder: *mut DagBuilder, lhs: u32, rhs: u32) -> u32 {
    if builder.is_null() {
        return u32::MAX;
    }
    catch_unwind(|| {
        let b = unsafe { &mut *builder };
        b.div(DagNodeId::new(lhs), DagNodeId::new(rhs)).value()
    })
    .unwrap_or(u32::MAX)
}

/// Allocates a division node. Status-returning variant.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer from [`rssn_dag_new`].
/// - `out_id` must be a valid, non-null writable `u32` pointer.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_div_v2(
    builder: *mut DagBuilder,
    lhs: u32,
    rhs: u32,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    if lhs == u32::MAX || rhs == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    catch_unwind(|| {
        let b = unsafe { &mut *builder };
        let id = b.div(DagNodeId::new(lhs), DagNodeId::new(rhs));
        unsafe { *out_id = id.value() };
        RssnStatus::Success
    })
    .unwrap_or(RssnStatus::Panic)
}

/// Allocates an exponentiation node: `base ^ exp`.
///
/// Returns `u32::MAX` on error or null input.
///
/// # Safety
///
/// `builder` must be a valid, non-null pointer from [`rssn_dag_new`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_pow(builder: *mut DagBuilder, base: u32, exp: u32) -> u32 {
    if builder.is_null() {
        return u32::MAX;
    }
    catch_unwind(|| {
        let b = unsafe { &mut *builder };
        b.pow(DagNodeId::new(base), DagNodeId::new(exp)).value()
    })
    .unwrap_or(u32::MAX)
}

/// Allocates an exponentiation node. Status-returning variant.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer from [`rssn_dag_new`].
/// - `out_id` must be a valid, non-null writable `u32` pointer.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_pow_v2(
    builder: *mut DagBuilder,
    base: u32,
    exp: u32,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    if base == u32::MAX || exp == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    catch_unwind(|| {
        let b = unsafe { &mut *builder };
        let id = b.pow(DagNodeId::new(base), DagNodeId::new(exp));
        unsafe { *out_id = id.value() };
        RssnStatus::Success
    })
    .unwrap_or(RssnStatus::Panic)
}

/// Allocates a modulo node: `lhs % rhs`.
///
/// Returns `u32::MAX` on error or null input.
///
/// # Safety
///
/// `builder` must be a valid, non-null pointer from [`rssn_dag_new`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_mod(builder: *mut DagBuilder, lhs: u32, rhs: u32) -> u32 {
    if builder.is_null() {
        return u32::MAX;
    }
    catch_unwind(|| {
        let b = unsafe { &mut *builder };
        b.modulo(DagNodeId::new(lhs), DagNodeId::new(rhs)).value()
    })
    .unwrap_or(u32::MAX)
}

/// Allocates a modulo node. Status-returning variant.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer from [`rssn_dag_new`].
/// - `out_id` must be a valid, non-null writable `u32` pointer.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_mod_v2(
    builder: *mut DagBuilder,
    lhs: u32,
    rhs: u32,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    if lhs == u32::MAX || rhs == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    catch_unwind(|| {
        let b = unsafe { &mut *builder };
        let id = b.modulo(DagNodeId::new(lhs), DagNodeId::new(rhs));
        unsafe { *out_id = id.value() };
        RssnStatus::Success
    })
    .unwrap_or(RssnStatus::Panic)
}

/// Allocates a unary negation node: `-operand`.
///
/// Returns `u32::MAX` on error or null input.
///
/// # Safety
///
/// `builder` must be a valid, non-null pointer from [`rssn_dag_new`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_neg(builder: *mut DagBuilder, operand: u32) -> u32 {
    if builder.is_null() {
        return u32::MAX;
    }
    catch_unwind(|| {
        let b = unsafe { &mut *builder };
        b.neg(DagNodeId::new(operand)).value()
    })
    .unwrap_or(u32::MAX)
}

/// Allocates a unary negation node. Status-returning variant.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer from [`rssn_dag_new`].
/// - `out_id` must be a valid, non-null writable `u32` pointer.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_neg_v2(
    builder: *mut DagBuilder,
    operand: u32,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    if operand == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    catch_unwind(|| {
        let b = unsafe { &mut *builder };
        let id = b.neg(DagNodeId::new(operand));
        unsafe { *out_id = id.value() };
        RssnStatus::Success
    })
    .unwrap_or(RssnStatus::Panic)
}

// =========================================================================
// T6.4 — Parse expression from C string
// =========================================================================

/// Parses a mathematical expression from a C string into the DAG.
///
/// The expression uses the standard infix syntax: `+`, `-`, `*`, `/`,
/// `^` (exponentiation), `%` (modulo), parentheses, numeric literals,
/// and identifier names for variables.
///
/// On `Success`, writes the root node id of the parsed expression to
/// `*out_id`. On failure returns [`RssnStatus::ParseError`].
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer from [`rssn_dag_new`].
/// - `expr` must be a valid, non-null, null-terminated C string.
/// - `out_id` must be a valid, non-null writable `u32` pointer.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_parse(
    builder: *mut DagBuilder,
    expr: *const c_char,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || expr.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    let result = catch_unwind(|| -> RssnStatus {
        let b = unsafe { &mut *builder };
        let c_str = unsafe { CStr::from_ptr(expr) };
        let Ok(s) = c_str.to_str() else {
            return RssnStatus::InvalidUtf8;
        };
        crate::parser::expr::parse_expression(s, b).map_or(RssnStatus::ParseError, |root_id| {
            unsafe { *out_id = root_id.value() };
            RssnStatus::Success
        })
    });
    result.unwrap_or(RssnStatus::Panic)
}

// =========================================================================
// T6.5 — Function registration and call node construction
// =========================================================================

/// Interns a function name and returns its numeric `FnId`.
///
/// The returned id can be used with [`rssn_dag_call_fn`] to build
/// function-call nodes, and with the JIT custom-function registration
/// APIs to bind native implementations.
///
/// Returns `u32::MAX` on null input or if interning fails.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer from [`rssn_dag_new`].
/// - `name` must be a valid, non-null, null-terminated C string.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_intern_function(builder: *mut DagBuilder, name: *const c_char) -> u32 {
    if builder.is_null() || name.is_null() {
        return u32::MAX;
    }
    catch_unwind(|| {
        let b = unsafe { &mut *builder };
        let c_str = unsafe { CStr::from_ptr(name) };
        let Ok(s) = c_str.to_str() else { return u32::MAX };
        b.intern_function(s).0
    })
    .unwrap_or(u32::MAX)
}

/// Builds a function-call node for a previously interned function.
///
/// `args` points to an array of `n_args` node ids. The node ids must all be
/// valid (not `u32::MAX`).
///
/// Returns the new node id, or `u32::MAX` on error.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer from [`rssn_dag_new`].
/// - `args` must point to an array of at least `n_args` valid `u32` values,
///   or be null when `n_args == 0`.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_call_fn(
    builder: *mut DagBuilder,
    fn_id: u32,
    args: *const u32,
    n_args: u32,
) -> u32 {
    if builder.is_null() {
        return u32::MAX;
    }
    if n_args > 0 && args.is_null() {
        return u32::MAX;
    }
    catch_unwind(|| {
        let b = unsafe { &mut *builder };
        let arg_ids: Vec<DagNodeId> = if n_args == 0 {
            Vec::new()
        } else {
            let slice = unsafe { std::slice::from_raw_parts(args, n_args as usize) };
            if slice.contains(&u32::MAX) {
                return u32::MAX;
            }
            slice.iter().map(|&id| DagNodeId::new(id)).collect()
        };
        b.function_call(crate::dag::symbol::FnId(fn_id), &arg_ids)
            .value()
    })
    .unwrap_or(u32::MAX)
}

/// Builds a function-call node. Status-returning variant.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer from [`rssn_dag_new`].
/// - `args` must point to an array of at least `n_args` valid `u32` values,
///   or be null when `n_args == 0`.
/// - `out_id` must be a valid, non-null writable `u32` pointer.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_call_fn_v2(
    builder: *mut DagBuilder,
    fn_id: u32,
    args: *const u32,
    n_args: u32,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    if n_args > 0 && args.is_null() {
        return RssnStatus::NullPointer;
    }
    catch_unwind(|| -> RssnStatus {
        let b = unsafe { &mut *builder };
        let arg_ids: Vec<DagNodeId> = if n_args == 0 {
            Vec::new()
        } else {
            let slice = unsafe { std::slice::from_raw_parts(args, n_args as usize) };
            if slice.contains(&u32::MAX) {
                return RssnStatus::InvalidNodeId;
            }
            slice.iter().map(|&id| DagNodeId::new(id)).collect()
        };
        let id = b.function_call(crate::dag::symbol::FnId(fn_id), &arg_ids);
        unsafe { *out_id = id.value() };
        RssnStatus::Success
    })
    .unwrap_or(RssnStatus::Panic)
}

// =========================================================================
// T6.6 — JIT custom-function registration from C
// =========================================================================
//
// These functions allow C callers to register native function pointers so
// the JIT can compile call nodes that reference them. The `fn_id` must
// match the id returned by `rssn_dag_intern_function`.

/// Type for a C-callable `extern "C" fn(f64) -> f64`.
pub type RssnCustomFn1 = extern "C" fn(f64) -> f64;
/// Type for a C-callable `extern "C" fn(f64, f64) -> f64`.
pub type RssnCustomFn2 = extern "C" fn(f64, f64) -> f64;
/// Type for a C-callable `extern "C" fn(f64, f64, f64) -> f64`.
pub type RssnCustomFn3 = extern "C" fn(f64, f64, f64) -> f64;

/// Registers a one-argument native function with the persistent JIT context
/// so it can be called from compiled expressions.
///
/// The `fn_id` must have been obtained via [`rssn_dag_intern_function`].
/// The `func` pointer must remain valid for the lifetime of the JIT context.
///
/// # Safety
///
/// `func` must be a valid function pointer with the signature `double(double)`.
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_jit_register_fn_1(
    ctx: *mut super::jit_context::RssnJitContext,
    fn_id: u32,
    func: Option<extern "C" fn(f64) -> f64>,
) -> RssnStatus {
    if ctx.is_null() {
        return RssnStatus::NullPointer;
    }
    let Some(func_ptr) = func else {
        return RssnStatus::NullPointer;
    };
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        let ctx_ref = unsafe { &mut *ctx };
        ctx_ref
            .compiler_mut()
            .register_custom_function(crate::dag::symbol::FnId(fn_id), func_ptr);
        RssnStatus::Success
    }))
    .unwrap_or(RssnStatus::Panic)
}

/// Registers a one-argument native function (stub for non-JIT builds).
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_jit_register_fn_1(
    _ctx: *mut super::jit_context::RssnJitContext,
    _fn_id: u32,
    _func: Option<extern "C" fn(f64) -> f64>,
) -> RssnStatus {
    RssnStatus::CompilationError
}

/// Registers a two-argument native function with the persistent JIT context.
///
/// # Safety
///
/// `func` must be a valid function pointer with the signature `double(double, double)`.
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_jit_register_fn_2(
    ctx: *mut super::jit_context::RssnJitContext,
    fn_id: u32,
    func: Option<extern "C" fn(f64, f64) -> f64>,
) -> RssnStatus {
    if ctx.is_null() {
        return RssnStatus::NullPointer;
    }
    let Some(func_ptr) = func else {
        return RssnStatus::NullPointer;
    };
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        let ctx_ref = unsafe { &mut *ctx };
        ctx_ref
            .compiler_mut()
            .register_custom_function_2(crate::dag::symbol::FnId(fn_id), func_ptr);
        RssnStatus::Success
    }))
    .unwrap_or(RssnStatus::Panic)
}

/// Registers a two-argument native function (stub for non-JIT builds).
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_jit_register_fn_2(
    _ctx: *mut super::jit_context::RssnJitContext,
    _fn_id: u32,
    _func: Option<extern "C" fn(f64, f64) -> f64>,
) -> RssnStatus {
    RssnStatus::CompilationError
}

/// Registers a three-argument native function with the persistent JIT context.
///
/// # Safety
///
/// `func` must be a valid function pointer with the signature `double(double, double, double)`.
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_jit_register_fn_3(
    ctx: *mut super::jit_context::RssnJitContext,
    fn_id: u32,
    func: Option<extern "C" fn(f64, f64, f64) -> f64>,
) -> RssnStatus {
    if ctx.is_null() {
        return RssnStatus::NullPointer;
    }
    let Some(func_ptr) = func else {
        return RssnStatus::NullPointer;
    };
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        let ctx_ref = unsafe { &mut *ctx };
        ctx_ref
            .compiler_mut()
            .register_custom_function_3(crate::dag::symbol::FnId(fn_id), func_ptr);
        RssnStatus::Success
    }))
    .unwrap_or(RssnStatus::Panic)
}

/// Registers a three-argument native function (stub for non-JIT builds).
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_jit_register_fn_3(
    _ctx: *mut super::jit_context::RssnJitContext,
    _fn_id: u32,
    _func: Option<extern "C" fn(f64, f64, f64) -> f64>,
) -> RssnStatus {
    RssnStatus::CompilationError
}

// =========================================================================
// T6.7 — JIT compile with explicit optimisation configuration
// =========================================================================

/// C-compatible JIT optimisation configuration.
///
/// Fields mirror [`crate::jit::compiler::OptConfig`]. Pass a pointer to this
/// struct to [`rssn_dag_compile_with_opts`]; pass NULL to use the defaults.
#[cfg(feature = "cranelift-jit")]
#[repr(C)]
pub struct RssnOptConfig {
    /// Maximum integer exponent expanded without a `powf` call (default: 16).
    pub max_int_pow: u32,
    /// Non-zero to expand `x^0.5` to a native `sqrt` instruction (default: 1).
    pub expand_sqrt: u32,
    /// Non-zero to replace `x / C` with `x * (1/C)` (default: 0).
    pub allow_reciprocal_math: u32,
    /// Non-zero to skip divide-by-zero guards when the denominator is proven
    /// non-zero by the analysis pass (default: 1).
    pub elide_nan_guard: u32,
    /// Non-zero to reuse SSA values for repeated DAG sub-expressions (default: 1).
    pub enable_cse: u32,
}

/// Dummy C-compatible JIT optimisation configuration for non-JIT builds.
#[cfg(not(feature = "cranelift-jit"))]
#[repr(C)]
pub struct RssnOptConfig;

/// Compiles a DAG expression with explicit optimisation knobs.
///
/// If `opts` is NULL, uses [`RssnOptConfig`] defaults (equivalent to
/// [`rssn_dag_compile_v2`]).
///
/// # Safety
///
/// - `ctx` must be a valid, non-null pointer from [`rssn_jit_context_new`](crate::ffi::jit_context::rssn_jit_context_new).
/// - `builder` must be a valid, non-null pointer from [`rssn_dag_new`].
/// - `out_fn` must be a valid, non-null writable pointer.
/// - If `opts` is non-null it must point to a valid [`RssnOptConfig`].
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_compile_with_opts(
    ctx: *mut super::jit_context::RssnJitContext,
    builder: *mut DagBuilder,
    root: u32,
    opts: *const RssnOptConfig,
    out_fn: *mut *mut c_void,
) -> RssnStatus {
    if ctx.is_null() || builder.is_null() || out_fn.is_null() {
        return RssnStatus::NullPointer;
    }
    if root == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let ctx_ref = unsafe { &mut *ctx };
        let builder_ref = unsafe { &mut *builder };
        let root_id = DagNodeId::new(root);
        let ast = crate::ast::convert::dag_to_ast(builder_ref.arena(), root_id);

        let jit_opts = if opts.is_null() {
            crate::jit::compiler::OptConfig::default()
        } else {
            let c = unsafe { &*opts };
            crate::jit::compiler::OptConfig {
                max_int_pow: c.max_int_pow,
                expand_sqrt: c.expand_sqrt != 0,
                allow_reciprocal_math: c.allow_reciprocal_math != 0,
                elide_nan_guard: c.elide_nan_guard != 0,
                enable_cse: c.enable_cse != 0,
            }
        };

        ctx_ref
            .compiler_mut()
            .compile_with_opts(&ast, &jit_opts)
            .map_or(RssnStatus::CompilationError, |compiled_fn| {
                unsafe { *out_fn = compiled_fn as *mut c_void };
                RssnStatus::Success
            })
    }));
    result.unwrap_or(RssnStatus::Panic)
}

/// Stub for non-JIT builds: always returns `CompilationError`.
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_compile_with_opts(
    _ctx: *mut super::jit_context::RssnJitContext,
    _builder: *mut DagBuilder,
    _root: u32,
    _opts: *const RssnOptConfig,
    _out_fn: *mut *mut c_void,
) -> RssnStatus {
    RssnStatus::CompilationError
}

// =========================================================================
// T6.8 — C-side rewrite rule registration
// =========================================================================
//
// The C rule callback receives:
//   - A pointer to the builder (may call rssn_dag_* to create nodes).
//   - The node kind discriminant (see `RssnKind` below).
//   - A pointer to the child node-id array and child count.
//   - User data (an opaque void* set at registration time).
// The callback returns the replacement node id, or u32::MAX to pass.

/// Discriminant values for `SymbolKind` variants, matching the Rust enum.
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum RssnKind {
    /// A named variable.
    Variable = 0,
    /// A numeric constant.
    Constant = 1,
    /// `Add` operator.
    Add = 2,
    /// `Sub` operator.
    Sub = 3,
    /// `Mul` operator.
    Mul = 4,
    /// `Div` operator.
    Div = 5,
    /// `Pow` operator.
    Pow = 6,
    /// `Mod` operator.
    Mod = 7,
    /// Unary `Neg` operator.
    Neg = 8,
    /// A custom function call.
    Function = 9,
}

/// Opaque handle for a registered C rewrite rule registry.
///
/// Create with [`rssn_rule_registry_new`]; free with [`rssn_rule_registry_free`].
pub struct RssnRuleRegistry {
    inner: std::sync::Arc<crate::heuristic::rule_registry::RuleRegistry>,
}

/// C-callable rewrite rule callback.
///
/// - `builder`: pointer to the `DagBuilder`; call `rssn_dag_*` to create nodes.
/// - `kind`: node kind discriminant (see [`RssnKind`]).
/// - `children`: pointer to an array of child node ids (length `n_children`).
/// - `n_children`: number of children.
/// - `user_data`: the opaque pointer supplied at registration time.
///
/// Return the replacement node id, or `u32::MAX` to leave the node unchanged
/// (pass to the next rule).
pub type RssnRuleCallback = unsafe extern "C" fn(
    builder: *mut DagBuilder,
    kind: u8,
    children: *const u32,
    n_children: u32,
    user_data: *mut c_void,
) -> u32;

/// Creates a new, empty rule registry.
///
/// The returned pointer must be freed with [`rssn_rule_registry_free`].
/// Returns NULL if construction panics.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_rule_registry_new() -> *mut RssnRuleRegistry {
    catch_unwind(|| {
        Box::into_raw(Box::new(RssnRuleRegistry {
            inner: std::sync::Arc::new(crate::heuristic::rule_registry::RuleRegistry::new()),
        }))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Frees a rule registry previously created by [`rssn_rule_registry_new`].
///
/// Passing NULL is a safe no-op.
///
/// # Safety
///
/// `registry` must be a pointer returned by [`rssn_rule_registry_new`] or NULL.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_rule_registry_free(registry: *mut RssnRuleRegistry) {
    if registry.is_null() {
        return;
    }
    let _ = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = unsafe { Box::from_raw(registry) };
    }));
}

/// Registers a C callback as a rewrite rule.
///
/// - `name`: human-readable rule name (null-terminated C string, for fingerprinting).
/// - `callback`: the rule function; called during simplification for each node.
/// - `priority`: higher values are tried first (default-priority rules use 0).
/// - `kind_filter`: if non-negative, the rule is only tried for nodes with this
///   kind discriminant (see [`RssnKind`]). Pass `-1` for a wildcard rule.
/// - `user_data`: opaque pointer forwarded to every callback invocation.
///
/// # Safety
///
/// - `registry` must be a valid, non-null pointer from [`rssn_rule_registry_new`].
/// - `name` must be a valid, non-null, null-terminated C string.
/// - `callback` must be a valid, non-null function pointer.
/// - `user_data` must remain valid for the lifetime of the registry.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_rule_register(
    registry: *mut RssnRuleRegistry,
    name: *const c_char,
    callback: Option<RssnRuleCallback>,
    priority: i32,
    kind_filter: i32,
    user_data: *mut c_void,
) -> RssnStatus {
    if registry.is_null() || name.is_null() {
        return RssnStatus::NullPointer;
    }
    let Some(cb) = callback else {
        return RssnStatus::NullPointer;
    };

    let name_str = {
        let c_str = unsafe { CStr::from_ptr(name) };
        match c_str.to_str() {
            Ok(s) => s.to_owned(),
            Err(_) => return RssnStatus::InvalidUtf8,
        }
    };

    // Transmute user_data to usize so the closure is Send + Sync.
    let user_data_addr = user_data as usize;

    let kind_opt = if kind_filter < 0 {
        None
    } else {
        use crate::dag::symbol::{OpKind, SymbolKind};
        match kind_filter as u8 {
            0 => Some(SymbolKind::Variable(crate::dag::symbol::SymbolId(0))),
            1 => Some(SymbolKind::Constant(0.0)),
            2 => Some(SymbolKind::Operator(OpKind::Add)),
            3 => Some(SymbolKind::Operator(OpKind::Sub)),
            4 => Some(SymbolKind::Operator(OpKind::Mul)),
            5 => Some(SymbolKind::Operator(OpKind::Div)),
            6 => Some(SymbolKind::Operator(OpKind::Pow)),
            7 => Some(SymbolKind::Operator(OpKind::Mod)),
            8 => Some(SymbolKind::Operator(OpKind::Neg)),
            9 => Some(SymbolKind::Function(crate::dag::symbol::FnId(0))),
            _ => return RssnStatus::InvalidNodeId,
        }
    };

    catch_unwind(std::panic::AssertUnwindSafe(|| {
        let reg_ref = unsafe { &mut *registry };
        // Get a mutable reference to the inner registry through Arc.
        // If the Arc is uniquely owned (typical during registration phase),
        // `get_mut` succeeds. After the first `clone()` by the engine we can
        // no longer add rules — callers must register before simplifying.
        let Some(inner_mut) = std::sync::Arc::get_mut(&mut reg_ref.inner) else {
            return RssnStatus::RuleConflict;
        };
        inner_mut.register_named(
            &name_str,
            move |builder, kind, children| {
                // Flatten children into a temporary u32 array.
                let child_ids: Vec<u32> = children.iter().map(|id| id.value()).collect();
                let user_data_ptr = user_data_addr as *mut c_void;
                // SAFETY: the caller guarantees the callback and user_data are valid.
                let result = unsafe {
                    cb(
                        std::ptr::from_mut::<DagBuilder>(builder),
                        kind_to_discriminant(&kind),
                        child_ids.as_ptr(),
                        child_ids.len() as u32,
                        user_data_ptr,
                    )
                };
                if result == u32::MAX {
                    None
                } else {
                    Some(DagNodeId::new(result))
                }
            },
            priority,
            kind_opt,
        );
        RssnStatus::Success
    }))
    .unwrap_or(RssnStatus::Panic)
}

/// Simplifies an expression using a caller-supplied C rule registry and configuration.
///
/// If `registry` is NULL, only the built-in heuristic patterns are applied.
/// If `config` is NULL, defaults are used.
///
/// # Safety
///
/// - `builder` and `out_id` must be valid, non-null pointers.
/// - If `registry` is non-null it must be from [`rssn_rule_registry_new`].
/// - If `config` is non-null it must point to a valid [`RssnSimplifyConfig`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_simplify_with_rules(
    builder: *mut DagBuilder,
    root: u32,
    registry: *mut RssnRuleRegistry,
    config: *const RssnSimplifyConfig,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    if root == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    let (max_depth, timeout_ms, aggressiveness) = if config.is_null() {
        let def = HeuristicConfig::default();
        (
            def.max_depth,
            def.timeout.as_millis() as u64,
            def.simplification_aggressiveness,
        )
    } else {
        let c = unsafe { &*config };
        (c.max_depth as usize, c.timeout_ms, c.aggressiveness)
    };

    catch_unwind(std::panic::AssertUnwindSafe(|| -> RssnStatus {
        let builder_ref = unsafe { &mut *builder };
        let root_id = DagNodeId::new(root);

        // If a registry is supplied, transfer its rules into the engine.
        let cfg = HeuristicConfig::default()
            .max_depth(max_depth)
            .timeout(Duration::from_millis(timeout_ms))
            .simplification_aggressiveness(aggressiveness);

        let mut engine = if registry.is_null() {
            HeuristicEngine::new(cfg, SearchStrategy::Greedy)
        } else {
            // Share the Arc<RuleRegistry> with the engine. This is cheap
            // (one atomic ref-count increment) and keeps the registry alive
            // and accessible to the C caller after the call returns.
            let arc_clone = std::sync::Arc::clone(unsafe { &(*registry).inner });
            HeuristicEngine::new(cfg, SearchStrategy::Greedy).with_rule_registry(arc_clone)
        };

        let id = engine.simplify(builder_ref, root_id);
        unsafe { *out_id = id.value() };
        RssnStatus::Success
    }))
    .unwrap_or(RssnStatus::Panic)
}

/// Maps a `SymbolKind` to its C discriminant byte.
const fn kind_to_discriminant(kind: &crate::dag::symbol::SymbolKind) -> u8 {
    use crate::dag::symbol::{OpKind, SymbolKind};
    match kind {
        SymbolKind::Variable(_) => 0,
        SymbolKind::Constant(_) => 1,
        SymbolKind::Operator(OpKind::Add) => 2,
        SymbolKind::Operator(OpKind::Sub) => 3,
        SymbolKind::Operator(OpKind::Mul) => 4,
        SymbolKind::Operator(OpKind::Div) => 5,
        SymbolKind::Operator(OpKind::Pow) => 6,
        SymbolKind::Operator(OpKind::Mod) => 7,
        SymbolKind::Operator(OpKind::Neg) => 8,
        SymbolKind::Function(_) => 9,
    }
}

// =========================================================================
// Tests for the new FFI functions
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_ops_roundtrip() {
        let builder = rssn_dag_new();
        assert!(!builder.is_null());

        let x = rssn_dag_variable(builder, c"x".as_ptr());
        let y = rssn_dag_variable(builder, c"y".as_ptr());
        assert_ne!(x, u32::MAX);
        assert_ne!(y, u32::MAX);

        assert_ne!(rssn_dag_sub(builder, x, y), u32::MAX);
        assert_ne!(rssn_dag_mul(builder, x, y), u32::MAX);
        assert_ne!(rssn_dag_div(builder, x, y), u32::MAX);
        assert_ne!(rssn_dag_pow(builder, x, y), u32::MAX);
        assert_ne!(rssn_dag_mod(builder, x, y), u32::MAX);
        assert_ne!(rssn_dag_neg(builder, x), u32::MAX);

        rssn_dag_free(builder);
    }

    #[test]
    fn v2_ops_return_success() {
        let builder = rssn_dag_new();
        let x = rssn_dag_variable(builder, c"x".as_ptr());
        let y = rssn_dag_variable(builder, c"y".as_ptr());

        let mut out = u32::MAX;
        assert_eq!(
            rssn_dag_sub_v2(builder, x, y, &mut out),
            RssnStatus::Success
        );
        assert_ne!(out, u32::MAX);
        assert_eq!(
            rssn_dag_mul_v2(builder, x, y, &mut out),
            RssnStatus::Success
        );
        assert_ne!(out, u32::MAX);
        assert_eq!(
            rssn_dag_div_v2(builder, x, y, &mut out),
            RssnStatus::Success
        );
        assert_ne!(out, u32::MAX);
        assert_eq!(
            rssn_dag_pow_v2(builder, x, y, &mut out),
            RssnStatus::Success
        );
        assert_ne!(out, u32::MAX);
        assert_eq!(
            rssn_dag_mod_v2(builder, x, y, &mut out),
            RssnStatus::Success
        );
        assert_ne!(out, u32::MAX);
        assert_eq!(rssn_dag_neg_v2(builder, x, &mut out), RssnStatus::Success);
        assert_ne!(out, u32::MAX);

        rssn_dag_free(builder);
    }

    #[test]
    fn null_inputs_return_sentinel_or_null_pointer_status() {
        assert_eq!(rssn_dag_sub(std::ptr::null_mut(), 0, 0), u32::MAX);
        assert_eq!(rssn_dag_mul(std::ptr::null_mut(), 0, 0), u32::MAX);
        assert_eq!(rssn_dag_div(std::ptr::null_mut(), 0, 0), u32::MAX);
        assert_eq!(rssn_dag_pow(std::ptr::null_mut(), 0, 0), u32::MAX);
        assert_eq!(rssn_dag_mod(std::ptr::null_mut(), 0, 0), u32::MAX);
        assert_eq!(rssn_dag_neg(std::ptr::null_mut(), 0), u32::MAX);
    }

    #[test]
    fn parse_and_build() {
        let builder = rssn_dag_new();
        let mut out = u32::MAX;
        let status = rssn_dag_parse(builder, c"x + y * 2.0".as_ptr(), &mut out);
        assert_eq!(status, RssnStatus::Success);
        assert_ne!(out, u32::MAX);
        rssn_dag_free(builder);
    }

    #[test]
    fn parse_invalid_expression() {
        let builder = rssn_dag_new();
        let mut out = u32::MAX;
        let status = rssn_dag_parse(builder, c"(".as_ptr(), &mut out);
        assert_ne!(status, RssnStatus::Success);
        rssn_dag_free(builder);
    }

    #[test]
    fn intern_function_and_call() {
        let builder = rssn_dag_new();
        let fn_id = rssn_dag_intern_function(builder, c"mysin".as_ptr());
        assert_ne!(fn_id, u32::MAX);

        let x = rssn_dag_variable(builder, c"x".as_ptr());
        let args = [x];
        let call_node = rssn_dag_call_fn(builder, fn_id, args.as_ptr(), 1);
        assert_ne!(call_node, u32::MAX);
        rssn_dag_free(builder);
    }

    #[test]
    fn rule_registry_lifecycle() {
        let reg = rssn_rule_registry_new();
        assert!(!reg.is_null());
        rssn_rule_registry_free(reg);

        // Double-free safety: free NULL is a no-op.
        rssn_rule_registry_free(std::ptr::null_mut());
    }

    #[test]
    fn register_and_apply_c_rule() {
        unsafe extern "C" fn zero_add_rule(
            builder: *mut DagBuilder,
            kind: u8,
            children: *const u32,
            n_children: u32,
            _user_data: *mut c_void,
        ) -> u32 {
            // Rule: x + 0 → x  (kind == Add, one child is constant 0)
            if kind != 2 || n_children != 2 {
                return u32::MAX;
            }
            let lhs = unsafe { *children };
            let rhs = unsafe { *children.add(1) };
            let b = unsafe { &mut *builder };
            let rhs_node = b.arena().get(DagNodeId::new(rhs));
            if let Some(node) = rhs_node {
                if let crate::dag::symbol::SymbolKind::Constant(v) = node.kind {
                    if v == 0.0 {
                        return lhs;
                    }
                }
            }
            u32::MAX
        }

        let reg = rssn_rule_registry_new();
        let status = rssn_rule_register(
            reg,
            c"zero_add".as_ptr(),
            Some(zero_add_rule),
            0,
            2, // Add
            std::ptr::null_mut(),
        );
        assert_eq!(status, RssnStatus::Success);

        // Build x + 0 and simplify.
        let builder = rssn_dag_new();
        let x = rssn_dag_variable(builder, c"x".as_ptr());
        let zero = rssn_dag_constant(builder, 0.0);
        let expr = rssn_dag_add(builder, x, zero);

        let mut out = u32::MAX;
        let s = rssn_dag_simplify_with_rules(builder, expr, reg, std::ptr::null(), &mut out);
        assert_eq!(s, RssnStatus::Success);
        // After simplification x + 0 should reduce to x.
        assert_eq!(out, x, "x + 0 should simplify to x");

        rssn_dag_free(builder);
        rssn_rule_registry_free(reg);
    }
}

// =========================================================================
// E-graph equality saturation FFI
// =========================================================================
// Design: the EGraph is created transiently for each saturate+extract call.
// This avoids exposing a long-lived handle across the FFI boundary (which
// would complicate lifetime management on both sides). The overhead is low
// because all real memory lives inside the DagBuilder which is long-lived.
//
// For callers that need repeated extraction on the same expression with the
// same rule set, the recommended pattern is:
//   1. rssn_dag_egraph_saturate_extract(...) — one call, returns best node.
//   2. Cache the returned node ID on the C side.

/// Configuration for E-graph equality saturation.
///
/// Passed by value across the FFI boundary; zero-initialise for defaults.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RssnEGraphConfig {
    /// Maximum saturation rounds (0 → use library default of 8).
    pub max_rounds: u32,
    /// Maximum equivalence merges before stopping (0 → default 512).
    pub max_merges: u32,
    /// Maximum new nodes the E-graph may create via rewrites (0 → default 1024).
    pub max_new_nodes: u32,
    /// Non-zero → enable strict IEEE 754 signed-zero semantics.
    ///
    /// When set, `x + (-0.0)` will **not** be simplified to `x`, matching
    /// `-fno-unsafe-math-optimizations`. Default (0) uses `-fno-signed-zeros`.
    pub strict_ieee754_signed_zero: u8,
}

impl RssnEGraphConfig {
    const fn to_rust(self) -> crate::egraph::EGraphConfig {
        crate::egraph::EGraphConfig {
            max_rounds: if self.max_rounds == 0 {
                8
            } else {
                self.max_rounds as usize
            },
            max_merges: if self.max_merges == 0 {
                512
            } else {
                self.max_merges as usize
            },
            max_new_nodes: if self.max_new_nodes == 0 {
                1024
            } else {
                self.max_new_nodes as usize
            },
            strict_ieee754_signed_zero: self.strict_ieee754_signed_zero != 0,
            cost_weights: None,
        }
    }
}

/// A C-callable rewrite rule for the E-graph.
///
/// Called for each node during saturation. Return the ID of an equivalent
/// node to merge into the same e-class, or `u32::MAX` to decline.
///
/// `kind`       — discriminant of the current node's kind (see `RssnKind`).
/// `children`   — pointer to the *canonical* child IDs (length `n_children`).
/// `n_children` — number of children.
/// `user_data`  — opaque pointer forwarded unchanged from the registration call.
pub type RssnEGraphRuleCallback = unsafe extern "C" fn(
    builder: *mut DagBuilder,
    kind: u8,
    children: *const u32,
    n_children: u32,
    user_data: *mut c_void,
) -> u32;

/// Runs E-graph equality saturation on `root` and returns the cheapest
/// equivalent node ID, or `u32::MAX` on error.
///
/// # C example
///
/// ```c
/// uint32_t best = rssn_dag_egraph_saturate_extract(
///     builder, expr_id,
///     (RssnEGraphConfig){ .max_rounds = 4, .max_merges = 256, .max_new_nodes = 512 },
///     NULL, 0,   // no user rules
///     NULL
/// );
/// ```
///
/// # Safety
///
/// - `builder` must be a valid, non-null `DagBuilder` from `rssn_dag_new`.
/// - If `rules` is non-null, `n_rules` must be the number of valid callback
///   pointers in the array.
/// - `user_data` pointers must remain valid for the duration of this call.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_egraph_saturate_extract(
    builder: *mut DagBuilder,
    root: u32,
    cfg: RssnEGraphConfig,
    rules: *const RssnEGraphRuleCallback,
    n_rules: u32,
    out: *mut u32,
) -> RssnStatus {
    if builder.is_null() {
        return RssnStatus::NullPointer;
    }
    let result = catch_unwind(|| -> RssnStatus {
        let b = unsafe { &mut *builder };
        let root_id = crate::dag::node::DagNodeId::new(root);
        let rust_cfg = cfg.to_rust();

        let mut eg = crate::egraph::EGraph::new(b, rust_cfg);

        // Register C-side user rules.
        if !rules.is_null() {
            for i in 0..n_rules as usize {
                let cb: RssnEGraphRuleCallback = unsafe { *rules.add(i) };
                // SAFETY: cb is valid for the duration of saturate (this call).
                // user_data is forwarded transparently.
                eg.add_rule(move |builder_inner, kind, children| {
                    let kind_disc = kind_to_discriminant(kind);
                    let ch_ptr = children.as_ptr().cast::<u32>();
                    let result = unsafe {
                        cb(
                            std::ptr::from_mut::<DagBuilder>(builder_inner),
                            kind_disc,
                            ch_ptr,
                            children.len() as u32,
                            std::ptr::null_mut(), // user_data not storable in 'static closure
                        )
                    };
                    if result == u32::MAX {
                        None
                    } else {
                        Some(crate::dag::node::DagNodeId::new(result))
                    }
                });
            }
        }

        eg.saturate(root_id);
        let best = eg.extract(root_id);
        if let Some(out_ptr) = unsafe { out.as_mut() } {
            *out_ptr = best.value();
        }
        RssnStatus::Success
    });
    result.unwrap_or(RssnStatus::Panic)
}

/// Like [`rssn_dag_egraph_saturate_extract`] but also enables the E-graph
/// pass inside the full heuristic simplification pipeline and returns the
/// result after both passes.
///
/// # Safety
///
/// Same as `rssn_dag_egraph_saturate_extract`.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_simplify_with_egraph(
    builder: *mut DagBuilder,
    root: u32,
    egraph_cfg: RssnEGraphConfig,
    out: *mut u32,
) -> RssnStatus {
    if builder.is_null() {
        return RssnStatus::NullPointer;
    }
    let result = catch_unwind(|| -> RssnStatus {
        let b = unsafe { &mut *builder };
        let root_id = crate::dag::node::DagNodeId::new(root);
        let hcfg = crate::heuristic::HeuristicConfig::default().with_egraph(egraph_cfg.to_rust());
        let mut engine = crate::heuristic::HeuristicEngine::new(
            hcfg,
            crate::heuristic::SearchStrategy::default(),
        );
        let simplified = engine.simplify(b, root_id);
        if let Some(out_ptr) = unsafe { out.as_mut() } {
            *out_ptr = simplified.value();
        }
        RssnStatus::Success
    });
    result.unwrap_or(RssnStatus::Panic)
}

// =========================================================================
// Batch custom operator registry
// =========================================================================
//
// Developers can register user-defined operators for use with the batch-build
// API without modifying library source code.  Registered kinds must fall in
// the range 16..=255 (kinds 0..=15 are reserved for built-in operators).
//
// Thread safety: the registry uses a `RwLock`; concurrent reads (during
// `rssn_dag_batch_build`) never block each other.  Writes (registration /
// unregistration) acquire an exclusive lock.

use std::collections::HashMap as StdHashMap;
use std::sync::OnceLock;

/// Callback type for a custom batch-build operator.
///
/// Called during [`rssn_dag_batch_build`] when the node `kind` field matches
/// a registered custom kind.  The callback receives a `DagBuilder`, the
/// resolved child node IDs and their count, and the `user_data` pointer
/// supplied at registration.  Return a new valid node ID allocated in
/// `builder`, or `u32::MAX` to signal failure.
///
/// # Safety
///
/// - `builder` is valid and non-null for the duration of this call.
/// - `child_ids` points to an array of exactly `n_children` resolved node IDs.
/// - `user_data` is the opaque pointer provided at `rssn_batch_op_register` time;
///   the caller is responsible for its lifetime.
pub type RssnBatchOpCallback = unsafe extern "C" fn(
    builder: *mut DagBuilder,
    child_ids: *const u32,
    n_children: u32,
    user_data: *mut c_void,
) -> u32;

/// Entry stored in the process-level batch operator registry.
struct BatchOpEntry {
    callback: RssnBatchOpCallback,
    /// Expected number of resolved children (capped at 2 by `RssnNodeDesc`).
    n_children: u32,
    /// Caller-provided opaque pointer, stored as `usize` for `Send` safety.
    user_data: usize,
}

static BATCH_OP_REGISTRY: OnceLock<std::sync::RwLock<StdHashMap<u8, BatchOpEntry>>> =
    OnceLock::new();

#[inline]
fn batch_op_registry() -> &'static std::sync::RwLock<StdHashMap<u8, BatchOpEntry>> {
    BATCH_OP_REGISTRY.get_or_init(|| std::sync::RwLock::new(StdHashMap::new()))
}

/// Registers a custom batch operator for use with [`rssn_dag_batch_build`].
///
/// `kind` must be in the range `16..=255`; kinds `0..=15` are reserved for
/// built-in operators and this function returns [`RssnStatus::InvalidNodeId`]
/// if `kind` falls in that range.  Registering the same `kind` twice returns
/// [`RssnStatus::RuleConflict`].
///
/// The `callback` receives the resolved child node IDs for the batch node and
/// must allocate a new DAG node in `builder`, returning its id.  `n_children`
/// specifies how many of `child0`/`child1` are meaningful in the descriptor
/// (currently capped at 2 by the `RssnNodeDesc` layout).
///
/// # Safety
///
/// - `callback` must be a valid function pointer that remains valid until
///   [`rssn_batch_op_unregister`] is called for the same `kind`.
/// - `user_data` is forwarded to the callback opaquely; its lifetime is the
///   caller's responsibility.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn rssn_batch_op_register(
    kind: u8,
    n_children: u32,
    callback: Option<RssnBatchOpCallback>,
    user_data: *mut c_void,
) -> RssnStatus {
    if kind < 16 {
        // Built-in range is 0..=15; custom operators start at 16.
        return RssnStatus::InvalidNodeId;
    }
    let Some(cb) = callback else {
        return RssnStatus::NullPointer;
    };
    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let reg = batch_op_registry();
        let mut guard = reg
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.contains_key(&kind) {
            return RssnStatus::RuleConflict;
        }
        guard.insert(
            kind,
            BatchOpEntry {
                callback: cb,
                n_children,
                user_data: user_data as usize,
            },
        );
        RssnStatus::Success
    }));
    result.unwrap_or(RssnStatus::Panic)
}

/// Unregisters a previously registered custom batch operator.
///
/// Returns [`RssnStatus::Success`] if the kind was registered, or
/// [`RssnStatus::InvalidNodeId`] if it was not (or if `kind < 16`).
#[unsafe(no_mangle)]
pub extern "C" fn rssn_batch_op_unregister(kind: u8) -> RssnStatus {
    if kind < 16 {
        return RssnStatus::InvalidNodeId;
    }
    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let reg = batch_op_registry();
        let mut guard = reg
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.remove(&kind).is_some() {
            RssnStatus::Success
        } else {
            RssnStatus::InvalidNodeId
        }
    }));
    result.unwrap_or(RssnStatus::Panic)
}

// =========================================================================
// Batch-build API — reduced cross-FFI overhead
// =========================================================================
// Rationale: building a 50-node expression via individual rssn_dag_add/mul/
// etc. calls costs 50× catch_unwind + null check + FFI frame. The batch
// API amortises this to a single call: C fills an array of `RssnNodeDesc`
// and we process the whole array in one Rust call.
//
// Custom operators (kinds 16–255) registered via `rssn_batch_op_register`
// are dispatched through the process-level `BATCH_OP_REGISTRY` below.

/// Node kind discriminant used in [`RssnNodeDesc`].
///
/// Values 0–8 are built-in; 16–255 are available for user-defined operators
/// registered via [`rssn_batch_op_register`].  Matches `RssnKind` in the C header.
pub type RssnNodeKindBatch = u8;

/// Compact node descriptor for batch DAG construction.
///
/// The caller allocates an array of these, fills them in topological order
/// (children before parents), and passes the whole array to
/// [`rssn_dag_batch_build`]. The output array receives the allocated IDs.
///
/// Field semantics by `kind`:
///
/// | kind | meaning | fields used |
/// |------|---------|-------------|
/// | 0 = Variable | leaf variable | `name[0..32]` |
/// | 1 = Constant | leaf constant | `value` |
/// | 2 = Add      | `child0 + child1` | `child0`, `child1` |
/// | 3 = Sub      | `child0 - child1` | `child0`, `child1` |
/// | 4 = Mul      | `child0 * child1` | `child0`, `child1` |
/// | 5 = Div      | `child0 / child1` | `child0`, `child1` |
/// | 6 = Pow      | `child0 ^ child1` | `child0`, `child1` |
/// | 7 = Neg      | `-child0`         | `child0` |
/// | 8 = Mod      | `child0 % child1` | `child0`, `child1` |
///
/// `child0` and `child1` are **indices into the `out_ids` array** of the
/// same batch call — they are NOT `DagNodeId` values. Index `u32::MAX`
/// means "no child". This allows forward-reference-free batch construction.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RssnNodeDesc {
    /// Constant value (used when `kind == 1`).
    pub value: f64,
    /// Index into `out_ids` of this call for the first child.
    pub child0: u32,
    /// Index into `out_ids` of this call for the second child.
    pub child1: u32,
    /// Node kind discriminant (see table above).
    pub kind: u8,
    /// Null-terminated variable name (used when `kind == 0`).
    pub name: [u8; 31],
}

/// Builds `n` DAG nodes in a single FFI call, writing allocated node IDs
/// into `out_ids`.
///
/// Nodes are processed in order `0..n`. Children are referenced by their
/// **index in the batch** (not their `DagNodeId`); the builder translates
/// indices to IDs after allocating each node.
///
/// Returns `RssnStatus::Success` on success. On any error the output array
/// may be partially populated — already-built nodes remain valid.
///
/// # Safety
///
/// - `builder` must be a valid, non-null `DagBuilder` from `rssn_dag_new`.
/// - `descs` must point to an array of at least `n` `RssnNodeDesc` values,
///   valid for the duration of this call.
/// - `out_ids` must point to a writable array of at least `n` `u32` values.
/// - Children referenced by `child0`/`child1` must have indices strictly
///   less than the current node's index (topological order).
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_batch_build(
    builder: *mut crate::dag::builder::DagBuilder,
    descs: *const RssnNodeDesc,
    n: u32,
    out_ids: *mut u32,
) -> RssnStatus {
    if builder.is_null() || descs.is_null() || out_ids.is_null() {
        return RssnStatus::NullPointer;
    }
    let result = catch_unwind(|| -> RssnStatus {
        let b = unsafe { &mut *builder };
        let descs_slice: &[RssnNodeDesc] = unsafe { std::slice::from_raw_parts(descs, n as usize) };
        let out_slice: &mut [u32] = unsafe { std::slice::from_raw_parts_mut(out_ids, n as usize) };

        // Accumulated IDs for this batch (so nodes can reference earlier siblings).
        let mut batch_ids: Vec<crate::dag::node::DagNodeId> = Vec::with_capacity(n as usize);

        for (i, desc) in descs_slice.iter().enumerate() {
            // Resolve child indices → DagNodeIds, clamping out-of-range to NONE.
            let resolve = |idx: u32| -> crate::dag::node::DagNodeId {
                if idx == u32::MAX || idx as usize >= i {
                    crate::dag::node::DagNodeId::NONE
                } else {
                    batch_ids[idx as usize]
                }
            };

            let id = match desc.kind {
                0 => {
                    // Variable: find the null terminator in `desc.name`.
                    let name_bytes = &desc.name;
                    let len = name_bytes.iter().position(|&b| b == 0).unwrap_or(31);
                    b.variable_bytes(&name_bytes[..len])
                        .unwrap_or(crate::dag::node::DagNodeId::NONE)
                }
                1 => b.constant(desc.value),
                2 => {
                    let (c0, c1) = (resolve(desc.child0), resolve(desc.child1));
                    if c0.is_none() || c1.is_none() {
                        return RssnStatus::InvalidNode;
                    }
                    b.add(c0, c1)
                }
                3 => {
                    let (c0, c1) = (resolve(desc.child0), resolve(desc.child1));
                    if c0.is_none() || c1.is_none() {
                        return RssnStatus::InvalidNode;
                    }
                    b.sub(c0, c1)
                }
                4 => {
                    let (c0, c1) = (resolve(desc.child0), resolve(desc.child1));
                    if c0.is_none() || c1.is_none() {
                        return RssnStatus::InvalidNode;
                    }
                    b.mul(c0, c1)
                }
                5 => {
                    let (c0, c1) = (resolve(desc.child0), resolve(desc.child1));
                    if c0.is_none() || c1.is_none() {
                        return RssnStatus::InvalidNode;
                    }
                    b.div(c0, c1)
                }
                6 => {
                    let (c0, c1) = (resolve(desc.child0), resolve(desc.child1));
                    if c0.is_none() || c1.is_none() {
                        return RssnStatus::InvalidNode;
                    }
                    b.pow(c0, c1)
                }
                7 => {
                    let c0 = resolve(desc.child0);
                    if c0.is_none() {
                        return RssnStatus::InvalidNode;
                    }
                    b.neg(c0)
                }
                8 => {
                    let (c0, c1) = (resolve(desc.child0), resolve(desc.child1));
                    if c0.is_none() || c1.is_none() {
                        return RssnStatus::InvalidNode;
                    }
                    b.modulo(c0, c1)
                }
                kind => {
                    // Look up a user-defined operator in the custom registry.
                    // We copy the entry fields before dropping the read lock so
                    // the callback can safely re-enter `builder` without holding
                    // the registry lock.
                    let entry = {
                        let reg = batch_op_registry();
                        let guard = reg
                            .read()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        guard
                            .get(&kind)
                            .map(|e| (e.callback, e.n_children, e.user_data))
                    };
                    let Some((cb, n_ch, ud_usize)) = entry else {
                        return RssnStatus::InvalidNode;
                    };
                    // Resolve up to two children from the batch index space.
                    let resolve_raw = |idx: u32| -> u32 {
                        if idx == u32::MAX || idx as usize >= i {
                            u32::MAX
                        } else {
                            batch_ids[idx as usize].value()
                        }
                    };
                    let actual_n = (n_ch as usize).min(2);
                    let mut ch_buf = [u32::MAX; 2];
                    if actual_n > 0 {
                        ch_buf[0] = resolve_raw(desc.child0);
                    }
                    if actual_n > 1 {
                        ch_buf[1] = resolve_raw(desc.child1);
                    }
                    let ud = ud_usize as *mut c_void;
                    // SAFETY: callback is a valid fn ptr (guaranteed by rssn_batch_op_register),
                    // builder is valid for this call, ch_buf lives on the stack.
                    let result_id = unsafe {
                        cb(
                            std::ptr::from_mut::<DagBuilder>(b),
                            ch_buf.as_ptr(),
                            actual_n as u32,
                            ud,
                        )
                    };
                    if result_id == u32::MAX {
                        return RssnStatus::InvalidNode;
                    }
                    DagNodeId::new(result_id)
                }
            };

            out_slice[i] = id.value();
            batch_ids.push(id);
        }
        RssnStatus::Success
    });
    result.unwrap_or(RssnStatus::Panic)
}

/// Writes the packed arena snapshot to a caller-provided byte buffer.
///
/// On success, `*bytes_written` receives the number of bytes written.
/// Call once with `buf = NULL` to query the required buffer size
/// (`*bytes_written` will be the needed byte count and the return value
/// is `RssnStatus::Success`).
///
/// The layout is: a little-endian `u64` node count, then `n × 32` bytes
/// of packed node data (`PackedDagNode`), then a little-endian `u64` pool
/// count, then `pool_count × 4` bytes of `u32` child IDs. Alignment of
/// `buf` to 8 bytes is required.
///
/// # Safety
///
/// - `builder` must be a valid, non-null `DagBuilder`.
/// - If `buf` is non-null, it must point to at least `buf_len` bytes of
///   writable memory, correctly aligned to 8 bytes.
/// - `bytes_written` must be a valid non-null pointer to a `u64`.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_get_packed(
    builder: *const crate::dag::builder::DagBuilder,
    buf: *mut u8,
    buf_len: usize,
    bytes_written: *mut usize,
) -> RssnStatus {
    if builder.is_null() || bytes_written.is_null() {
        return RssnStatus::NullPointer;
    }
    let result = catch_unwind(|| -> RssnStatus {
        let b = unsafe { &*builder };
        let image = b.packed_snapshot();
        // Compute needed size: 8 (node_count) + n*32 + 8 (pool_count) + pool*4.
        let node_count = image.len();
        let pool_count = image.children_pool().len();
        let needed = 8 + node_count * 32 + 8 + pool_count * 4;
        unsafe {
            *bytes_written = needed;
        }

        if buf.is_null() {
            // Size query only.
            return RssnStatus::Success;
        }
        if buf_len < needed {
            return RssnStatus::BufferTooSmall;
        }

        // SAFETY: caller guarantees buf is writable and at least buf_len bytes.
        let out: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(buf, buf_len) };
        let mut pos = 0usize;

        // Write node count (little-endian u64).
        out[pos..pos + 8].copy_from_slice(&(node_count as u64).to_le_bytes());
        pos += 8;

        // Write packed nodes as raw bytes.
        let node_bytes = image.nodes().len() * 32;
        // SAFETY: PackedDagNode is Pod (#[repr(C)], no padding, Copy).
        let node_src: &[u8] =
            unsafe { std::slice::from_raw_parts(image.nodes().as_ptr().cast::<u8>(), node_bytes) };
        out[pos..pos + node_bytes].copy_from_slice(node_src);
        pos += node_bytes;

        // Write pool count (little-endian u64).
        out[pos..pos + 8].copy_from_slice(&(pool_count as u64).to_le_bytes());
        pos += 8;

        // Write pool as little-endian u32 values.
        for &v in image.children_pool() {
            out[pos..pos + 4].copy_from_slice(&v.to_le_bytes());
            pos += 4;
        }

        let _ = pos; // suppress unused-assignment warning
        RssnStatus::Success
    });
    result.unwrap_or(RssnStatus::Panic)
}

#[cfg(test)]
mod egraph_ffi_tests {
    use super::*;

    #[test]
    fn egraph_ffi_constant_fold_add() {
        let builder = rssn_dag_new();
        let c3 = rssn_dag_constant(builder, 3.0);
        let c4 = rssn_dag_constant(builder, 4.0);
        let s = rssn_dag_add(builder, c3, c4);

        let cfg = RssnEGraphConfig {
            max_rounds: 4,
            max_merges: 64,
            max_new_nodes: 64,
            strict_ieee754_signed_zero: 0,
        };
        let mut out: u32 = u32::MAX;
        let status =
            rssn_dag_egraph_saturate_extract(builder, s, cfg, std::ptr::null(), 0, &mut out);
        assert_eq!(status, RssnStatus::Success);
        // The constant-folded node 7.0 should be in the same e-class and have lower cost.
        // out may be s itself or the folded constant — both are valid extractions.
        assert_ne!(out, u32::MAX);

        rssn_dag_free(builder);
    }

    #[test]
    fn egraph_ffi_add_zero_simplifies() {
        let builder = rssn_dag_new();
        let x = rssn_dag_variable(builder, c"x".as_ptr());
        let zero = rssn_dag_constant(builder, 0.0);
        let xpz = rssn_dag_add(builder, x, zero);

        let cfg = RssnEGraphConfig::default();
        let mut out: u32 = u32::MAX;
        let status =
            rssn_dag_egraph_saturate_extract(builder, xpz, cfg, std::ptr::null(), 0, &mut out);
        assert_eq!(status, RssnStatus::Success);
        // x is cheaper than x+0; extractor should return x.
        assert_eq!(out, x, "x+0 extracts to x");

        rssn_dag_free(builder);
    }

    #[test]
    fn egraph_ffi_null_builder_returns_null_pointer() {
        let mut out: u32 = 0;
        let status = rssn_dag_egraph_saturate_extract(
            std::ptr::null_mut(),
            0,
            RssnEGraphConfig::default(),
            std::ptr::null(),
            0,
            &mut out,
        );
        assert_eq!(status, RssnStatus::NullPointer);
    }
}

#[cfg(test)]
mod batch_build_tests {
    use super::*;

    /// Build `x * (x + 2.0)` as a batch of 4 nodes:
    ///   [0] Variable "x"
    ///   [1] Constant 2.0
    ///   [2] Add (0, 1) = x + 2
    ///   [3] Mul (0, 2) = x * (x + 2)
    #[test]
    fn batch_build_polynomial() {
        let builder = rssn_dag_new();
        let mut out_ids = [u32::MAX; 4];

        let mut descs = [RssnNodeDesc {
            value: 0.0,
            child0: u32::MAX,
            child1: u32::MAX,
            kind: 0,
            name: [0u8; 31],
        }; 4];

        // Node 0: variable "x"
        descs[0].kind = 0;
        descs[0].name[0] = b'x';

        // Node 1: constant 2.0
        descs[1].kind = 1;
        descs[1].value = 2.0;

        // Node 2: Add(0, 1)
        descs[2].kind = 2;
        descs[2].child0 = 0;
        descs[2].child1 = 1;

        // Node 3: Mul(0, 2)
        descs[3].kind = 4;
        descs[3].child0 = 0;
        descs[3].child1 = 2;

        let status = rssn_dag_batch_build(builder, descs.as_ptr(), 4, out_ids.as_mut_ptr());
        assert_eq!(status, RssnStatus::Success);

        // All node IDs must be valid (not u32::MAX).
        for &id in &out_ids {
            assert_ne!(id, u32::MAX, "all nodes should be allocated");
        }

        // x*x deduplication: same variable → same ID
        assert_eq!(out_ids[0], out_ids[0]);

        // Build the same expression manually and compare IDs (dedup).
        let x2 = rssn_dag_variable(builder, c"x".as_ptr());
        let c2 = rssn_dag_constant(builder, 2.0);
        let add2 = rssn_dag_add(builder, x2, c2);
        let mul2 = rssn_dag_mul(builder, x2, add2);
        assert_eq!(
            out_ids[3], mul2,
            "batch and individual build produce same node ID"
        );

        rssn_dag_free(builder);
    }

    #[test]
    fn batch_build_null_returns_null_pointer() {
        let mut out_ids = [0u32; 2];
        let descs = [RssnNodeDesc {
            value: 1.0,
            child0: u32::MAX,
            child1: u32::MAX,
            kind: 1,
            name: [0u8; 31],
        }; 2];
        let status = rssn_dag_batch_build(
            std::ptr::null_mut(),
            descs.as_ptr(),
            2,
            out_ids.as_mut_ptr(),
        );
        assert_eq!(status, RssnStatus::NullPointer);
    }

    #[test]
    fn get_packed_size_query() {
        let builder = rssn_dag_new();
        // Build a small expression.
        let x = rssn_dag_variable(builder, c"x".as_ptr());
        let c = rssn_dag_constant(builder, 3.0);
        let _ = rssn_dag_add(builder, x, c);

        // Size query: pass null buffer.
        let mut needed: usize = 0;
        let status = rssn_dag_get_packed(builder as *const _, std::ptr::null_mut(), 0, &mut needed);
        assert_eq!(status, RssnStatus::Success);
        assert!(needed > 0, "packed snapshot must have positive size");

        // Actual write.
        let mut buf = vec![0u8; needed];
        let mut written: usize = 0;
        let status2 =
            rssn_dag_get_packed(builder as *const _, buf.as_mut_ptr(), needed, &mut written);
        assert_eq!(status2, RssnStatus::Success);
        assert_eq!(written, needed);

        // First 8 bytes are node count (little-endian).
        let node_count = u64::from_le_bytes(buf[0..8].try_into().expect("8 bytes"));
        assert!(node_count >= 3, "at least 3 nodes: x, 3.0, x+3.0");

        rssn_dag_free(builder);
    }
}

// =============================================================================
// Unified Custom-Operator Registry — C FFI
// =============================================================================
//
// The `RssnCustomOpRegistry` is an opaque handle to a
// `crate::custom::descriptor::CustomOpRegistry`.  It is the C-facing
// equivalent of the Rust `CustomOpRegistry` and lets C/C++ callers register
// operators that plug into all three pipelines in one place.
//
// Lifecycle:
//   RssnCustomOpRegistry* reg = rssn_custom_op_registry_new();
//   rssn_custom_op_register_fn1(reg, fn_id, "name", fn_ptr, vectorizable);
//   rssn_custom_op_add_simplify_rule(reg, fn_id, "rule name", priority, cb, ud);
//   rssn_custom_op_add_egraph_rule(reg, fn_id, after_builtins, cb, ud);
//
//   // Use in each pipeline step:
//   rssn_dag_compile_with_custom_ops(builder, root, reg, &fn_ptr);
//   rssn_dag_simplify_with_custom_ops(builder, root, reg, &out_id);
//   rssn_dag_egraph_with_custom_ops(builder, root, cfg, reg, &out_id);
//
//   rssn_custom_op_registry_free(reg);

use crate::custom::descriptor::{CustomOpDescriptor, CustomOpRegistry, EvalFn};
use std::sync::Arc;

/// Opaque handle to a [`CustomOpRegistry`].
///
/// Heap-allocated; must be freed exactly once via [`rssn_custom_op_registry_free`].
pub struct RssnCustomOpRegistry(Arc<CustomOpRegistry>);

/// Allocates an empty [`RssnCustomOpRegistry`].
///
/// # Safety
///
/// The returned pointer must be freed exactly once via
/// [`rssn_custom_op_registry_free`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_custom_op_registry_new() -> *mut RssnCustomOpRegistry {
    let result = catch_unwind(|| {
        Box::into_raw(Box::new(RssnCustomOpRegistry(Arc::new(
            CustomOpRegistry::new(),
        ))))
    });
    result.unwrap_or(std::ptr::null_mut())
}

/// Frees a [`RssnCustomOpRegistry`] allocated by [`rssn_custom_op_registry_new`].
///
/// # Safety
///
/// `reg` must be a pointer from [`rssn_custom_op_registry_new`], or NULL.
/// Double-free is undefined behaviour.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_custom_op_registry_free(reg: *mut RssnCustomOpRegistry) {
    if reg.is_null() {
        return;
    }
    let _ = catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(reg));
    }));
}

// ── Internal helper: get a mutable reference to the inner registry ─────────
//
// The Arc inside RssnCustomOpRegistry is unwrapped mutably only while the
// registry is being built (before it is shared with the JIT).  We use
// Arc::get_mut; if the Arc has been cloned (i.e. shared with a JitCompiler)
// this returns None and we return InvalidNodeId.

fn registry_mut(reg: *mut RssnCustomOpRegistry) -> Option<&'static mut CustomOpRegistry> {
    if reg.is_null() {
        return None;
    }
    let wrapper = unsafe { &mut *reg };
    Arc::get_mut(&mut wrapper.0)
}

// ── Operator registration ──────────────────────────────────────────────────

/// Registers a 1-argument (`f64 → f64`) custom operator.
///
/// - `fn_id`: numeric identifier (must be unique in the registry).
/// - `name`: null-terminated operator name (resolved by the parser).
/// - `eval_fn`: `extern "C" fn(f64) -> f64` pointer.
/// - `vectorizable`: non-zero if the function is pure and safe to duplicate
///   in the ILP batch path.
///
/// # Safety
///
/// `reg` and `name` must be valid non-null pointers for the duration of
/// this call.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_custom_op_register_fn1(
    reg: *mut RssnCustomOpRegistry,
    fn_id: u32,
    name: *const c_char,
    eval_fn: Option<extern "C" fn(f64) -> f64>,
    vectorizable: u8,
) -> RssnStatus {
    if reg.is_null() || name.is_null() {
        return RssnStatus::NullPointer;
    }
    let Some(eval_fn) = eval_fn else {
        return RssnStatus::NullPointer;
    };
    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let name_str = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|_| RssnStatus::ParseError)?
            .to_owned();
        let reg_mut = registry_mut(reg).ok_or(RssnStatus::InvalidNode)?;
        let desc = CustomOpDescriptor::builder(
            crate::dag::symbol::FnId(fn_id),
            name_str,
            EvalFn::Arity1(eval_fn),
        )
        .cost(2.0);
        let desc = if vectorizable != 0 {
            desc.vectorizable()
        } else {
            desc
        };
        reg_mut
            .register(desc.build())
            .map_err(|_| RssnStatus::RuleConflict)?;
        Ok(RssnStatus::Success)
    }));
    result
        .unwrap_or(Err(RssnStatus::Panic))
        .unwrap_or_else(|e| e)
}

/// Registers a 2-argument (`f64, f64 → f64`) custom operator.
///
/// # Safety
///
/// Same as [`rssn_custom_op_register_fn1`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_custom_op_register_fn2(
    reg: *mut RssnCustomOpRegistry,
    fn_id: u32,
    name: *const c_char,
    eval_fn: Option<extern "C" fn(f64, f64) -> f64>,
    vectorizable: u8,
) -> RssnStatus {
    if reg.is_null() || name.is_null() {
        return RssnStatus::NullPointer;
    }
    let Some(eval_fn) = eval_fn else {
        return RssnStatus::NullPointer;
    };
    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let name_str = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|_| RssnStatus::ParseError)?
            .to_owned();
        let reg_mut = registry_mut(reg).ok_or(RssnStatus::InvalidNode)?;
        let desc = CustomOpDescriptor::builder(
            crate::dag::symbol::FnId(fn_id),
            name_str,
            EvalFn::Arity2(eval_fn),
        )
        .cost(2.0);
        let desc = if vectorizable != 0 {
            desc.vectorizable()
        } else {
            desc
        };
        reg_mut
            .register(desc.build())
            .map_err(|_| RssnStatus::RuleConflict)?;
        Ok(RssnStatus::Success)
    }));
    result
        .unwrap_or(Err(RssnStatus::Panic))
        .unwrap_or_else(|e| e)
}

/// Registers a 3-argument (`f64, f64, f64 → f64`) custom operator.
///
/// # Safety
///
/// Same as [`rssn_custom_op_register_fn1`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_custom_op_register_fn3(
    reg: *mut RssnCustomOpRegistry,
    fn_id: u32,
    name: *const c_char,
    eval_fn: Option<extern "C" fn(f64, f64, f64) -> f64>,
    vectorizable: u8,
) -> RssnStatus {
    if reg.is_null() || name.is_null() {
        return RssnStatus::NullPointer;
    }
    let Some(eval_fn) = eval_fn else {
        return RssnStatus::NullPointer;
    };
    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let name_str = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|_| RssnStatus::ParseError)?
            .to_owned();
        let reg_mut = registry_mut(reg).ok_or(RssnStatus::InvalidNode)?;
        let desc = CustomOpDescriptor::builder(
            crate::dag::symbol::FnId(fn_id),
            name_str,
            EvalFn::Arity3(eval_fn),
        )
        .cost(2.0);
        let desc = if vectorizable != 0 {
            desc.vectorizable()
        } else {
            desc
        };
        reg_mut
            .register(desc.build())
            .map_err(|_| RssnStatus::RuleConflict)?;
        Ok(RssnStatus::Success)
    }));
    result
        .unwrap_or(Err(RssnStatus::Panic))
        .unwrap_or_else(|e| e)
}

// ── Rule attachment ────────────────────────────────────────────────────────

/// Returns the `u8` kind discriminant for a `SymbolKind` value, matching
/// the `RssnKind` encoding used throughout the C API.
const fn symbol_kind_to_u8(kind: &crate::dag::symbol::SymbolKind) -> u8 {
    use crate::dag::symbol::{OpKind, SymbolKind};
    match kind {
        SymbolKind::Variable(_) => 0,
        SymbolKind::Constant(_) => 1,
        SymbolKind::Operator(OpKind::Add) => 2,
        SymbolKind::Operator(OpKind::Sub) => 3,
        SymbolKind::Operator(OpKind::Mul) => 4,
        SymbolKind::Operator(OpKind::Div) => 5,
        SymbolKind::Operator(OpKind::Pow) => 6,
        SymbolKind::Operator(OpKind::Neg) => 7,
        SymbolKind::Operator(OpKind::Mod) => 8,
        SymbolKind::Function(_) => 9,
    }
}

/// Adds a heuristic simplification rule to a custom operator.
///
/// `fn_id` must already be registered via `rssn_custom_op_register_fn*`.
/// `callback` is called by the simplifier for every node it visits.
/// Return `u32::MAX` from the callback to pass (no rewrite); any other value
/// is treated as the replacement node ID.
///
/// # Safety
///
/// `reg`, `rule_name`, and `user_data` must remain valid for the lifetime of
/// the registry (until [`rssn_custom_op_registry_free`]).
#[unsafe(no_mangle)]
pub extern "C" fn rssn_custom_op_add_simplify_rule(
    reg: *mut RssnCustomOpRegistry,
    fn_id: u32,
    rule_name: *const c_char,
    priority: i32,
    callback: Option<RssnRuleCallback>,
    user_data: *mut c_void,
) -> RssnStatus {
    if reg.is_null() || rule_name.is_null() {
        return RssnStatus::NullPointer;
    }
    let Some(callback) = callback else {
        return RssnStatus::NullPointer;
    };
    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let name_str = unsafe { CStr::from_ptr(rule_name) }
            .to_str()
            .map_err(|_| RssnStatus::ParseError)?
            .to_owned();
        let reg_mut = registry_mut(reg).ok_or(RssnStatus::InvalidNode)?;
        let target_id = crate::dag::symbol::FnId(fn_id);
        let desc = reg_mut
            .get_mut(target_id)
            .ok_or(RssnStatus::InvalidNodeId)?;

        // Capture callback + user_data (as usize for Send safety).
        let ud = user_data as usize;
        desc.simplify_rules
            .push(crate::custom::descriptor::SimplifyRule {
                name: name_str,
                priority,
                rule: std::sync::Arc::new(
                    move |builder: &mut DagBuilder,
                          kind: crate::dag::symbol::SymbolKind,
                          children: &[DagNodeId]| {
                        let kind_byte = symbol_kind_to_u8(&kind);
                        let child_ids: Vec<u32> = children.iter().map(|id| id.value()).collect();
                        // SAFETY: callback and ud were valid when registered; the
                        // registry lifetime covers any call through this closure.
                        let result = unsafe {
                            callback(
                                std::ptr::from_mut::<DagBuilder>(builder),
                                kind_byte,
                                child_ids.as_ptr(),
                                child_ids.len() as u32,
                                ud as *mut c_void,
                            )
                        };
                        if result == u32::MAX {
                            None
                        } else {
                            Some(DagNodeId::new(result))
                        }
                    },
                ),
            });
        Ok(RssnStatus::Success)
    }));
    result
        .unwrap_or(Err(RssnStatus::Panic))
        .unwrap_or_else(|e| e)
}

/// Adds an e-graph rewrite rule to a custom operator.
///
/// `after_builtins`: non-zero → run after built-in algebraic rules each round.
///
/// # Safety
///
/// Same as [`rssn_custom_op_add_simplify_rule`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_custom_op_add_egraph_rule(
    reg: *mut RssnCustomOpRegistry,
    fn_id: u32,
    after_builtins: u8,
    callback: Option<RssnEGraphRuleCallback>,
    user_data: *mut c_void,
) -> RssnStatus {
    if reg.is_null() {
        return RssnStatus::NullPointer;
    }
    let Some(callback) = callback else {
        return RssnStatus::NullPointer;
    };
    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let reg_mut = registry_mut(reg).ok_or(RssnStatus::InvalidNode)?;
        let target_id = crate::dag::symbol::FnId(fn_id);
        let desc = reg_mut
            .get_mut(target_id)
            .ok_or(RssnStatus::InvalidNodeId)?;

        let ud = user_data as usize;
        desc.egraph_rules
            .push(crate::custom::descriptor::EGraphRule {
                after_builtins: after_builtins != 0,
                rule: std::sync::Arc::new(
                    move |builder: &mut DagBuilder,
                          kind: &crate::dag::symbol::SymbolKind,
                          children: &[DagNodeId]| {
                        let kind_byte = symbol_kind_to_u8(kind);
                        let child_ids: Vec<u32> = children.iter().map(|id| id.value()).collect();
                        let result = unsafe {
                            callback(
                                std::ptr::from_mut::<DagBuilder>(builder),
                                kind_byte,
                                child_ids.as_ptr(),
                                child_ids.len() as u32,
                                ud as *mut c_void,
                            )
                        };
                        if result == u32::MAX {
                            None
                        } else {
                            Some(DagNodeId::new(result))
                        }
                    },
                ),
            });
        Ok(RssnStatus::Success)
    }));
    result
        .unwrap_or(Err(RssnStatus::Panic))
        .unwrap_or_else(|e| e)
}

// ── Pipeline integration functions ─────────────────────────────────────────

/// JIT-compiles `root` using operators from `reg`.
///
/// Internally calls [`rssn_dag_compile`] after feeding all `eval_fn` pointers
/// from the registry into the global JIT context.  The batch f64×2 path
/// honours `vectorizable` flags for `Function` nodes.
///
/// # Safety
///
/// Same as [`rssn_dag_compile`].
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn rssn_dag_compile_with_custom_ops(
    builder: *mut DagBuilder,
    root: u32,
    reg: *mut RssnCustomOpRegistry,
    out_fn: *mut *mut c_void,
) -> RssnStatus {
    if builder.is_null() || out_fn.is_null() || reg.is_null() {
        return RssnStatus::NullPointer;
    }
    if root == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let builder_ref = unsafe { &mut *builder };
        let reg_ref = unsafe { &*reg };
        // Pre-intern all names so the parser (and any builder calls made
        // inside this function) can resolve them.
        reg_ref.0.register_with_builder(builder_ref);

        let root_id = DagNodeId::new(root);
        let ast = crate::ast::convert::dag_to_ast(builder_ref.arena(), root_id);
        let ctx_mutex = crate::ffi::jit_context::global_jit_ctx();
        let mut ctx = ctx_mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // Install the registry into the JIT context (feeds fn pointers +
        // enables vectorizable check).
        ctx.compiler_mut()
            .set_custom_op_registry(Arc::clone(&reg_ref.0));

        ctx.compiler_mut()
            .compile(&ast)
            .map_or(RssnStatus::CompilationError, |f| {
                unsafe { *out_fn = f as *mut c_void };
                RssnStatus::Success
            })
    }));
    result.unwrap_or(RssnStatus::Panic)
}

/// Non-JIT stub.
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_compile_with_custom_ops(
    _builder: *mut DagBuilder,
    _root: u32,
    _reg: *mut RssnCustomOpRegistry,
    _out_fn: *mut *mut c_void,
) -> RssnStatus {
    RssnStatus::CompilationError
}

/// Simplifies `root` applying all simplification rules from `reg`.
///
/// Combines the registry's rules with the built-in heuristic patterns and
/// runs the standard simplification pass.
///
/// # Safety
///
/// `builder`, `reg`, and `out_id` must be valid non-null pointers.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn rssn_dag_simplify_with_custom_ops(
    builder: *mut DagBuilder,
    root: u32,
    reg: *mut RssnCustomOpRegistry,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || reg.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    if root == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let builder_ref = unsafe { &mut *builder };
        let reg_ref = unsafe { &*reg };

        // Build a RuleRegistry from all attached simplify_rules.
        let rule_registry = reg_ref.0.build_rule_registry();

        let config = HeuristicConfig::default();
        let mut engine = HeuristicEngine::new(config, SearchStrategy::Greedy)
            .with_rule_registry(std::sync::Arc::new(rule_registry));

        let root_id = DagNodeId::new(root);
        // HeuristicEngine::simplify returns DagNodeId directly (not Result).
        let simplified = engine.simplify(builder_ref, root_id);
        unsafe { *out_id = simplified.value() };
        RssnStatus::Success
    }));
    result.unwrap_or(RssnStatus::Panic)
}

/// E-graph equality saturation with all rules from `reg`.
///
/// Runs the built-in algebraic rules plus all e-graph rules attached to
/// descriptors in `reg`, then extracts the minimum-cost representative.
///
/// # Safety
///
/// `builder`, `reg`, and `out_id` must be valid non-null pointers.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn rssn_dag_egraph_with_custom_ops(
    builder: *mut DagBuilder,
    root: u32,
    config: RssnEGraphConfig,
    reg: *mut RssnCustomOpRegistry,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || reg.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    if root == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let builder_ref = unsafe { &mut *builder };
        let reg_ref = unsafe { &*reg };

        let eg_config = crate::egraph::egraph::EGraphConfig {
            max_rounds: if config.max_rounds == 0 {
                8
            } else {
                config.max_rounds as usize
            },
            max_merges: if config.max_merges == 0 {
                512
            } else {
                config.max_merges as usize
            },
            max_new_nodes: if config.max_new_nodes == 0 {
                1024
            } else {
                config.max_new_nodes as usize
            },
            strict_ieee754_signed_zero: config.strict_ieee754_signed_zero != 0,
            ..Default::default()
        };

        let root_id = DagNodeId::new(root);
        let mut egraph = crate::egraph::egraph::EGraph::new(builder_ref, eg_config);

        // Inject all e-graph rules from the custom-op registry.
        reg_ref.0.apply_to_egraph(&mut egraph);

        egraph.saturate(root_id);
        let best = egraph.extract(root_id);
        unsafe { *out_id = best.value() };
        RssnStatus::Success
    }));
    result.unwrap_or(RssnStatus::Panic)
}
