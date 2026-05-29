//! Async CFFI bridge for multi-language integration.
//!
//! Provides a callback-based asynchronous C API that allows external
//! languages to invoke long-running simplification, compilation, and
//! one-shot evaluation without blocking the calling thread.
//!
//! Dispatches through the `dtact` fiber pool; the fiber handle is returned
//! to the caller and joined explicitly via a `rssn_async_*_join` function.
//!
//! # Handle types
//!
//! | Handle | Async operation | Join function |
//! |--------|----------------|---------------|
//! | [`RssnAsyncHandle`] | simplify | [`rssn_async_join`] |
//! | [`RssnAsyncCompileHandle`] | simplify → compile | [`rssn_async_compile_join`] |
//! | [`RssnAsyncEvalHandle`] | simplify → compile → execute | [`rssn_async_eval_join`] |
//!
//! # Safety contract
//!
//! The caller must call the corresponding `*_join` function before freeing
//! the `DagBuilder` (and, for eval, the `vars` array) passed to the async
//! function.  Failure to join is a use-after-free.

#![allow(unsafe_code)]

use std::os::raw::c_void;
use std::panic::catch_unwind;

use super::types::RssnStatus;
use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use crate::heuristic::{HeuristicConfig, HeuristicEngine, SearchStrategy};
use crate::runtime::{TaskHandle, ensure_runtime, join, spawn_task};

// =========================================================================
// v2: joinable async handle
// =========================================================================

/// Opaque async simplification handle. Returned by
/// [`rssn_dag_simplify_async_v2`]; pass to [`rssn_async_join`] to block
/// until the fiber completes and obtain the result.
///
/// The handle stores the `TaskHandle` value as a `u64` for C ABI
/// compatibility. A value of `u64::MAX` signals an early error (null
/// builder, etc.).
#[repr(C)]
pub struct RssnAsyncHandle {
    /// Internal fiber handle, or `u64::MAX` on error.
    pub handle_id: u64,
    /// Predicted simplified root (written by the fiber before completion).
    /// Zero until the fiber writes it; read only after [`rssn_async_join`].
    pub simplified_root: u32,
    /// Status code set by the fiber.
    pub status: RssnStatus,
}

/// Fires a simplification fiber and returns an opaque handle.
///
/// The caller must call [`rssn_async_join`] before freeing `builder`, making
/// the use-after-free hazard explicit and auditable from C.
///
/// The returned [`RssnAsyncHandle`] must be freed with [`rssn_async_join`].
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn rssn_dag_simplify_async(
    builder: *mut DagBuilder,
    root: u32,
) -> *mut RssnAsyncHandle {
    let handle_box = Box::new(RssnAsyncHandle {
        handle_id: u64::MAX,
        simplified_root: u32::MAX,
        status: RssnStatus::NullPointer,
    });
    let handle_ptr = Box::into_raw(handle_box);

    if builder.is_null() {
        // handle_ptr already carries NullPointer status; caller joins immediately.
        return handle_ptr;
    }

    let builder_addr = builder as usize;
    let handle_addr = handle_ptr as usize;

    let gate = ensure_runtime();
    let task = spawn_task(gate, move || {
        let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
            let builder_ref = unsafe { &mut *(builder_addr as *mut DagBuilder) };
            let root_id = DagNodeId::new(root);
            let config = HeuristicConfig::default();
            let mut engine = HeuristicEngine::new(config, SearchStrategy::Greedy);
            engine.simplify(builder_ref, root_id).value()
        }));

        // SAFETY: `handle_ptr` was heap-allocated above and is not freed
        // until `rssn_async_join` is called (which happens after this fiber
        // completes — the join blocks).
        let h = unsafe { &mut *(handle_addr as *mut RssnAsyncHandle) };
        match result {
            Ok(simplified) => {
                h.simplified_root = simplified;
                h.status = RssnStatus::Success;
            }
            Err(_) => {
                h.status = RssnStatus::Panic;
            }
        }
    });

    // Stash the task handle id so `rssn_async_join` can await it.
    // SAFETY: we wrote handle_ptr just above; the fiber hasn't touched it yet
    // because it's racing to acquire builder_ref after we return.
    unsafe {
        (*handle_ptr).handle_id = task.raw_id();
    }

    handle_ptr
}

/// Blocks until the fiber behind `handle` completes, then frees the handle
/// and writes the result to `*out_root` (if non-null).
///
/// Returns the final [`RssnStatus`]. After this call, `builder` may be freed.
///
/// # Safety
///
/// `handle` must have been obtained from [`rssn_dag_simplify_async_v2`] and
/// must not be used after this call.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn rssn_async_join(handle: *mut RssnAsyncHandle, out_root: *mut u32) -> RssnStatus {
    if handle.is_null() {
        return RssnStatus::NullPointer;
    }
    let h = unsafe { &*handle };
    if h.handle_id != u64::MAX {
        let task_handle = TaskHandle::from_raw(h.handle_id);
        join(task_handle);
    }
    let h = unsafe { &*handle };
    if !out_root.is_null() {
        unsafe { *out_root = h.simplified_root };
    }
    let status = h.status;
    // Free the handle.
    let _ = unsafe { Box::from_raw(handle) };
    status
}

// =========================================================================
// Async compile handle — simplify then JIT-compile
// =========================================================================
//
// Fires a fiber that runs: dag_to_ast → JitCompiler::compile.
// This amortises the first-compile latency (Cranelift IR generation + linking)
// off the calling thread while keeping the global JIT context mutex held only
// inside the fiber.
//
// The compiled function pointer is stored as a `u64` (raw usize cast) so the
// handle struct is `Send` without additional unsafe markers.

/// Opaque handle returned by [`rssn_dag_compile_async`].
///
/// Pass to [`rssn_async_compile_join`] to block until the fiber finishes and
/// retrieve the compiled function pointer.
///
/// The `fn_ptr_bits` field stores the raw function pointer as a `u64`; it is
/// undefined until after a successful join.
#[repr(C)]
pub struct RssnAsyncCompileHandle {
    /// Internal fiber handle id, or `u64::MAX` on early error.
    pub handle_id: u64,
    /// Compiled function pointer bits (valid only after a successful join).
    /// Cast to `double (*)(const double*)` before calling.
    pub fn_ptr_bits: u64,
    /// Status set by the fiber; valid after join.
    pub status: RssnStatus,
}

// SAFETY: `fn_ptr_bits` is a raw function pointer stored as an integer.
// It is only written by the fiber and only read after `join` — no concurrent
// access.  The `DagBuilder` lifetime is the caller's responsibility.
unsafe impl Send for RssnAsyncCompileHandle {}

/// Simplifies and JIT-compiles an expression asynchronously.
///
/// Fires a fiber that runs `simplify(builder, root)` followed by
/// `JitCompiler::compile(ast)` using the process-level JIT context.
/// Returns immediately with an opaque handle; call [`rssn_async_compile_join`]
/// to block until the fiber finishes and obtain the function pointer.
///
/// The caller **must** call [`rssn_async_compile_join`] before freeing `builder`.
///
/// Returns a non-null handle even on early error (null `builder`) — the handle
/// will have `status = NullPointer` and joining it is a no-op.
///
/// # Safety
///
/// - `builder` must remain valid and unmodified until after
///   [`rssn_async_compile_join`] returns.
/// - The returned handle must be freed by [`rssn_async_compile_join`].
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn rssn_dag_compile_async(
    builder: *mut DagBuilder,
    root: u32,
) -> *mut RssnAsyncCompileHandle {
    let handle_box = Box::new(RssnAsyncCompileHandle {
        handle_id: u64::MAX,
        fn_ptr_bits: 0,
        status: RssnStatus::NullPointer,
    });
    let handle_ptr = Box::into_raw(handle_box);

    if builder.is_null() {
        return handle_ptr;
    }

    let builder_addr = builder as usize;
    let handle_addr = handle_ptr as usize;

    let gate = ensure_runtime();
    let task = spawn_task(gate, move || {
        let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
            let builder_ref = unsafe { &mut *(builder_addr as *mut DagBuilder) };
            let root_id = DagNodeId::new(root);

            // Simplify first to reduce JIT work.
            let config = HeuristicConfig::default();
            let mut engine = HeuristicEngine::new(config, SearchStrategy::Greedy);
            let simplified_id = engine.simplify(builder_ref, root_id);

            // Convert to AST and compile.
            let ast = crate::ast::convert::dag_to_ast(builder_ref.arena(), simplified_id);
            let ctx_mutex = crate::ffi::jit_context::global_jit_ctx();
            let mut ctx = ctx_mutex
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            ctx.compiler_mut().compile(&ast).map(|f| f as usize as u64)
        }));

        let h = unsafe { &mut *(handle_addr as *mut RssnAsyncCompileHandle) };
        match result {
            Ok(Ok(fn_bits)) => {
                h.fn_ptr_bits = fn_bits;
                h.status = RssnStatus::Success;
            }
            Ok(Err(_)) => {
                h.status = RssnStatus::CompilationError;
            }
            Err(_) => {
                h.status = RssnStatus::Panic;
            }
        }
    });

    unsafe { (*handle_ptr).handle_id = task.raw_id() };
    handle_ptr
}

/// Stub for non-JIT builds: returns an immediately-joined error handle.
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn rssn_dag_compile_async(
    _builder: *mut DagBuilder,
    _root: u32,
) -> *mut RssnAsyncCompileHandle {
    Box::into_raw(Box::new(RssnAsyncCompileHandle {
        handle_id: u64::MAX,
        fn_ptr_bits: 0,
        status: RssnStatus::CompilationError,
    }))
}

/// Blocks until the compile fiber completes, writes the function pointer to
/// `*out_fn` (if non-null), frees the handle, and returns the fiber's status.
///
/// After this call `builder` may be freed.  The function pointer written to
/// `*out_fn` (if `status == Success`) is valid for as long as the process-level
/// JIT module lives — i.e. for the lifetime of the process.
///
/// # Safety
///
/// `handle` must have been obtained from [`rssn_dag_compile_async`] and must
/// not be used after this call.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn rssn_async_compile_join(
    handle: *mut RssnAsyncCompileHandle,
    out_fn: *mut *mut c_void,
) -> RssnStatus {
    if handle.is_null() {
        return RssnStatus::NullPointer;
    }
    let h = unsafe { &*handle };
    if h.handle_id != u64::MAX {
        let task_handle = TaskHandle::from_raw(h.handle_id);
        join(task_handle);
    }
    let h = unsafe { &*handle };
    if !out_fn.is_null() {
        unsafe { *out_fn = h.fn_ptr_bits as *mut c_void };
    }
    let status = h.status;
    let _ = unsafe { Box::from_raw(handle) };
    status
}

// =========================================================================
// Async one-shot eval handle — simplify, compile, then execute
// =========================================================================
//
// Useful for benchmarks and first-time evaluations where the expression has
// never been compiled before.  The caller provides the variable array up-front;
// the fiber holds a raw pointer to it and the caller must keep it alive until
// after `rssn_async_eval_join` returns.
//
// Result is a `f64` stored in the handle.

/// Opaque handle returned by [`rssn_dag_eval_async`].
///
/// Pass to [`rssn_async_eval_join`] to block until the fiber finishes and
/// retrieve the computed `f64` result.
#[repr(C)]
pub struct RssnAsyncEvalHandle {
    /// Internal fiber handle id, or `u64::MAX` on early error.
    pub handle_id: u64,
    /// Computed result (valid only after a successful join).
    pub result: f64,
    /// Status set by the fiber; valid after join.
    pub status: RssnStatus,
}

// SAFETY: same reasoning as RssnAsyncCompileHandle.
unsafe impl Send for RssnAsyncEvalHandle {}

/// Simplifies, JIT-compiles, and evaluates an expression asynchronously.
///
/// The full pipeline — `simplify → compile → execute(vars)` — runs in a fiber.
/// Returns immediately; call [`rssn_async_eval_join`] to block and get the
/// `f64` result.
///
/// The caller **must** call [`rssn_async_eval_join`] before freeing `builder`
/// **or** `vars`.
///
/// # Safety
///
/// - `builder` must remain valid and unmodified until after
///   [`rssn_async_eval_join`] returns.
/// - `vars` must point to an array of at least as many `f64` values as there
///   are distinct variables in the expression, ordered by `SymbolId`.  It must
///   remain valid and unmodified until after [`rssn_async_eval_join`] returns.
/// - The returned handle must be freed by [`rssn_async_eval_join`].
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn rssn_dag_eval_async(
    builder: *mut DagBuilder,
    root: u32,
    vars: *const f64,
) -> *mut RssnAsyncEvalHandle {
    let handle_box = Box::new(RssnAsyncEvalHandle {
        handle_id: u64::MAX,
        result: f64::NAN,
        status: RssnStatus::NullPointer,
    });
    let handle_ptr = Box::into_raw(handle_box);

    if builder.is_null() || vars.is_null() {
        return handle_ptr;
    }

    let builder_addr = builder as usize;
    let vars_addr = vars as usize;
    let handle_addr = handle_ptr as usize;

    let gate = ensure_runtime();
    let task = spawn_task(gate, move || {
        let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
            let builder_ref = unsafe { &mut *(builder_addr as *mut DagBuilder) };
            let vars_ptr = vars_addr as *const f64;
            let root_id = DagNodeId::new(root);

            // Simplify.
            let config = HeuristicConfig::default();
            let mut engine = HeuristicEngine::new(config, SearchStrategy::Greedy);
            let simplified_id = engine.simplify(builder_ref, root_id);

            // Compile.
            let ast = crate::ast::convert::dag_to_ast(builder_ref.arena(), simplified_id);
            let compiled_fn = {
                let ctx_mutex = crate::ffi::jit_context::global_jit_ctx();
                let mut ctx = ctx_mutex
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                ctx.compiler_mut().compile(&ast)?
            };

            // Execute.
            Ok::<f64, crate::error::JitError>(compiled_fn(vars_ptr))
        }));

        let h = unsafe { &mut *(handle_addr as *mut RssnAsyncEvalHandle) };
        match result {
            Ok(Ok(val)) => {
                h.result = val;
                h.status = RssnStatus::Success;
            }
            Ok(Err(_)) => {
                h.status = RssnStatus::CompilationError;
            }
            Err(_) => {
                h.status = RssnStatus::Panic;
            }
        }
    });

    unsafe { (*handle_ptr).handle_id = task.raw_id() };
    handle_ptr
}

/// Stub for non-JIT builds.
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn rssn_dag_eval_async(
    _builder: *mut DagBuilder,
    _root: u32,
    _vars: *const f64,
) -> *mut RssnAsyncEvalHandle {
    Box::into_raw(Box::new(RssnAsyncEvalHandle {
        handle_id: u64::MAX,
        result: f64::NAN,
        status: RssnStatus::CompilationError,
    }))
}

/// Blocks until the eval fiber completes, writes the result to `*out_val`
/// (if non-null), frees the handle, and returns the fiber's status.
///
/// After this call `builder` and `vars` may be freed.
///
/// # Safety
///
/// `handle` must have been obtained from [`rssn_dag_eval_async`] and must
/// not be used after this call.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn rssn_async_eval_join(
    handle: *mut RssnAsyncEvalHandle,
    out_val: *mut f64,
) -> RssnStatus {
    if handle.is_null() {
        return RssnStatus::NullPointer;
    }
    let h = unsafe { &*handle };
    if h.handle_id != u64::MAX {
        let task_handle = TaskHandle::from_raw(h.handle_id);
        join(task_handle);
    }
    let h = unsafe { &*handle };
    if !out_val.is_null() {
        unsafe { *out_val = h.result };
    }
    let status = h.status;
    let _ = unsafe { Box::from_raw(handle) };
    status
}
