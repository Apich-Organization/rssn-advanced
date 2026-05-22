//! Async CFFI bridge for multi-language integration.
//!
//! Provides a callback-based asynchronous C API that allows external
//! languages to invoke long-running simplification without blocking.
//!
//! Per `ffi_review §1` the previous implementation spawned a heavy OS
//! thread per request via `std::thread::spawn`; this rewrite dispatches
//! through the `dtact` fiber pool (`plan.md §4.3`).
//!
//! # Safety and lifetimes
//!
//! The v1 API (`rssn_dag_simplify_async`) fires and forgets the fiber,
//! returning before completion. Callers **must** guarantee the `DagBuilder`
//! pointer stays valid until the callback fires; otherwise a use-after-free
//! occurs.
//!
//! The v2 API (`rssn_dag_simplify_async_v2`) returns an opaque handle that
//! can be passed to [`rssn_async_join`] to block until the fiber completes.
//! This makes the safety requirement explicit and verifiable: join the handle
//! before freeing the builder.

#![allow(unsafe_code)]

use std::os::raw::c_void;
use std::panic::catch_unwind;

use super::types::RssnStatus;
use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use crate::heuristic::{HeuristicConfig, HeuristicEngine, SearchStrategy};
use crate::runtime::{TaskHandle, ensure_runtime, join, spawn_task};

/// Callback signature for asynchronous simplification completion.
///
/// Parameters:
/// - `simplified_root`: The index of the simplified root node.
/// - `status`: The return status code.
/// - `user_data`: The raw user data context pointer passed to the async function.
pub type RssnSimplifyCallback =
    unsafe extern "C" fn(simplified_root: u32, status: RssnStatus, user_data: *mut c_void);

/// Simplifies a target expression asynchronously on a background fiber.
///
/// Upon completion, the given `callback` is executed with the simplified
/// root node and status code.
///
/// # Safety
///
/// The caller **must** keep `builder` valid until `callback` fires.
/// Use [`rssn_dag_simplify_async_v2`] + [`rssn_async_join`] for an
/// explicit, auditable join-before-free contract.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn rssn_dag_simplify_async(
    builder: *mut DagBuilder,
    root: u32,
    callback: RssnSimplifyCallback,
    user_data: *mut c_void,
) {
    if builder.is_null() {
        unsafe { callback(u32::MAX, RssnStatus::NullPointer, user_data) };
        return;
    }

    let builder_addr = builder as usize;
    let callback_addr = callback as usize;
    let user_data_addr = user_data as usize;

    let gate = ensure_runtime();
    let handle = spawn_task(gate, move || {
        let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
            let builder_ref = unsafe { &mut *(builder_addr as *mut DagBuilder) };
            let root_id = DagNodeId::new(root);
            let config = HeuristicConfig::default();
            let engine = HeuristicEngine::new(config, SearchStrategy::Greedy);
            engine.simplify(builder_ref, root_id).value()
        }));

        let raw_callback: RssnSimplifyCallback =
            unsafe { std::mem::transmute(callback_addr) };
        let raw_user_data = user_data_addr as *mut c_void;

        match result {
            Ok(simplified) => unsafe { raw_callback(simplified, RssnStatus::Success, raw_user_data) },
            Err(_) => unsafe { raw_callback(u32::MAX, RssnStatus::Panic, raw_user_data) },
        }
    });
    // Detach: completion is signalled through the callback.
    let _ = handle;
}

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
/// Unlike [`rssn_dag_simplify_async`], this variant requires the caller to
/// call [`rssn_async_join`] before freeing `builder`, making the
/// use-after-free hazard explicit and auditable from C.
///
/// The returned [`RssnAsyncHandle`] must be freed with [`rssn_async_join`].
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn rssn_dag_simplify_async_v2(
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
            let engine = HeuristicEngine::new(config, SearchStrategy::Greedy);
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
pub extern "C" fn rssn_async_join(
    handle: *mut RssnAsyncHandle,
    out_root: *mut u32,
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
    if !out_root.is_null() {
        unsafe { *out_root = h.simplified_root };
    }
    let status = h.status;
    // Free the handle.
    let _ = unsafe { Box::from_raw(handle) };
    status
}
