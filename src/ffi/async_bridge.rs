//! Async CFFI bridge for multi-language integration.
//!
//! Provides a callback-based asynchronous C API that allows external
//! languages to invoke long-running simplification without blocking.
//!
//! Per `ffi_review §1` the previous implementation spawned a heavy OS
//! thread per request via `std::thread::spawn`; this rewrite dispatches
//! through the `dtact` fiber pool (`plan.md §4.3`).

#![allow(unsafe_code)]

use std::os::raw::c_void;
use std::panic::catch_unwind;

use super::types::RssnStatus;
use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use crate::heuristic::{HeuristicConfig, HeuristicEngine, SearchStrategy};
use crate::runtime::{ensure_runtime, spawn_task};

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

    // Cast raw pointers to Send-safe usize addresses so the fiber
    // closure can move them across the dispatch boundary.
    let builder_addr = builder as usize;
    let callback_addr = callback as usize;
    let user_data_addr = user_data as usize;

    let gate = ensure_runtime();
    let handle = spawn_task(gate, move || {
        let result = catch_unwind(|| {
            // SAFETY: `builder_addr` was a valid `*mut DagBuilder` at
            // the call site; the C caller guarantees it stays valid for
            // the duration of the async request.
            let builder_ref = unsafe { &mut *(builder_addr as *mut DagBuilder) };
            let root_id = DagNodeId::new(root);

            let config = HeuristicConfig::default();
            let engine = HeuristicEngine::new(config, SearchStrategy::Greedy);
            engine.simplify(builder_ref, root_id).value()
        });

        // SAFETY: `callback_addr` came from a valid `RssnSimplifyCallback`
        // function pointer at the call site.
        let raw_callback: RssnSimplifyCallback =
            unsafe { std::mem::transmute(callback_addr) };
        let raw_user_data = user_data_addr as *mut c_void;

        match result {
            Ok(simplified) => unsafe {
                raw_callback(simplified, RssnStatus::Success, raw_user_data);
            },
            Err(_) => unsafe {
                raw_callback(u32::MAX, RssnStatus::Panic, raw_user_data);
            },
        }
    });
    // Detach the handle: completion is signalled through the user
    // callback, not via a join from C.
    let _ = handle;
}
