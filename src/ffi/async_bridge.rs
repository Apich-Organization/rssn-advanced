//! Async CFFI bridge for multi-language integration.
//!
//! Provides a callback-based asynchronous C API that allows external
//! languages to invoke long-running simplification without blocking.

#![allow(unsafe_code)]

use std::os::raw::c_void;
use std::panic::catch_unwind;
use std::thread;
use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use crate::heuristic::{HeuristicEngine, HeuristicConfig, SearchStrategy};
use super::types::RssnStatus;

/// Callback signature for asynchronous simplification completion.
///
/// Parameters:
/// - `simplified_root`: The index of the simplified root node.
/// - `status`: The return status code.
/// - `user_data`: The raw user data context pointer passed to the async function.
pub type RssnSimplifyCallback = unsafe extern "C" fn(
    simplified_root: u32,
    status: RssnStatus,
    user_data: *mut c_void,
);

/// Simplifies a target expression asynchronously on a background thread.
///
/// Upon completion, the given `callback` is executed with the simplified root
/// node and status code.
#[unsafe(no_mangle)]
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

    // Cast raw pointers to Send-safe usize addresses to pass to thread safely
    let builder_addr = builder as usize;
    let callback_addr = callback as usize;
    let user_data_addr = user_data as usize;

    thread::spawn(move || {
        let result = catch_unwind(|| {
            let builder_ref = unsafe { &mut *(builder_addr as *mut DagBuilder) };
            let root_id = DagNodeId::new(root);

            let config = HeuristicConfig::default();
            let engine = HeuristicEngine::new(config, SearchStrategy::Greedy);
            
            engine.simplify(builder_ref.arena_mut(), root_id).value()
        });

        // Reconstruct raw callback and user data pointers within the thread boundary
        let raw_callback: RssnSimplifyCallback = unsafe { std::mem::transmute(callback_addr) };
        let raw_user_data = user_data_addr as *mut c_void;

        match result {
            Ok(simplified) => {
                unsafe { raw_callback(simplified, RssnStatus::Success, raw_user_data) };
            }
            Err(_) => {
                unsafe { raw_callback(u32::MAX, RssnStatus::Panic, raw_user_data) };
            }
        }
    });
}
