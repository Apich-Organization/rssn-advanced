//! Fiber-based task runtime built on `dtact`.
//!
//! `plan.md` §4.3 mandates an async-fiber interface, and the review found
//! every async path (`ffi::async_bridge`, `parallel::solver`) still using
//! heavyweight `std::thread::spawn`. This module funnels all task dispatch
//! through `dtact`'s lock-free fiber pool, exposing three primitives:
//!
//! * [`ensure_runtime`] — idempotent one-shot init of the global fiber pool.
//! * [`spawn_task`]     — fire-and-forget fiber for a `FnOnce() + Send`.
//! * [`parallel_for_each`] — fan-out / fan-in over an iterator of closures.
//!
//! The wrapper is intentionally thin. Higher-level scheduling concerns
//! (work stealing, NUMA pinning) belong to `dtact` itself; we just give the
//! rest of the crate an ergonomic Rust surface.

#![allow(unsafe_code)]

use std::os::raw::c_void;
use std::sync::OnceLock;

use dtact::c_ffi::{dtact_default_config, dtact_fiber_launch, dtact_handle_t, dtact_init};
use dtact::dtact_await;

use crate::error::{FfiError, cold_ffi_error_runtime_uninitialized};

/// Marker returned by [`ensure_runtime`] so callers can prove the pool is
/// alive without re-checking. Stored once in [`RUNTIME_GATE`] and copied
/// freely thereafter.
#[derive(Clone, Copy)]
pub struct RuntimeGate {
    _private: (),
}

/// Initialization sentinel. `dtact_init` itself uses `OnceLock` internally,
/// but we wrap it again so that callers from this crate share a single
/// thread-safe init path and never race on the `dtact_default_config()`
/// argument construction.
static RUNTIME_GATE: OnceLock<RuntimeGate> = OnceLock::new();

/// Initializes the global `dtact` runtime on first call; subsequent calls
/// are O(1) and return the same `RuntimeGate`.
///
/// Safe to call from any thread, including pre-`main` static init paths.
pub fn ensure_runtime() -> RuntimeGate {
    *RUNTIME_GATE.get_or_init(|| {
        // SAFETY: `dtact_default_config` returns a fully-initialized config
        // on every call, and `dtact_init` reads from the pointer only for
        // the duration of the call.
        let cfg = dtact_default_config();
        let cfg_ptr: *const _ = &raw const cfg;
        unsafe {
            let _ = dtact_init(cfg_ptr);
        }
        // `dtact_init` only constructs the pool; worker threads are not
        // launched until `Runtime::start()` is called. Without this the
        // fiber pool would accept submissions but never schedule them
        // and any `dtact_await` would block forever.
        if let Some(rt) = dtact::GLOBAL_RUNTIME.get() {
            rt.start();
        }
        RuntimeGate { _private: () }
    })
}

/// Returns the active runtime gate if the pool is initialized.
///
/// FFI entry points use this to refuse work rather than implicitly start
/// the runtime.
///
/// # Errors
///
/// Returns [`FfiError::RuntimeUninitialized`] if [`ensure_runtime`] has not
/// been called yet on this process.
pub fn runtime_gate() -> Result<RuntimeGate, FfiError> {
    RUNTIME_GATE
        .get()
        .copied()
        .map_or_else(cold_ffi_error_runtime_uninitialized, Ok)
}

// =========================================================================
// Single fiber spawn
// =========================================================================

/// Typed envelope for a single closure — packed into one heap allocation.
///
/// The old `spawn_task` double-boxed: `Box<Box<dyn FnOnce() + Send>>`. Each
/// `Box` is a fat pointer (16 bytes), so the outer box stored 16 bytes of
/// vtable + data pointer plus a fresh allocation for the inner fat pointer.
/// `TaskEnvelope<F>` stores the closure and the trampoline function pointer
/// in a single monomorphised struct — one allocation.
///
/// `#[repr(C)]` ensures `invoke` is the first field, so the trampoline can
/// read it from a type-erased pointer without knowing `F`.
#[repr(C)]
struct TaskEnvelope<F: FnOnce() + Send + 'static> {
    /// Monomorphised trampoline — the only field whose offset must be stable.
    invoke: unsafe fn(*mut ()),
    /// The closure itself, stored inline.
    closure: core::mem::ManuallyDrop<F>,
}

impl<F: FnOnce() + Send + 'static> TaskEnvelope<F> {
    /// Extracts the closure from `raw`, drops the envelope, then calls `f`.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid `*mut TaskEnvelope<F>` that was produced by
    /// `Box::into_raw`. After this call the allocation is freed.
    unsafe fn invoke_and_drop(raw: *mut ()) {
        let mut env = unsafe { Box::from_raw(raw.cast::<Self>()) };
        // Move the closure out before the Box is dropped.
        // `ManuallyDrop` never runs `F`'s destructor automatically, so
        // dropping `env` afterwards is safe even though the field was moved.
        let f = unsafe { core::mem::ManuallyDrop::take(&mut env.closure) };
        drop(env); // free the allocation (ManuallyDrop<F> drop is a no-op)
        f();       // run the closure; its destructor fires here
    }
}

/// C-ABI trampoline for `dtact_fiber_launch`. `arg` is a raw `TaskEnvelope<F>`
/// pointer (type-erased). Reads the `invoke` function pointer at offset 0
/// (`#[repr(C)]` guarantees this) and dispatches to the monomorphised handler.
extern "C" fn task_trampoline(arg: *mut c_void) {
    // SAFETY: `arg` was produced by `Box::into_raw::<TaskEnvelope<F>>` for some
    // `F`. With `#[repr(C)]`, the `invoke` field is at offset 0, so reading it
    // via a `*const unsafe fn(*mut ())` cast is well-defined.
    let invoke: unsafe fn(*mut ()) =
        unsafe { *arg.cast::<unsafe fn(*mut ())>() };
    unsafe { invoke(arg.cast::<()>()) };
}

/// Opaque handle for a spawned task. Returned by [`spawn_task`] and
/// consumed by [`join`].
#[derive(Clone, Copy)]
pub struct TaskHandle(dtact_handle_t);

impl TaskHandle {
    /// Returns the raw numeric id of the underlying fiber handle.
    ///
    /// Used by the async FFI bridge to stash the handle in a C-visible struct.
    #[must_use]
    pub fn raw_id(self) -> u64 {
        self.0.0
    }

    /// Reconstructs a `TaskHandle` from a raw id previously obtained via
    /// [`Self::raw_id`]. The caller must ensure the id is still valid (i.e.,
    /// the fiber has not been joined yet).
    #[must_use]
    pub fn from_raw(id: u64) -> Self {
        Self(dtact_handle_t(id))
    }
}

/// Spawns `f` onto the fiber pool with a single heap allocation.
///
/// `dtact` then runs it on whichever worker is currently coldest.
///
/// Returns a [`TaskHandle`] that can be passed to [`join`] later. If you
/// don't care about completion, drop the handle (the fiber will still
/// run to completion — fibers are detached by default).
pub fn spawn_task<F: FnOnce() + Send + 'static>(_gate: RuntimeGate, f: F) -> TaskHandle {
    // Single allocation: `TaskEnvelope<F>` stores both the trampoline
    // function pointer and the closure inline, replacing the old double-box.
    let env = Box::new(TaskEnvelope::<F> {
        invoke: TaskEnvelope::<F>::invoke_and_drop,
        closure: core::mem::ManuallyDrop::new(f),
    });
    let arg: *mut TaskEnvelope<F> = Box::into_raw(env);
    // SAFETY: `task_trampoline` reads `arg.invoke` and calls it; `invoke`
    // reconstructs and drops the `TaskEnvelope` via `Box::from_raw`.
    let handle = unsafe { dtact_fiber_launch(task_trampoline, arg.cast::<c_void>()) };
    TaskHandle(handle)
}

/// Blocks the calling thread (or yields the calling fiber) until the task
/// behind `handle` finishes.
pub fn join(handle: TaskHandle) {
    dtact_await(handle.0);
}

// =========================================================================
// Fan-out / fan-in
// =========================================================================

/// Runs each closure in `tasks` on its own fiber and waits for all of them
/// to finish before returning. Closures produce a `T` which is collected
/// into the returned `Vec` in input order.
///
/// Panics inside individual tasks are caught via [`std::panic::catch_unwind`]
/// and mapped to `None` in the output; the returned `Vec` contains all slots
/// including `None` values for panicking tasks, preserving input order.
///
/// Uses a lock-free write path: each fiber writes directly into its own
/// pre-allocated slot in an `UnsafeCell<Vec<Option<T>>>` using the slot
/// index as the exclusive key — no `Mutex` contention between workers.
/// The fan-in join barrier (`dtact_await`) provides the happens-before
/// edge that makes the final read of all slots safe.
///
/// This is the workhorse used by `parallel::solver` and `ffi::async_bridge`
/// to replace the `std::thread::spawn` pattern.
pub fn parallel_for_each<I, F, T>(gate: RuntimeGate, tasks: I) -> Vec<Option<T>>
where
    I: IntoIterator<Item = F>,
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    use std::cell::UnsafeCell;
    use std::sync::Arc;

    let tasks: Vec<F> = tasks.into_iter().collect();
    let n = tasks.len();

    // SAFETY: `UnsafeCell<Vec<Option<T>>>` is not `Sync` by default.
    // We assert it here because:
    //   (a) fibers write to disjoint indices (no aliased mutable refs), and
    //   (b) the join loop below provides the happens-before fence before
    //       the caller ever reads from `slots`.
    struct SendSync<T>(T);
    unsafe impl<T: Send> Send for SendSync<T> {}
    unsafe impl<T: Send> Sync for SendSync<T> {}

    // Pre-allocate one slot per task. Each fiber owns exactly one index
    // and writes to it without touching any other slot — no locking needed.
    let slots: Arc<SendSync<UnsafeCell<Vec<Option<T>>>>> =
        Arc::new(SendSync(UnsafeCell::new((0..n).map(|_| None).collect())));

    let mut handles: Vec<TaskHandle> = Vec::with_capacity(n);
    for (i, task) in tasks.into_iter().enumerate() {
        let slots_arc = Arc::clone(&slots);
        handles.push(spawn_task(gate, move || {
            // Catch panics so one failing task doesn't abort the whole fan-out.
            let value = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task)).ok();
            // SAFETY: `i` is unique per fiber; no two fibers share an index.
            unsafe {
                let vec_ptr: *mut Vec<Option<T>> = slots_arc.0.get();
                (&mut *vec_ptr)[i] = value;
            }
        }));
    }

    for h in handles {
        join(h);
    }

    // SAFETY: all fibers have been joined; we hold the only live reference
    // to `slots`. Unwrapping the Arc gives exclusive access to the inner Vec.
    let inner = Arc::try_unwrap(slots)
        .unwrap_or_else(|_| unreachable!("all fibers have been joined; Arc is unique"));
    inner.0.into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn ensure_runtime_is_idempotent() {
        let g1 = ensure_runtime();
        let g2 = ensure_runtime();
        // `RuntimeGate` is a unit-shaped token; we only assert that the
        // second call did not panic and that the gate is now retrievable.
        let _ = (g1, g2);
        assert!(runtime_gate().is_ok());
    }

    #[test]
    fn spawn_task_runs_closure() {
        let gate = ensure_runtime();
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        let h = spawn_task(gate, move || {
            c.fetch_add(7, Ordering::Release);
        });
        join(h);
        assert_eq!(counter.load(Ordering::Acquire), 7);
    }

    #[test]
    fn parallel_for_each_preserves_order_and_runs_all() {
        let gate = ensure_runtime();
        let results = parallel_for_each(
            gate,
            (0u32..8).map(|i| move || i * i),
        );
        assert_eq!(results, alloc::vec![Some(0), Some(1), Some(4), Some(9), Some(16), Some(25), Some(36), Some(49)]);
    }

    #[test]
    fn runtime_gate_reports_uninit_only_before_init() {
        // After `ensure_runtime`, the gate must be available. We cannot
        // safely test the pre-init branch in a process-shared test since
        // every test in this module triggers init.
        let _ = ensure_runtime();
        assert!(runtime_gate().is_ok());
    }
}

#[cfg(test)]
extern crate alloc;
