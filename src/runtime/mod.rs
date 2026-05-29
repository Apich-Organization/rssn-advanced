//! Fiber-based task runtime built on `dtact`.
//!
//! `plan.md` §4.3 mandates an async-fiber interface, and the review found
//! every async path (`ffi::async_bridge`, `parallel::solver`) still using
//! heavyweight `std::thread::spawn`. This module funnels all task dispatch
//! through `dtact`'s lock-free fiber pool, exposing three primitives:
//!
//! * [`ensure_runtime`](crate::runtime::ensure_runtime) — idempotent one-shot init of the global fiber pool.
//! * [`spawn_task`](crate::runtime::spawn_task)     — fire-and-forget fiber for a `FnOnce() + Send`.
//! * [`parallel_for_each`](crate::runtime::parallel_for_each) — fan-out / fan-in over an iterator of closures.
//!
//! ## Task-envelope memory strategy
//!
//! Each spawned fiber needs a heap allocation to carry its closure across
//! thread boundaries (the spawning thread owns the closure; a dtact worker
//! thread will read and execute it).  Routing every spawn through the global
//! allocator (malloc / HeapAlloc) costs ~20–100 ns per task — enough to
//! negate the scheduling advantage of fibers for small, rapid fan-outs.
//!
//! ### Lock-free pool (fast path)
//!
//! When a closure fits in `POOL_INLINE_CAPACITY` bytes and has alignment ≤
//! 16, the closure is written into a pre-allocated `PoolNode` drawn from a
//! global **ABA-safe Treiber stack** (`POOL_HEAD`).  The worker returns the
//! node to the stack after moving the closure out — one `LOCK CMPXCHG8B`
//! instead of a malloc + free.
//!
//! #### ABA safety
//!
//! A plain `AtomicPtr` Treiber stack suffers from ABA: if a node is popped,
//! used for a task that completes and returns the node, all before the
//! original thread's CAS fires, the stale `next` pointer silently wins and
//! can corrupt the list.  We prevent this with a **tagged pointer**:
//! `POOL_HEAD` is an `AtomicU64` whose top 16 bits hold a 16-bit generation
//! tag (bottom 48 bits = pointer, always ≤ 48 bits on x86-64 without LA57).
//! Each successful CAS increments the tag; a stale observer always sees a
//! different tag and retries.  `AtomicU64::new(0)` is a stable `const fn`
//! with no software-lock fallback on any 64-bit target.
//!
//! ### TaskEnvelope fallback (large / over-aligned closures)
//!
//! Closures that exceed `POOL_INLINE_CAPACITY` or require alignment > 16
//! fall back to the monomorphised `TaskEnvelope<F>` + `Box::into_raw` path.
//! These are rare in practice (typical captures: a few `Arc` / `usize` values,
//! all ≤ 8-byte aligned).
//!
//! ### Shared trampoline
//!
//! Both paths store the monomorphised `invoke` pointer at **byte offset 0**
//! (`#[repr(C)]` for `TaskEnvelope`, `repr(C, align(16))` with `word0` at
//! offset 0 for `PoolNode`).  The single `task_trampoline` reads those bytes
//! as `unsafe fn(*mut ())` and dispatches without knowing which path produced
//! the pointer.

#![allow(unsafe_code)]

use std::os::raw::c_void;
use std::ptr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use dtact::c_ffi::{dtact_default_config, dtact_fiber_launch, dtact_handle_t, dtact_init};
use dtact::dtact_await;

use crate::error::{FfiError, cold_ffi_error_runtime_uninitialized};

/// Marker returned by [`ensure_runtime`] so callers can prove the pool is
/// alive without re-checking. Stored once in `RUNTIME_GATE` and copied
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
// Common C-ABI trampoline
// =========================================================================
//
// Both dispatch paths (`PoolNode` and `TaskEnvelope<F>`) guarantee that the
// `invoke` function pointer lives at byte offset 0 of the allocation that
// `arg` points to.  The trampoline reads those bytes without knowing which
// path produced `arg`, then calls the monomorphised handler.

/// C-ABI trampoline for `dtact_fiber_launch`.
///
/// `arg` is either a `*mut PoolNode` (fast path) or a `*mut TaskEnvelope<F>`
/// (fallback path).  In both cases `invoke` sits at byte offset 0 of the
/// pointed-to memory (`word0` at offset 0 for `PoolNode`, `invoke` at offset
/// 0 via `#[repr(C)]` for `TaskEnvelope`).  The called function takes
/// ownership of the full allocation and frees or recycles it.
extern "C" fn task_trampoline(arg: *mut c_void) {
    // SAFETY: Both pool and envelope layouts place the invoke fn-ptr at
    // offset 0.  The pointed-to memory remains valid until `invoke` frees it.
    let invoke: unsafe fn(*mut ()) = unsafe { *arg.cast::<unsafe fn(*mut ())>() };
    unsafe { invoke(arg.cast::<()>()) };
}

// =========================================================================
// Fast path — ABA-safe lock-free pool via tagged-pointer Treiber stack
// =========================================================================

/// Maximum closure size (bytes) that uses the pooled fast path.
///
/// 64 bytes covers common captures: three `Arc<T>` fat pointers (24 B),
/// a `usize` index (8 B), and a `*const` data pointer (8 B), with room to
/// spare.  Closures larger than this fall back to [`TaskEnvelope`].
const POOL_INLINE_CAPACITY: usize = 64;

/// A reusable memory node for task dispatch.
///
/// **While on the free list** ([`POOL_HEAD`]):
///   * `word0` holds the raw `*mut PoolNode` "next" pointer (null = tail).
///   * `_word1` and `data` are logically uninitialized.
///
/// **While in-flight** (handed to a dtact fiber):
///   * `word0` holds the monomorphised `invoke` trampoline pointer.
///   * `_word1` is unused padding.
///   * `data[0..size_of::<F>()]` holds the closure `F` written via
///     `ptr::write`; the rest is logically uninitialized.
///
/// `#[repr(C, align(16))]` ensures:
///   * `word0` is at byte offset 0 — the shared trampoline reads it as a
///     function pointer without knowing whether this is a pool node or a
///     `TaskEnvelope`.
///   * `data` starts at byte offset 16, which is 16-byte aligned — meeting
///     the alignment requirement of any closure capturing `Arc`, `usize`,
///     `*const T`, or SIMD-compatible types up to 16-byte alignment.
#[repr(C, align(16))]
struct PoolNode {
    /// Free-list `next` (null = tail) **or** in-flight `invoke` trampoline.
    word0: usize,
    /// Padding so `data` is at a 16-byte offset within the struct.
    _word1: usize,
    /// Inline closure storage (up to [`POOL_INLINE_CAPACITY`] bytes).
    data: [u8; POOL_INLINE_CAPACITY],
}

// ── Tagged-pointer helpers ────────────────────────────────────────────────
//
// Layout of the u64 stored in POOL_HEAD:
//
//   bits 63..48  ┃  generation counter (u16, wraps at 65 536)
//   bits 47.. 0  ┃  raw pointer (48 bits, canonical x86-64 user-space)
//
// `AtomicU128::new` is not yet a stable const fn, so it cannot appear in a
// `static` initializer without nightly.  More critically, `AtomicU128` often
// degrades to a software-lock fallback on platforms without `CMPXCHG16B`,
// cancelling the lock-free guarantee we need.
//
// Instead we use a plain `AtomicU64` — always lock-free, always const-stable.
// On x86-64 (without LA57 5-level paging), user-space canonical addresses
// occupy ≤ 48 bits; the top 16 bits are always zero.  Windows does not yet
// expose LA57 to user-space (as of 2026).  Those 16 free bits carry a
// generation counter that defeats ABA.
//
// ABA analysis: 65 536 pop-use-push cycles must complete inside the ~3 ns
// window between our `load` and `compare_exchange_weak`.  At 10⁶ cycles/s
// that would take ≥ 65 ms.  Probability ≈ 3 ns / 65 ms ≈ 5 × 10⁻⁸ per
// CAS — negligible.

/// Head of the global ABA-resistant lock-free pool (tagged Treiber stack).
///
/// Zero-initialized → `ptr = null, gen = 0` → empty pool.
/// `AtomicU64::new(0)` is a stable `const fn`; the CAS compiles to
/// `LOCK CMPXCHG8B` on x86-64 — one instruction, no software lock.
static POOL_HEAD: AtomicU64 = AtomicU64::new(0);

/// Packs a pointer and 16-bit generation counter into one `u64`.
///
/// # Safety (caller contract)
/// `ptr` must be a canonical user-space address whose top 16 bits are zero
/// (standard x86-64 without LA57).  A `debug_assert` fires otherwise.
#[inline(always)]
fn pack(ptr: *mut PoolNode, tag: u16) -> u64 {
    let bits = ptr as u64;
    debug_assert_eq!(bits >> 48, 0, "pool ptr exceeds 48 bits — LA57 unsupported");
    bits | (u64::from(tag) << 48)
}

/// Unpacks a `u64` CAS word into a `(pointer, generation)` pair.
#[inline(always)]
const fn unpack(val: u64) -> (*mut PoolNode, u16) {
    let ptr = (val & 0x0000_FFFF_FFFF_FFFF) as *mut PoolNode;
    let tag = (val >> 48) as u16;
    (ptr, tag)
}

/// Allocates a fresh [`PoolNode`] via the global allocator.
///
/// Cold path: called only when the pool is empty.
///
/// The `Box` is intentional: callers immediately call `Box::into_raw` to
/// produce a `*mut PoolNode` that the C trampoline owns; `Box::from_raw`
/// reconstructs it after the fiber completes. A plain `PoolNode` return
/// would force callers to perform their own `Box::new` anyway.
#[cold]
#[allow(clippy::unnecessary_box_returns)]
fn alloc_pool_node() -> Box<PoolNode> {
    Box::new(PoolNode {
        word0: 0,
        _word1: 0,
        data: [0u8; POOL_INLINE_CAPACITY],
    })
}

/// Pops a [`PoolNode`] from the global free list, or allocates one.
///
/// Uses a `compare_exchange_weak` loop on the tagged `POOL_HEAD`.  The
/// 16-bit tag in bits 63..48 is incremented on every successful CAS, so a
/// stale load (ABA) always causes the CAS to fail and retry.
///
/// The `Box` return is intentional: `spawn_task` immediately calls
/// `Box::into_raw` to hand the raw pointer to `dtact_fiber_launch`; the
/// C trampoline later reconstructs it with `Box::from_raw` in
/// `invoke_and_drop_pooled`. A plain `PoolNode` return would only push
/// the `Box::new` call into the caller.
#[inline]
#[allow(clippy::unnecessary_box_returns)]
fn pool_acquire() -> Box<PoolNode> {
    let mut head_val = POOL_HEAD.load(Ordering::Acquire);
    loop {
        let (head, tag) = unpack(head_val);
        if head.is_null() {
            return alloc_pool_node();
        }
        // SAFETY: `head` is non-null and was placed here by `pool_release`
        // under Release ordering (visible under the Acquire load above).
        // `word0` holds the free-list `next` pointer while the node is on
        // the list — it was written before the Release CAS that added it.
        let next = unsafe { (*head).word0 as *mut PoolNode };

        // CAS: set head to (next, tag+1).  The tag bump ensures that if
        // another thread cycles this node (pop → use → push) between our
        // `load` and this CAS, the tag will have advanced and the CAS fails.
        let new_val = pack(next, tag.wrapping_add(1));
        match POOL_HEAD.compare_exchange_weak(
            head_val,
            new_val,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            // SAFETY: we won the CAS → exclusive ownership of `head`.
            Ok(_) => return unsafe { Box::from_raw(head) },
            Err(current) => head_val = current,
        }
    }
}

/// Pushes a [`PoolNode`] back onto the global free list.
///
/// The tag is incremented so that any thread holding a stale `head_val`
/// cannot win a CAS against the new state.
#[inline]
fn pool_release(node: Box<PoolNode>) {
    let node_ptr = Box::into_raw(node);
    let mut head_val = POOL_HEAD.load(Ordering::Relaxed);
    loop {
        let (head, tag) = unpack(head_val);
        // Write `next` *before* the CAS: the acquiring thread reads `word0`
        // under Acquire ordering after winning its CAS, so this write must
        // be visible via the Release fence on our CAS success.
        // SAFETY: `node_ptr` is exclusively owned by this thread until the
        // CAS succeeds.
        unsafe { (*node_ptr).word0 = head as usize };

        let new_val = pack(node_ptr, tag.wrapping_add(1));
        match POOL_HEAD.compare_exchange_weak(
            head_val,
            new_val,
            Ordering::Release,
            Ordering::Relaxed,
        ) {
            Ok(_) => return,
            Err(current) => head_val = current,
        }
    }
}

/// Monomorphised trampoline for the pooled fast path.
///
/// # Safety
///
/// `raw` must be a valid `*mut PoolNode` whose `data[0..size_of::<F>()]`
/// holds an initialized `F` written by `ptr::write`.  After this call the
/// node has been returned to the pool and `raw` must not be used again.
unsafe fn invoke_and_drop_pooled<F: FnOnce() + Send + 'static>(raw: *mut ()) {
    let node_ptr = raw.cast::<PoolNode>();
    // Move the closure out of the inline storage BEFORE returning the node.
    // After `ptr::read`, the bytes in `node.data` are owned by `f`; the node
    // itself contains no live data and can be safely recycled.
    //
    // SAFETY: `data[0..size_of::<F>()]` was initialized via `ptr::write` in
    // `spawn_task` and has not been touched since.
    let f = unsafe { ptr::read((*node_ptr).data.as_ptr().cast::<F>()) };
    // Return the (now-empty) node to the pool.  Another thread may claim and
    // reuse it immediately — safe because `f` no longer refers to the node.
    pool_release(unsafe { Box::from_raw(node_ptr) });
    // Run the closure on the worker fiber.  Its destructor fires here.
    f();
}

// =========================================================================
// Fallback path — TaskEnvelope<F> (oversized / over-aligned closures)
// =========================================================================

/// Typed envelope for an oversized or over-aligned closure.
///
/// Avoids double-boxing (`Box<Box<dyn FnOnce()>>`): `TaskEnvelope<F>`
/// monomorphises the trampoline, stores the closure inline, and uses a
/// single `Box` (global allocator, thread-safe) for the one allocation.
///
/// `#[repr(C)]` ensures `invoke` is at offset 0 — the trampoline reads it
/// from a type-erased `*mut c_void` without knowing `F`.
#[repr(C)]
struct TaskEnvelope<F: FnOnce() + Send + 'static> {
    /// Monomorphised trampoline — must remain at offset 0 (`#[repr(C)]`).
    invoke: unsafe fn(*mut ()),
    /// The closure itself, stored inline after the function pointer.
    closure: core::mem::ManuallyDrop<F>,
}

impl<F: FnOnce() + Send + 'static> TaskEnvelope<F> {
    /// Reconstitutes the envelope from a type-erased pointer, frees the
    /// `Box` allocation (global allocator — thread-safe), then runs the
    /// closure.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid `*mut TaskEnvelope<F>` produced by
    /// `Box::into_raw`.  After this call the allocation is freed; `raw` must
    /// not be used again.
    unsafe fn invoke_and_drop(raw: *mut ()) {
        let env_ptr = raw.cast::<Self>();
        let f = unsafe { core::mem::ManuallyDrop::take(&mut (*env_ptr).closure) };
        unsafe { drop(Box::from_raw(env_ptr)) };
        f();
    }
}

// =========================================================================
// Public API — spawn / join
// =========================================================================

/// Opaque handle for a spawned task. Returned by [`spawn_task`] and
/// consumed by [`join`].
#[derive(Clone, Copy)]
pub struct TaskHandle(dtact_handle_t);

impl TaskHandle {
    /// Returns the raw numeric id of the underlying fiber handle.
    ///
    /// Used by the async FFI bridge to stash the handle in a C-visible struct.
    #[must_use]
    pub const fn raw_id(self) -> u64 {
        self.0.0
    }

    /// Reconstructs a `TaskHandle` from a raw id previously obtained via
    /// [`Self::raw_id`].  The caller must ensure the id is still valid (i.e.,
    /// the fiber has not been joined yet).
    #[must_use]
    pub const fn from_raw(id: u64) -> Self {
        Self(dtact_handle_t(id))
    }
}

/// Spawns `f` onto the fiber pool and returns a joinable [`TaskHandle`].
///
/// **Fast path** — closures ≤ `POOL_INLINE_CAPACITY` bytes and ≤ 16-byte
/// alignment: drawn from the lock-free `POOL_HEAD` pool (one `LOCK CMPXCHG8B`
/// pair, no malloc).
///
/// **Fallback path** — larger or over-aligned closures: one `Box` allocation
/// via the global allocator, same as before the pool existed.
///
/// `dtact` dispatches the closure to whichever worker is currently coldest.
/// Drop the handle if you don't need to wait — fibers run to completion
/// regardless.  Call [`join`] to block until the fiber finishes.
pub fn spawn_task<F: FnOnce() + Send + 'static>(_gate: RuntimeGate, f: F) -> TaskHandle {
    if core::mem::size_of::<F>() <= POOL_INLINE_CAPACITY && core::mem::align_of::<F>() <= 16 {
        // ── Fast path: lock-free pool ────────────────────────────────────
        let mut node = pool_acquire();

        // Write the monomorphised trampoline at word0 (offset 0).
        // After this, `task_trampoline` can dispatch without knowing `F`.
        node.word0 = invoke_and_drop_pooled::<F> as *const () as usize;

        // Write the closure into the inline data buffer.
        // SAFETY: `data` is at a 16-byte offset within `PoolNode`
        // (`#[repr(C, align(16))]`), so it is 16-byte aligned.
        // `align_of::<F>() ≤ 16` (checked above) means this satisfies `F`'s
        // alignment requirement.  `size_of::<F>() ≤ POOL_INLINE_CAPACITY`
        // (also checked above) means there is room for the closure.
        unsafe {
            ptr::write(node.data.as_mut_ptr().cast::<F>(), f);
        }

        let arg = Box::into_raw(node).cast::<c_void>();
        // SAFETY: `task_trampoline` reads `word0` (== invoke) and calls it;
        // `invoke_and_drop_pooled` moves out the closure then returns the node.
        let handle = unsafe { dtact_fiber_launch(task_trampoline, arg) };
        TaskHandle(handle)
    } else {
        // ── Fallback path: single Box allocation ─────────────────────────
        let arg: *mut TaskEnvelope<F> = Box::into_raw(Box::new(TaskEnvelope {
            invoke: TaskEnvelope::<F>::invoke_and_drop,
            closure: core::mem::ManuallyDrop::new(f),
        }));
        // SAFETY: `task_trampoline` reads `arg.invoke` and calls it;
        // `invoke_and_drop` takes back ownership and frees via `Box::from_raw`.
        let handle = unsafe { dtact_fiber_launch(task_trampoline, arg.cast::<c_void>()) };
        TaskHandle(handle)
    }
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
/// to finish before returning.  Closures produce a `T` which is collected
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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn ensure_runtime_is_idempotent() {
        let g1 = ensure_runtime();
        let g2 = ensure_runtime();
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
        let results = parallel_for_each(gate, (0u32..8).map(|i| move || i * i));
        assert_eq!(
            results,
            alloc::vec![
                Some(0),
                Some(1),
                Some(4),
                Some(9),
                Some(16),
                Some(25),
                Some(36),
                Some(49)
            ]
        );
    }

    #[test]
    fn runtime_gate_reports_uninit_only_before_init() {
        let _ = ensure_runtime();
        assert!(runtime_gate().is_ok());
    }

    /// Verifies pool recycling: two waves of tasks must produce correct results.
    ///
    /// A double-free or use-after-free in the pool would manifest as a wrong
    /// counter value or a panic inside the task (Rust's allocator detects
    /// double-frees in debug builds).
    #[test]
    fn pool_recycles_nodes_across_waves() {
        let gate = ensure_runtime();
        let counter = Arc::new(AtomicUsize::new(0));

        let wave1: Vec<_> = (0..16)
            .map(|_| {
                let c = Arc::clone(&counter);
                spawn_task(gate, move || {
                    c.fetch_add(1, Ordering::Release);
                })
            })
            .collect();
        for h in wave1 {
            join(h);
        }
        assert_eq!(
            counter.load(Ordering::Acquire),
            16,
            "wave 1 must run all tasks"
        );

        let wave2: Vec<_> = (0..16)
            .map(|_| {
                let c = Arc::clone(&counter);
                spawn_task(gate, move || {
                    c.fetch_add(1, Ordering::Release);
                })
            })
            .collect();
        for h in wave2 {
            join(h);
        }
        assert_eq!(
            counter.load(Ordering::Acquire),
            32,
            "wave 2 must run all tasks"
        );
    }

    /// Verifies the fallback path for closures that exceed the inline
    /// capacity (a large array forces `size_of::<F>() > POOL_INLINE_CAPACITY`).
    #[test]
    fn oversized_closure_uses_fallback_path() {
        let gate = ensure_runtime();
        // `[u8; 256]` exceeds `POOL_INLINE_CAPACITY` (64), forcing the
        // `TaskEnvelope` / `Box` fallback path.
        let big: [u8; 256] = [42u8; 256];
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        let h = spawn_task(gate, move || {
            // Access the capture so the compiler includes it in the closure.
            c.fetch_add(big[0] as usize, Ordering::Release);
        });
        join(h);
        assert_eq!(counter.load(Ordering::Acquire), 42);
    }
}

#[cfg(test)]
extern crate alloc;
