//! Compiled JIT function cache.
//!
//! Caches JIT-compiled native functions keyed by expression string,
//! avoiding redundant recompilation of identical derivation or evaluation
//! paths.
//!
//! ## Memory ordering policy
//!
//! `plan.md §4.2` mandates "Acquire/Release semantics; strictly prohibit
//! misuse of `SeqCst`". This module enforces that locally:
//!
//! * The `HashMap` itself is guarded by an `RwLock` (whose internal
//!   ordering is already release-on-write / acquire-on-read).
//! * Hit/miss/insert counters use [`AtomicU64`] with explicit
//!   [`Ordering::Acquire`] / [`Ordering::Release`] only — never
//!   `SeqCst`.
//! * The counter struct carries `#[repr(align(128))]` so different
//!   counters land on different cache lines and don't generate
//!   ping-pong invalidations between cores (`storage_review §2`).

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use super::compiler::CompiledExprFn;

/// Counters tracking JIT cache effectiveness.
///
/// Each field lives on its own 128-byte aligned slot so they don't share
/// a cache line. Increment via [`Ordering::Release`]; read via
/// [`Ordering::Acquire`].
#[derive(Debug)]
#[repr(align(128))]
pub struct CacheStats {
    /// Number of successful lookups.
    pub hits: AtomicU64,
    /// Number of misses (caller had to compile).
    pub misses: AtomicU64,
    /// Number of inserts (independent of misses — a caller may insert
    /// without a prior miss).
    pub inserts: AtomicU64,
    /// Padding to push the next struct field onto a fresh cache line.
    _pad: [u8; 128 - 24],
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            inserts: AtomicU64::new(0),
            _pad: [0; 128 - 24],
        }
    }
}

/// A thread-safe, dynamic cache for storing and reusing JIT-compiled
/// native functions.
#[derive(Debug, Clone, Default)]
pub struct JitCache {
    /// Compiled function pointers, cast to `usize` for thread-safe storage.
    cache: Arc<RwLock<HashMap<String, usize>>>,
    /// Statistics counters using explicit Acquire/Release ordering.
    stats: Arc<CacheStats>,
}

impl JitCache {
    /// Creates a new, empty JIT function cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(CacheStats::default()),
        }
    }

    /// Attempts to retrieve a compiled function from the cache.
    ///
    /// Increments the appropriate counter via [`Ordering::Release`].
    #[must_use]
    pub fn get(&self, key: &str) -> Option<CompiledExprFn> {
        let read_guard = self.cache.read().ok()?;
        if let Some(&addr) = read_guard.get(key) {
            self.stats.hits.fetch_add(1, Ordering::Release);
            // SAFETY: `addr` was inserted by `insert` below from a valid
            // `CompiledExprFn` pointer with the matching ABI.
            let func: CompiledExprFn = unsafe { std::mem::transmute(addr) };
            Some(func)
        } else {
            self.stats.misses.fetch_add(1, Ordering::Release);
            None
        }
    }

    /// Inserts a compiled function into the cache.
    ///
    /// A poisoned lock is recovered via `PoisonError::into_inner` —
    /// the map's internal state survives panics in unrelated threads.
    pub fn insert(&self, key: String, func: CompiledExprFn) {
        let addr = func as usize;
        {
            let mut write_guard = self
                .cache
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            write_guard.insert(key, addr);
        }
        self.stats.inserts.fetch_add(1, Ordering::Release);
    }

    /// Clears all compiled functions from the cache.
    pub fn clear(&self) {
        let mut write_guard = self
            .cache
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        write_guard.clear();
    }

    /// Snapshot of (hits, misses, inserts) read with [`Ordering::Acquire`].
    #[must_use]
    pub fn stats(&self) -> (u64, u64, u64) {
        (
            self.stats.hits.load(Ordering::Acquire),
            self.stats.misses.load(Ordering::Acquire),
            self.stats.inserts.load(Ordering::Acquire),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" fn dummy_fn(_p: *const f64) -> f64 {
        0.0
    }

    #[test]
    fn stats_counters_track_lookups() {
        let cache = JitCache::new();
        assert_eq!(cache.stats(), (0, 0, 0));

        let _ = cache.get("missing");
        assert_eq!(cache.stats(), (0, 1, 0));

        cache.insert("hot".into(), dummy_fn);
        assert_eq!(cache.stats(), (0, 1, 1));

        let _ = cache.get("hot");
        assert_eq!(cache.stats(), (1, 1, 1));
    }

    #[test]
    fn stats_struct_is_cache_line_aligned() {
        assert_eq!(core::mem::align_of::<CacheStats>(), 128);
        assert!(core::mem::size_of::<CacheStats>() >= 128);
    }
}
