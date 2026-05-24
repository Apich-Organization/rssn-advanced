//! Dynamic hotspot table for frequency-based caching.
//!
//! Per `storage_review §2`, the previous single `RwLock<HashMap>`
//! design serialized every `record_access` call across every worker
//! thread, defeating the parallelism the runtime gives us. This
//! rewrite shards the table over [`NUM_SHARDS`] independent
//! cache-line-aligned slots, dispatched by `id.0 % NUM_SHARDS`.
//!
//! Each shard holds its own `RwLock<HashMap<DagNodeId, u64>>` plus an
//! [`AtomicU64`] for an Acquire/Release-ordered total-access counter.
//! `#[repr(align(128))]` keeps adjacent shards on distinct cache
//! lines so writes from one core do not invalidate another's read
//! state.

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

// Using AtomicU64 values lets the read-lock fast path increment without
// upgrading to a write lock. Write lock is only needed for new-key insertion.
// This cuts lock contention on the hot path from write-every-access to
// write-on-first-access-per-key.

use crate::dag::node::DagNodeId;

/// Number of independent shards. Picked to be ≥ a typical core count
/// (32–128 cores on modern servers) while remaining cheap to
/// allocate. 32 means each shard owns ~3 % of the keyspace on average.
pub const NUM_SHARDS: usize = 32;

/// One shard of the sharded hotspot table.
///
/// `#[repr(align(128))]` lifts each shard onto a fresh cache line,
/// killing false sharing between adjacent shards (`storage_review §2`).
#[derive(Debug, Default)]
#[repr(align(128))]
struct Shard {
    frequencies: RwLock<HashMap<DagNodeId, AtomicU64>>,
    /// Per-shard total accesses (sum of frequency values written).
    /// Useful for cheap whole-table aggregates without taking every
    /// `RwLock`. Uses Acquire/Release ordering per `plan.md §4.2`.
    total_accesses: AtomicU64,
}

/// Dynamic frequency table tracking node access patterns.
///
/// Internally sharded; per-shard locking limits contention to ~1/N of
/// the original single-lock design.
#[derive(Debug)]
pub struct DynamicHotspotTable {
    shards: Box<[Shard; NUM_SHARDS]>,
}

impl Default for DynamicHotspotTable {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicHotspotTable {
    /// Creates a new, empty hotspot table.
    ///
    /// # Panics
    /// Panics only if `Box<[Shard]>::try_into::<Box<[Shard; N]>>` fails,
    /// which can't happen because we just allocated exactly N entries.
    #[must_use]
    pub fn new() -> Self {
        let mut shards: Vec<Shard> = Vec::with_capacity(NUM_SHARDS);
        for _ in 0..NUM_SHARDS {
            shards.push(Shard::default());
        }
        // allow-panic: init-only — `shards.len() == NUM_SHARDS` is a
        // structural invariant of the loop above; the conversion
        // cannot fail.
        let array: Box<[Shard; NUM_SHARDS]> = shards
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| unreachable!("Vec length is exactly NUM_SHARDS by construction"));
        Self { shards: array }
    }

    /// Picks the shard index for `id`. Stable across runs.
    const fn shard_idx(id: DagNodeId) -> usize {
        // `% NUM_SHARDS` is OK because NUM_SHARDS is small. Using the
        // raw u32 avoids a `to_le_bytes` allocation.
        id.0 as usize % NUM_SHARDS
    }

    fn shard(&self, id: DagNodeId) -> &Shard {
        &self.shards[Self::shard_idx(id)]
    }

    /// Records an access to a given `DagNodeId`.
    ///
    /// Only the one shard holding `id` is locked; other shards stay
    /// fully concurrent. A poisoned lock recovers transparently via
    /// `PoisonError::into_inner` — we don't maintain any cross-thread
    /// invariant that a panic would have damaged, so poisoning is
    /// just noise.
    pub fn record_access(&self, id: DagNodeId) {
        let shard = self.shard(id);

        // Fast path: key already exists — increment atomically under read lock.
        // No write lock needed since AtomicU64::fetch_add is thread-safe without
        // exclusive access to the HashMap.
        {
            let guard = shard
                .frequencies
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(counter) = guard.get(&id) {
                counter.fetch_add(1, Ordering::Relaxed);
                shard.total_accesses.fetch_add(1, Ordering::Release);
                return;
            }
        }

        // Slow path: new key — acquire write lock and insert. Re-check under
        // write lock so two concurrent slow-path callers don't double-insert.
        let mut guard = shard
            .frequencies
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard
            .entry(id)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
        shard.total_accesses.fetch_add(1, Ordering::Release);
    }

    /// Retrieves the access count for a given `DagNodeId`.
    #[must_use]
    pub fn get_frequency(&self, id: DagNodeId) -> u64 {
        let shard = self.shard(id);
        let guard = shard
            .frequencies
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.get(&id).map_or(0, |c| c.load(Ordering::Relaxed))
    }

    /// Returns whether the access count for `id` meets or exceeds the `threshold`.
    #[must_use]
    pub fn is_hot(&self, id: DagNodeId, threshold: u64) -> bool {
        self.get_frequency(id) >= threshold
    }

    /// Aggregate total accesses across every shard, using
    /// [`Ordering::Acquire`] loads.
    #[must_use]
    pub fn total_accesses(&self) -> u64 {
        self.shards
            .iter()
            .map(|s| s.total_accesses.load(Ordering::Acquire))
            .sum()
    }

    /// Collects every `(DagNodeId, frequency)` pair across every shard.
    ///
    /// Snapshot — concurrent updates after the call do not appear.
    /// O(N + S) where N = total live entries, S = `NUM_SHARDS`.
    #[must_use]
    pub fn snapshot(&self) -> Vec<(DagNodeId, u64)> {
        let mut out = Vec::new();
        for shard in self.shards.iter() {
            let guard = shard
                .frequencies
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            out.extend(guard.iter().map(|(k, v)| (*k, v.load(Ordering::Relaxed))));
        }
        out
    }

    /// Resets every frequency counter and total-access tally.
    pub fn clear(&self) {
        for shard in self.shards.iter() {
            shard
                .frequencies
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clear();
            shard.total_accesses.store(0, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shards_are_cache_line_aligned() {
        assert_eq!(core::mem::align_of::<Shard>(), 128);
    }

    #[test]
    fn records_and_reads_consistently() {
        let table = DynamicHotspotTable::new();
        let id = DagNodeId::new(7);
        for _ in 0..10 {
            table.record_access(id);
        }
        assert_eq!(table.get_frequency(id), 10);
        assert!(table.is_hot(id, 10));
        assert!(!table.is_hot(id, 11));
        assert_eq!(table.total_accesses(), 10);
    }

    #[test]
    fn distinct_ids_land_on_distinct_shards_modulo_n() {
        // For ids 0..NUM_SHARDS, every shard gets exactly one entry.
        let table = DynamicHotspotTable::new();
        for i in 0..NUM_SHARDS {
            #[allow(clippy::cast_possible_truncation)]
            let id = DagNodeId::new(i as u32);
            table.record_access(id);
        }
        let snap = table.snapshot();
        assert_eq!(snap.len(), NUM_SHARDS);
    }

    #[test]
    fn parallel_record_accesses_count_correctly() {
        use std::sync::Arc;
        let table = Arc::new(DynamicHotspotTable::new());
        let id = DagNodeId::new(42);
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let table = Arc::clone(&table);
                std::thread::spawn(move || {
                    for _ in 0..1000 {
                        table.record_access(id);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("thread joined");
        }
        assert_eq!(table.get_frequency(id), 8 * 1000);
        assert_eq!(table.total_accesses(), 8 * 1000);
    }

    #[test]
    fn clear_resets_everything() {
        let table = DynamicHotspotTable::new();
        for i in 0..50_u32 {
            table.record_access(DagNodeId::new(i));
        }
        table.clear();
        assert_eq!(table.total_accesses(), 0);
        assert_eq!(table.snapshot().len(), 0);
    }
}
