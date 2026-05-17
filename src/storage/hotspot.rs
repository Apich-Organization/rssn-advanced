//! Dynamic hotspot table for frequency-based caching.
//!
//! Tracks how often each `DagNodeId` is referenced during computation.
//! High-frequency nodes are promoted to pinned memory; cold nodes are
//! candidates for eviction.

use std::collections::HashMap;
use std::sync::RwLock;
use crate::dag::node::DagNodeId;

/// Dynamic frequency table tracking node access patterns.
#[derive(Debug, Default)]
pub struct DynamicHotspotTable {
    // Thread-safe map tracking raw access frequency counts.
    frequencies: RwLock<HashMap<DagNodeId, u64>>,
}

impl DynamicHotspotTable {
    /// Creates a new, empty hotspot table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            frequencies: RwLock::new(HashMap::new()),
        }
    }

    /// Records an access to a given `DagNodeId`, incrementing its frequency count.
    ///
    /// # Panics
    /// Panics if the internal lock is poisoned.
    pub fn record_access(&self, id: DagNodeId) {
        let mut guard = self.frequencies.write().expect("Hotspot lock poisoned");
        let count = guard.entry(id).or_insert(0);
        *count += 1;
    }

    /// Retrieves the access count for a given `DagNodeId`.
    ///
    /// # Panics
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn get_frequency(&self, id: DagNodeId) -> u64 {
        let guard = self.frequencies.read().expect("Hotspot lock poisoned");
        guard.get(&id).copied().unwrap_or(0)
    }

    /// Returns whether the access count for `id` meets or exceeds the `threshold`.
    ///
    /// # Panics
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn is_hot(&self, id: DagNodeId, threshold: u64) -> bool {
        self.get_frequency(id) >= threshold
    }

    /// Resets all frequency counters.
    ///
    /// # Panics
    /// Panics if the internal lock is poisoned.
    pub fn clear(&self) {
        let mut guard = self.frequencies.write().expect("Hotspot lock poisoned");
        guard.clear();
    }
}
