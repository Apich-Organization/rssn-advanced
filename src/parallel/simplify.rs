//! Staged global simplification.
//!
//! Provides a configurable simplify config and cache-aligned thread-local counters
//! utilizing Acquire/Release memory ordering to completely eliminate false-sharing.

use std::sync::atomic::{AtomicU64, Ordering};

/// Staged simplification configuration parameters.
#[derive(Debug, Clone, Copy)]
pub struct SimplifyConfig {
    /// Number of intermediate local simplification rounds to perform.
    pub intermediate_rounds: usize,
    /// Whether to trigger global deduplication and constant folding.
    pub enable_global_dedup: bool,
}

impl Default for SimplifyConfig {
    fn default() -> Self {
        Self {
            intermediate_rounds: 3,
            enable_global_dedup: true,
        }
    }
}

impl SimplifyConfig {
    /// Creates a new configuration builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            intermediate_rounds: 0,
            enable_global_dedup: true,
        }
    }

    /// Sets the number of intermediate local simplification rounds.
    #[must_use]
    pub const fn intermediate_rounds(mut self, rounds: usize) -> Self {
        self.intermediate_rounds = rounds;
        self
    }
}

/// Cache-line padded thread-local state to physically isolate counters.
///
/// Under high concurrency, threads updating adjacent memory addresses trigger
/// false-sharing (cache line invalidations on MESI). This struct aligns memory
/// boundary to 128 bytes, isolating thread states.
#[derive(Debug)]
#[repr(align(128))]
pub struct ThreadLocalState {
    /// Number of evaluations or simplification steps performed by this thread.
    pub steps_count: AtomicU64,
}

impl Default for ThreadLocalState {
    fn default() -> Self {
        Self {
            steps_count: AtomicU64::new(0),
        }
    }
}

impl ThreadLocalState {
    /// Creates a new instance.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            steps_count: AtomicU64::new(0),
        }
    }

    /// Increments the local operation counter with strict memory ordering.
    pub fn increment(&self) {
        // plan.md §4.2: strict Acquire/Release memory ordering
        self.steps_count.fetch_add(1, Ordering::Release);
    }

    /// Retrieves the local operation count with strict memory ordering.
    #[must_use]
    pub fn get_count(&self) -> u64 {
        self.steps_count.load(Ordering::Acquire)
    }
}
