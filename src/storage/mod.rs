//! Streaming storage and dynamic caching.
//!
//! Provides memory management for large-scale symbolic computation:
//!
//! - `cache` — Disk-backed spillover for large-scale arenas.
//! - `hotspot` — Runtime frequency tracking for intermediate results.
//! - `eviction` — LRU/frequency-based eviction policy for auto-cleanup.

pub mod cache;
pub mod eviction;
pub mod hotspot;

pub use cache::DiskCache;
pub use eviction::evict_cold_nodes;
pub use hotspot::DynamicHotspotTable;
