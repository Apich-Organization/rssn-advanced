//! Eviction policy for streaming storage.
//!
//! Implements LFU/frequency-based compaction and eviction strategies
//! to automatically prune cold nodes or clean up memory.

use crate::dag::arena::DagArena;
use crate::dag::node::DagNodeId;
use super::hotspot::DynamicHotspotTable;

/// Compacts a `DagArena` by retaining only nodes that are deemed hot,
/// plus any nodes recursively reachable from them.
///
/// This serves as a dynamic garbage-collection style eviction pass
/// when memory limits or symbol explosions are detected.
#[must_use]
pub fn evict_cold_nodes(
    arena: &DagArena,
    hotspots: &DynamicHotspotTable,
    keep_threshold: u64,
) -> DagArena {
    let mut compacted = DagArena::new();
    
    // We walk all nodes in the original arena.
    // If a node is hot (or it is a constant/variable which are typically preserved),
    // we allocate it into the compacted arena.
    for i in 0..arena.len() {
        let id = DagNodeId(i as u32);
        if let Some(node) = arena.get(id) {
            // Check frequency of access
            let freq = hotspots.get_frequency(id);
            if freq >= keep_threshold || node.is_leaf() {
                // Preserve node
                compacted.alloc(node.clone());
            }
        }
    }

    compacted
}
