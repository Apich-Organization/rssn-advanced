//! Example 04: Disk Caching, Spillover, and Cold Node Eviction
//!
//! This example shows how to manage memory footprint under heavy symbolic load
//! by recording hot nodes, evicting cold nodes from the arena, spilling arenas
//! to disk, and restoring them seamlessly at a later time.
//!
//! Run with: `cargo run --example 04_disk_cache_and_spillover`

use rssn_advanced::dag::builder::DagBuilder;
use rssn_advanced::dag::node::DagNodeId;
use rssn_advanced::storage::{DiskCache, DynamicHotspotTable, evict_cold_nodes};
use std::fs::remove_dir_all;
use std::path::PathBuf;

fn main() {
    println!("=== RSSN-Advanced Example 04: Disk Caching & Spillover ===\n");

    // 1. Initialize our memory hotspot tracking table
    println!("Initializing hotspot table and recording node accesses...");
    let hotspots = DynamicHotspotTable::new();

    // We will simulate access patterns by recording accesses for specific node IDs
    let id_x = DagNodeId::new(0); // variable 'x'
    let id_y = DagNodeId::new(1); // variable 'y'
    let id_c = DagNodeId::new(2); // constant '5.0'

    hotspots.record_access(id_x);
    hotspots.record_access(id_x); // Access frequency = 2 (Very Hot!)
    hotspots.record_access(id_y); // Access frequency = 1 (Mildly Hot)
    // Node id_c is not accessed (Access frequency = 0 - Cold!)

    println!(
        "  Node 'x' frequency             : {}",
        hotspots.get_frequency(id_x)
    );
    println!(
        "  Node 'y' frequency             : {}",
        hotspots.get_frequency(id_y)
    );
    println!(
        "  Node 'constant' frequency      : {}\n",
        hotspots.get_frequency(id_c)
    );

    // 2. Build the arena and demonstrate eviction of cold nodes
    println!("Constructing expressions in memory...");
    let mut builder = DagBuilder::new();
    let x = builder.variable("x");
    let y = builder.variable("y");
    let _c = builder.constant(5.0);
    let _expr = builder.add(x, y);

    let original_size = builder.arena().len();
    println!("  Original DagArena node count   : {}", original_size);

    // Evict cold nodes with access threshold = 2
    // Only variables, leaf defaults, and nodes with freq >= 2 are preserved
    println!("Evicting cold nodes (frequency < 2) to reclaim memory...");
    let compacted_arena = evict_cold_nodes(builder.arena(), &hotspots, 2);
    println!(
        "  Compacted DagArena node count  : {}\n",
        compacted_arena.arena.len()
    );

    // 3. Disk Cache spilling and restoring
    let cache_dir = PathBuf::from("./example_spillover_cache");
    println!("Initializing DiskCache at location: {:?}", cache_dir);
    let disk_cache = DiskCache::new(&cache_dir).expect("Failed to initialize disk cache");

    // Spill the current compacted arena to disk cache
    let key = "partition_block_42";
    println!("Spilling compacted arena to disk with key: '{}'...", key);
    disk_cache
        .spill(key, &compacted_arena.arena)
        .expect("Failed to spill arena to disk");
    println!("  Arena successfully serialized and spilled to disk.");

    // Restore the arena from disk cache
    println!("Restoring arena from disk cache...");
    let restored_arena = disk_cache
        .restore(key)
        .expect("Failed to restore arena from disk");
    println!("  Arena successfully restored and deserialized!");
    println!(
        "  Restored DagArena node count   : {}\n",
        restored_arena.len()
    );

    // Verify restored node properties (Variable 'x' was preserved as it was Hot!)
    let restored_node = restored_arena.get(x).expect("Failed to get restored node");
    println!("Restored Node Verification:");
    println!("  Restored Hot node 'x' ID       : {:?}", x);
    println!(
        "  Restored Hot node value/kind   : {:?}",
        restored_node.kind
    );

    // Cleanup cache directory
    println!("\nCleaning up local cache directory...");
    disk_cache.delete(key).unwrap();
    let _ = remove_dir_all(&cache_dir);
    println!("Cleanup completed.");

    println!("===========================================================");
}
