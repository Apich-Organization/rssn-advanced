# Module Review: `storage`

## 1. Performance Issues (High Severity)

### 1.1 Hidden Copying in "Zero-Copy" Cache
The `DiskCache::restore` and `load_borrowed` methods claim to avoid whole-file allocations by using `MmapBuffer`. However, they immediately pass the resulting bytes to `AlignedBytes::from_slice(bytes)`:
```rust
let aligned = crate::zerocopy::AlignedBytes::from_slice(bytes);
let view = BorrowedArenaView::decode(&aligned)...
```
`AlignedBytes::from_slice` **performs a full copy** of the buffer into a new heap allocation to ensure 8-byte alignment. This completely invalidates the performance benefits of using memory-mapped files and doubles the memory pressure during restoration.

### 1.2 Excessive Write Locking in Hotspot Tracking
`DynamicHotspotTable::record_access` acquires a **write lock** on a shard for every single node access:
```rust
let mut guard = shard.frequencies.write()...;
let count = guard.entry(id).or_insert(0);
*count += 1;
```
Even with sharding (32 shards), frequent write-locking of the frequency map will cause significant contention in highly parallel workloads. For a hotspot tracker, a lock-free counter or an atomic-based approach would be far more efficient.

### 1.3 Memory Pressure during Spilling
The `spill` process creates a `PackedArenaImage` and then encodes it into `AlignedBytes` before writing to disk. This sequence requires multiple large temporary allocations proportional to the size of the arena, potentially leading to OOM (Out Of Memory) errors when spilling very large arenas that were already pushing memory limits.

## 2. Engineering Standards

### 2.1 Eager Materialization
`DiskCache::restore` eagerly converts the packed 32-byte representation back into the bloated 80-byte `DagNode` representation. This loses the space-saving benefits of the packed format immediately upon loading, even if the user only needs to perform a few operations on the restored arena.

## 3. Suggestions
- Implement a way to use memory-mapped bytes directly if they are already aligned, or ensure `MmapBuffer` provides aligned access without a full copy.
- Use a more concurrent frequency tracking structure, such as one based on `DashMap` or a simple fixed-size hash table with atomic counters.
- Stream the arena to disk during `spill` instead of buffering the entire packed image in memory.
- Provide a way to work with the `BorrowedArenaView` for longer periods without materializing the full `DagArena`.
