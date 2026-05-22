# Module Review: `storage` (Post-Upgrade)

## 1. Performance & Memory

### 1.1 Zero-Copy Mmap Restoration
The `DiskCache` now uses `MmapBuffer` and `decode_zerocopy_raw` correctly, avoiding the redundant whole-file heap allocation that plagued the previous version.

### 1.2 Lock-Free Hotspot Tracking
The `DynamicHotspotTable` now uses a "Read Lock + Atomic Increment" fast-path, which drastically reduces contention in parallel workloads. The 128-byte alignment of shards effectively prevents false sharing.

## 2. Dead Code & Functionality

### 2.1 Manual "Mark-and-Sweep" vs `NodeFlags`
The `evict_cold_nodes` function uses a local `Vec<bool>` for marking. While correct, this mirrors the functionality that `NodeFlags::CANONICAL` or a similar bit could provide if properly integrated into the arena.

## 3. Extensibility

### 3.1 Closed Eviction Policy
The `evict_cold_nodes` function implements a specific frequency-based policy. There is no way for a user to provide a custom eviction strategy (e.g. LRU, LFU, or priority-based) without modifying the `storage` module.

## 4. Suggestions
- Generalize the eviction logic to accept a `Policy` trait, allowing users to define their own hotness criteria.
- Consider using the `MmapBuffer` directly as a backend for the `DagArena` to enable "out-of-core" computation without explicit `restore` calls.
