# Module Review: `storage` (Phase 5 Audit)

## 2. Design Integrity

### 2.1 The Blocking Eviction Pass

**Answer:** `evict_cold_nodes_budgeted(budget)` was added in Phase 3 and limits the sweep to at most `budget` protected nodes per call. It is explicitly incremental: callers invoke it in a background thread or on a periodic tick, processing a bounded chunk each time. Fully concurrent mark-and-sweep (running while the DAG builder is active) would require a read-write lock on the arena across threads, which conflicts with the DAG builder's exclusive ownership model — the builder holds `&mut DagArena` during construction, and concurrent mutation would require unsafe interior mutability or lock-based access on every node allocation. The budgeted approach is the correct tradeoff: it gives real-time callers a knob to tune latency vs throughput without imposing lock overhead on the hot allocation path.

### 2.2 The Remap Table Memory

**Answer:** Fixed in Phase 5 — `EvictionResult.remap` is now a `CompactRemap` using a hierarchical rank structure. For N old slots: `blocks: Vec<u64>` is a bitset (1 bit per slot, N/8 bytes), `block_prefix: Vec<u32>` stores prefix popcounts at 64-slot boundaries (N/16 bytes). For 1 M nodes: 16 KB (bitset) + 64 KB (prefix counts) = 80 KB, a 50× reduction vs the previous `Vec<DagNodeId>` (4 MB). `CompactRemap::translate` is O(1): two indexed array accesses and one `count_ones()` instruction. The existing `EvictionResult::translate` API is unchanged; callers see no difference.
