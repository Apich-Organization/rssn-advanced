# Module Review: `storage` (Phase 3 Audit)

## 1. Performance

### 1.1 The "Stop-the-World" Eviction
`evict_cold_nodes` performs a full O(N) mark-and-sweep.
- **Sharp Question:** In a streaming symbolic engine, why is our only memory reclamation strategy a global, blocking compaction pass? If we have 10GB of nodes, do we really want to wait 2 seconds for a "mark" phase before we can keep calculating?

## 2. Design Consistency

### 2.1 The Remap Table Allocation
Eviction returns a `HashMap<DagNodeId, DagNodeId>`.
- **Sharp Question:** We just cleared memory, and then we immediately allocate a giant `HashMap` that is nearly as large as the arena itself just to tell the user where their nodes went. Is there no more efficient way to communicate a range-based or offset-based remap?

## 3. Extensibility

### 3.1 Hardcoded "Hotness"
The eviction policy is "access frequency >= threshold".
- **Sharp Question:** What if a user wants to protect nodes based on "Depth", "Recency (LRU)", or "Algebraic Complexity"? Why is our "DynamicHotspotTable" hardcoded to one specific policy?
