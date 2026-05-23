# Module Review: `storage` (Phase 5 Audit)

## 2. Design Integrity

### 2.1 The Blocking Eviction Pass
- **Sharp Question:** In a real-time physics simulation or a high-frequency trading bot, can we really afford to "stop the world" for an eviction pass? Why aren't we using an incremental or concurrent mark-and-sweep that reclaims memory in small chunks?

### 2.2 The Remap Table Memory
- **Sharp Question:** We still return a giant `Vec<DagNodeId>` for remapping. If an eviction pass keeps 1 million nodes, we've just allocated 4MB for the map alone. Why can't we use an offset-based mapping if our arena is contiguous?
