# Module Review: `storage` (Phase 4 Audit)

## 1. Design Integrity

### 1.1 The Blocking Eviction Pass
`evict_cold_nodes` remains a global, blocking mark-and-sweep.
- **Sharp Question:** In a real-time physics simulation or a high-frequency trading bot, can we really afford to "stop the world" for an eviction pass? Why aren't we using an incremental or concurrent mark-and-sweep that reclaims memory in small chunks?

### 1.2 The Remap Table Memory
- **Sharp Question:** We still return a giant `HashMap` for remapping. If an eviction pass keeps 1 million nodes, we've just allocated 40MB for the map alone. Why can't we use an offset-based mapping if our arena is contiguous?
