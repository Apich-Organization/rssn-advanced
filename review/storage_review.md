# RSSN-Advanced Review: `src/storage`

## **1. Alignment with `plan.md` & Instructions**

### **Zero-Copy Serialization (`bincode-next`)**
- **Status**: **CRITICAL FAIL**
- **Issue**: The user explicitly instructed: "it shall use the zero-copy feature of bincode-next".
- **Evidence**: `src/storage/cache.rs` uses `bincode_next::decode_from_slice(&bytes, config)`. However, since `DagArena` (and its component `DagNode`) only implements owned `Decode` and contains `Vec<DagNode>`, `bincode-next` is forced to allocate new memory and copy all data from the buffer into a new `DagArena`.
- **Impact**: For large DAGs spilled to disk (GBs), this results in massive redundant memory allocations and CPU time spent on copying, defeating the purpose of high-performance streaming storage.

### **Streaming Storage & Spillover**
- **Status**: **PASS (Partial)**
- **Observation**: `DiskCache` provides basic spillover.
- **Issue**: It reads the entire file into memory (`file.read_to_end(&mut bytes)`) before decoding. This is not "streaming" in the true sense and will OOM for very large datasets.
- **Recommendation**: Use memory-mapped files (`mmap`) or stream decoding to truly support "超大规模计算" (ultra-large scale computation).

---

## **2. Performance Issues**

### **Global Lock Contention**
- **Issue**: `DynamicHotspotTable` uses a single `RwLock<HashMap<DagNodeId, u64>>`.
- **Observation**: Every time a node is accessed during computation (which happens millions of times per second), `record_access` is called, requiring a write lock.
- **Impact**: This becomes a massive bottleneck in parallel execution, triggering the exact "MESI bus sync" and lock contention issues §4.1 of the plan seeks to avoid.
- **Recommendation**: Use thread-local frequency counters or a lock-free sharded hash map (e.g., `dashmap`).

### **Broken Eviction Logic**
- **Issue**: `evict_cold_nodes` only checks the frequency of the immediate node.
- **Critical Risk**: If a "hot" node depends on a "cold" child, the child will be evicted, leaving the hot node with a dangling `DagNodeId` reference.
- **Recommendation**: Implement a proper mark-and-sweep or reference-counting based eviction that preserves the transitive closure of hot nodes.

---

## **3. Zero-Copy & `bincode-next`**

### **Lack of `BorrowDecode`**
- **Observation**: As noted above, the system misses the zero-copy requirement.
- **Recommendation**: Refactor `DagArena` to support borrowing from the underlying byte buffer, likely by using a slice `&[DagNode]` or a specialized zero-copy arena type.

---

## **4. Error Handling**

### **Macro Non-Compliance**
- **Issue**: Does not use the requested cold-path error macro.
- **Recommendation**: IO errors and serialization failures should be handled via the macro.
