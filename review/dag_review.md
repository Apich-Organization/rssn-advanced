# RSSN-Advanced Review: `src/dag`

## **1. Alignment with `plan.md`**

### **Global DAG Storage**
- **Status**: **PASS**
- **Observation**: `DagArena` and `DagBuilder` correctly implement the hash-consed, structurally-shared storage model.
- **Issue**: The plan mentions "Metadata 巨大" (Metadata is huge). However, `NodeMetadata` is only ~24 bytes. The actual `DagNode` is much larger due to the `SymbolKind` enum and the `ChildList` enum.

### **Avoidance of Synchronization**
- **Status**: **PASS (Partial)**
- **Observation**: The current implementation is single-threaded (`DagBuilder` owns the arena). This avoids sync overhead but the plan mentions "avoiding MESI bus sync" specifically for "global simplification". The current design doesn't yet show how it handles the "global-local" hybrid in a multi-threaded context.

---

## **2. Performance & Memory Issues**

### **Large Node Size**
- **Issue**: `DagNode` is bloated.
    - `SymbolKind`: ~16 bytes.
    - `NodeMetadata`: ~24 bytes.
    - `ChildList`: ~32 bytes (if `Many`).
    - `Option<f64>`: ~16 bytes.
- **Total**: ~88 bytes per node.
- **Impact**: For millions of nodes, this exceeds the "KISS principle" and "扁平" (flat) requirements. The 88-byte size will cause significant cache pressure.
- **Recommendation**: Use a more compact representation. For example, `f64` can be part of a union or stored in a separate arena for constant nodes. `ChildList` could use a more compact encoding.

### **Linear Symbol Interning**
- **Issue**: `SymbolRegistry::intern` performs a linear scan (`for (i, existing) in self.names.iter().enumerate()`).
- **Risk**: As the number of variables increases, building the DAG becomes $O(N^2)$.
- **Recommendation**: Use a `HashMap<String, SymbolId>` to make interning $O(1)$.

### **Heap Allocation in `ChildList`**
- **Issue**: `ChildList::Many(Vec<DagNodeId>)` spills to the heap for >4 children.
- **Recommendation**: Consider using a "large child list" arena to keep all children contiguous and avoid individual `Vec` allocations for every high-arity node.

---

## **3. Zero-Copy & `bincode-next`**

### **Non-Zero-Copy Arena**
- **Issue**: `DagArena` stores `Vec<DagNode>`. Decoding this requires allocating a new `Vec` and copying all nodes.
- **Reminders**: The user explicitly requested zero-copy.
- **Recommendation**: The arena should be decodable as a reference to a slice `&[DagNode]` if the layout is stable, or use `Borrowed` types from `bincode-next`.

---

## **4. Error Handling**

### **Macro Non-Compliance**
- **Issue**: `get_or_insert` and `builder` methods do not use the requested cold-path error macro.
- **Observation**: While they don't panic as much as the AST module, they don't provide the requested "cold path" optimized error returns.

---

## **5. Extensibility**

### **Closed Symbol System**
- **Issue**: Similar to the AST review, `SymbolKind` is a closed enum.
- **Impact**: Hard to add new types of symbolic objects (e.g., matrices) without modifying the core DAG structure.
