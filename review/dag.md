# Module Review: `dag`

## 1. Performance Issues (High Severity)

### 1.1 Bloated In-Memory Node Representation
The `DagNode` struct is approximately **80 bytes** in size.
- `SymbolKind`: 8 bytes.
- `NodeMetadata`: 24 bytes (includes 8 bytes for `f64` coefficient and 5 bytes padding).
- `ChildList`: 32 bytes (due to `Vec` variant alignment/size).
- `value`: `Option<f64>`: 16 bytes.
This size is excessive for a symbolic computation engine where nodes are created by the millions. The 80-byte stride will cause severe cache misses during graph traversals. Most nodes are simple binary operators and do not need the `value` field or the heap-allocated `Vec` in `ChildList`.

### 1.2 Suboptimal Hash-Consing (`DedupMap`)
- **Incomplete Hashing:** `DedupMap::hash_operator` fails to include the `coefficient` and `flags` in the hash computation. This leads to guaranteed hash collisions for nodes that differ only by coefficient (e.g., `2*x` vs `3*x` if implemented via metadata coefficients), forcing an $O(N)$ linear scan of the collision bucket.
- **Bucket Overhead:** Using `HashMap<u64, Vec<DagNodeId>>` adds an unnecessary layer of indirection. A specialized hash table with open addressing would be significantly faster.

### 1.3 `SymbolRegistry` Allocation Overhead
- Interning a new string performs two allocations: one for the `names` vector and one for the `lookup` map key.
- It uses standard `std::collections::HashMap`, which involves locking if wrapped for thread-safety, or prevents parallel interning if not.

## 2. Correctness Issues

### 2.1 Packed Node Arity Overflow
In `src/dag/packed.rs`, nodes with arity > 255 are marked with `arity = 255`. The decoding logic in `BorrowedArenaView::children` assumes such nodes extend to the end of the `children_pool`:
```rust
let len = if arity == 255 {
    self.children_pool.as_slice().len().saturating_sub(start)
} else {
    arity
};
```
This is **broken** if more than one node has 255+ children or if a node with 255+ children is not the last one to use the pool.

## 3. Deviations from Plan

### 3.1 "Compact Pointers" vs 80-byte Nodes
The `plan.md` mentions that "DAG node metadata is huge" and "AST projection uses relative pointers to compress storage". However, the current in-memory DAG storage is extremely bloated. While `PackedDagNode` (32 bytes) exists, it is only used for serialization/zerocopy, not for primary computation.

## 4. Engineering Standards

### 4.1 Heavy Standard API Usage
- Reliance on `std::collections::HashMap` for performance-critical deduplication and interning.
- Use of `Vec` inside an enum (`ChildList`) makes every instance of the enum as large as the `Vec` variant.

## 5. Suggestions
- Implement a `SmallVec`-like optimization for `ChildList` or use a dedicated children pool even in the rich representation.
- Use a "Struct of Arrays" (SoA) approach in `DagArena` to improve cache locality for common operations (e.g., hashing, type checking).
- Fix the `arity == 255` logic by storing the actual length in the pool for large nodes.
- Include all identifying metadata in the node hash.
