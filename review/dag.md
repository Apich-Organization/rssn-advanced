# Module Review: `dag` (Post-Upgrade)

## 1. Performance & Memory

### 1.1 Persistently Bloated Node Representation
Despite improvements in other areas, the core `DagNode` remains **80 bytes** in size.
- The `ChildList` enum, although optimized with inline arrays, still forces the variant size to be large due to the `Many(Vec<DagNodeId>)` variant.
- The `value: Option<f64>` field adds 16 bytes to every node, even though it's only used for constants.
This 80-byte stride is the primary bottleneck for cache-heavy graph traversals.

## 2. Dead Code & Unfinished Updates

### 2.1 Unused `NodeFlags::CANONICAL`
The `CANONICAL` flag is defined in `metadata.rs` and can be set via `with_canonical()`. However, the `HeuristicEngine` in `src/heuristic/engine.rs` uses a local `HashSet<DagNodeId>` to track simplified nodes instead of utilizing this bit. This leads to redundant memory usage during simplification and misses an opportunity to persist the "simplified" state across different engine calls.

### 2.2 Hash-Consing "Wait-and-See"
The `DedupMap` still uses `HashMap<u64, Vec<DagNodeId>>`. While `rapidhash` makes the hashing fast, the bucket-based approach with `Vec` per collision is still less efficient than a flat, open-addressed hash table which would further reduce allocations.

## 3. Extensibility

### 3.1 Hardcoded Operator Set
The `OpKind` enum is fixed. There is no mechanism for users to define "Custom Operators" that carry their own algebraic properties (commutativity, associativity) or evaluation logic without modifying the core `dag` and `symbol` modules. This limits the library's use in domains with specialized mathematical operators (e.g. quantum gates, tensor contractions).

## 4. Suggestions
- Use a "Struct of Arrays" (SoA) or a more compact `DagNode` (e.g. 32 bytes) for the main arena, moving `value` and `Many` children to side-tables.
- Integrate `NodeFlags::CANONICAL` into the `HeuristicEngine` to skip redundant work across multiple `simplify` calls.
- Transition `OpKind` to a more extensible registration system if user-defined operators are a priority.
