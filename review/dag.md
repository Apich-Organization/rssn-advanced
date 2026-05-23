# Module Review: `dag` (Phase 4 Audit)

## 1. Extensibility

### 1.1 Pluggable Operators
While `OpKind` remains a closed enum for the most common operations, the JIT and Parallel engines now support `SymbolKind::Function` with user-defined native callbacks and arity.
- **Sharp Question:** If we have a high-performance "Function" path for custom logic, why do we still need a hardcoded `OpKind` at all? Could the built-in operators be treated as pre-registered "intrinsic" functions to simplify the core engine and make it truly "pluggable"?

## 2. Correctness

### 2.1 `CANONICAL` Flag Inconsistency
The `HeuristicEngine` now uses the `CANONICAL` bit in `NodeMetadata` to skip redundant work, but it still maintains a transient `HashSet` for safety during a single call.
- **Sharp Question:** If a node is marked `CANONICAL` and then its children are evicted or changed in a separate arena, does the bit become a "lie"? How do we ensure the `CANONICAL` bit remains globally valid in a system with streaming storage and multiple builders?
