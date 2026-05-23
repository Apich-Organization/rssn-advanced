# Module Review: `dag` (Phase 5 Audit)

## 2. Extensibility

### 2.1 Pluggable Operators
`OpKind` remains a closed enum, but `SymbolKind::Function` now supports user-defined native callbacks.
- **Sharp Question:** We still have a hardcoded `OpKind` enum. If `FnId` `0..=6` are reserved for these operators, why is `OpKind` still a distinct type rather than a set of "Intrinsic" constants in a unified `FnId` space? Are we keeping this duality only for the convenience of `match` statements, at the cost of architectural purity?

## 3. Correctness

### 3.1 `CANONICAL` Flag Inconsistency
The `HeuristicEngine` uses the `CANONICAL` bit, and `PackedArenaImage` now includes a `rule_fingerprint` to detect when these bits are stale.
- **Sharp Question:** Clearing `CANONICAL` bits on fingerprint mismatch is a "safety" feature, but it's an all-or-nothing approach. In a world of incremental updates, is there a way to clear only the bits affected by a *specific* rule change, or is global invalidation the only way to avoid mathematical "ghosts"?
