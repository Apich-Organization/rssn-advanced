# Module Review: `heuristic` (Post-Upgrade)

## 1. Performance & Memory

### 1.1 Iterative & Balanced Rewriting
The `HeuristicEngine` is now fully iterative, and `approximate_simplify` uses `balanced_add` to prevent the creation of deep, left-associative trees that could cause stack overflows in other modules. This is a critical stability improvement.

### 1.2 Allocation-Lite Rewrite Loop
The `rewrite_iterative` loop now borrows children as slices from the value stack, eliminating the per-node `Vec` allocation bottleneck.

## 2. Dead Code & Functionality

### 2.1 Unused `NodeFlags::CANONICAL`
As noted in the `dag` review, the `CANONICAL` bit is not used by the `HeuristicEngine`. Instead, it uses a local `HashSet` which is lost after the `simplify` call finishes.

## 3. Extensibility

### 3.1 Closed Rule Engine
The pattern matching logic is entirely hardcoded in `patterns.rs`. Users cannot add their own algebraic identities (e.g. `sin(x)^2 + cos(x)^2 -> 1`) without modifying the library source. This is the most significant "extensibility" gap in the current engine.

## 4. Suggestions
- Implement a `RuleRegistry` that allows users to register custom pattern-match and replacement closures.
- Persist the `CANONICAL` bit in the DAG so that nodes simplified by one engine call are automatically skipped by subsequent calls (even from different engines).
