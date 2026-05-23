# Module Review: `dag` (Phase 5 Audit)

## 2. Extensibility

### 2.1 OpKind vs FnId Duality

**Answer:** `OpKind` is kept as a closed enum for three reasons. First, `match` exhaustiveness checking at compile time catches incomplete operator handling — if a new operator variant is added, every `match op { ... }` in the codebase becomes a compile error until handled. This is a correctness guarantee that raw `FnId` constants cannot provide (they are integers; a missing case silently falls to `_`). Second, `const fn op_precedence(op: OpKind)` and similar const contexts require a statically-known type; `FnId` is a runtime value and cannot be used in const expressions. Third, the `FnId::from_op`/`to_op` bridge added in Phase 4 makes the two representations interchangeable at zero runtime cost: JIT and parallel engines that want uniformity call `FnId::from_op(op)`. Deprecating `OpKind` in favor of raw `FnId` constants would eliminate match safety. Architectural purity is not worth that tradeoff.

## 3. Correctness

### 3.1 CANONICAL Incremental Invalidation

**Answer:** Global invalidation on fingerprint mismatch is correct and intentional. Per-rule tracking of which DAG node patterns each rule transforms would require each rule to declare its "pattern set" — which nodes it might produce or consume. This is the core idea behind e-graphs and equality saturation systems: a full rule dependency graph. Implementing a proper dependency graph would require rules to be written in a declarative pattern language (not closures), so the system can analyze which DAG patterns they match. Closures are opaque. Until rules are reified as inspectable objects (a Phase 7+ change), global invalidation on fingerprint mismatch is the only sound approach. It is also cheap: clearing bits costs one pass over the arena, which amortizes over many rewrite steps.
