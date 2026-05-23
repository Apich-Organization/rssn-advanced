# Module Review: `heuristic` (Phase 3 Audit)

## 1. Performance

### 1.1 Cache Fragmentation
The `canonical_cache` is a `HashSet<DagNodeId>` owned by the `HeuristicEngine`.
- **Sharp Question:** In a large project where nodes are shared across multiple builders and engines, why is the "knowledge" of a node's simplified state trapped inside a local `HashSet`? Why aren't we persisting this bit in the DAG itself using the `CANONICAL` flag we already defined?

## 2. Extensibility

### 2.1 The Rule Registry Overhead
The `RuleRegistry` is a `Vec` of `Box<dyn Fn(...)>`. 
- **Sharp Question:** If a user registers 500 rules, we will perform 500 virtual function calls for **every single node** in the graph. Is this a "performant" engine or a "flexible" one that sacrifices all speed for ease of use? Why not use a more structured approach like E-graph matching or a state-machine based matcher?

## 3. Correctness

### 3.1 Unsound Constant Folding
`patterns::try_apply` performs basic identity folding (e.g. `x * 1 -> x`).
- **Sharp Question:** What happens if `x` is `NaN` or `Infinity`? Does our "algebraic simplification" preserve the IEEE-754 semantics that our JIT and Parallel engines rely on, or are we producing "simplified" expressions that give different numerical results?
