# RSSN-Advanced Review: `src/heuristic`

## **1. Alignment with `plan.md`**

### **Programmable Heuristic Toolbox**
- **Status**: **PASS (Partial)**
- **Observation**: `HeuristicConfig` (knobs) and `SearchStrategy` correctly implement the "programmable" aspect of the engine.
- **Issue**: The plan mentions "Pattern Matching (模式匹配)". However, the current engine does not implement any actual pattern-matching logic (e.g., matching `x + 0` or `x * 1`). It only performs a recursive traversal.

### **Avoidance of Symbol Explosion**
- **Status**: **CRITICAL FAIL**
- **Issue**: `HeuristicEngine::explore_and_rewrite` and `approximate_simplify_rec` bypass the structural deduplication (`DedupMap`) of the `DagBuilder`.
- **Evidence**: They take `&mut DagArena` and call `arena.alloc(new_node)` directly.
- **Impact**: Without deduplication, any "simplification" pass will create millions of duplicate nodes in the arena, leading to a massive memory leak and triggering the "symbol explosion" the project aims to avoid.
- **Recommendation**: The heuristic engine MUST use a `DagBuilder` or have access to the `DedupMap` to ensure all rewritten nodes are hash-consed.

---

## **2. Performance Issues**

### **Recursive Traversal**
- **Issue**: `explore_and_rewrite` and `approximate_simplify_rec` are recursive.
- **Risk**: Stack overflow on large expressions.
- **Recommendation**: Use an iterative work-list approach.

### **Crude Approximate Simplification**
- **Issue**: `approximate_simplify_rec` folds deep subtrees to `1.0` if `aggressiveness > 0.5`.
- **Observation**: This is mathematically unsound and will produce incorrect results for almost any symbolic expression.
- **Recommendation**: Implement more meaningful pruning (e.g., dropping low-coefficient terms in a polynomial) rather than arbitrary constant folding.

---

## **3. Zero-Copy & `bincode-next`**

### **Arena Inflation**
- **Observation**: Due to the lack of deduplication during simplification, serialized arenas will be much larger than necessary, increasing the IO burden on the zero-copy storage layer.

---

## **4. Error Handling**

### **Macro Non-Compliance**
- **Issue**: Does not use the requested cold-path error macro.
- **Recommendation**: Budget exhaustion or timeout events should be handled via the cold-path macro.
