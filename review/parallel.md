# Module Review: `parallel` (Post-Upgrade)

## 1. Performance & Memory

### 1.1 Zero-Clone Arena Sharing
The parallel evaluator now correctly uses `Arc<DagArena>` to share the graph across worker fibers, eliminating the expensive 80MB clones identified in the previous review.

### 1.2 Iterative & Allocation-Lite Evaluation
`evaluate_node` is now iterative and uses a single pre-allocated value stack, which is significantly faster and more stable than the previous recursive/allocation-heavy implementation.

## 2. Dead Code & Functionality

### 2.1 Inconsistent `SymbolKind::Function` Handling
While the `JitCompiler` can execute custom functions, the `parallel_evaluate` path still returns `0.0` for any function node. This makes the two evaluation paths produce different results for the same expression, which is a major correctness issue.

## 3. Extensibility

### 3.1 Hardcoded Operator Logic
The `apply_op` function is a monolithic match statement. Adding a new operator requires modifying this core evaluation loop.

## 4. Suggestions
- Synchronize the `parallel_evaluate` path with the `JitCompiler` by allowing the registration of function pointers for the parallel evaluator.
- Use a trait-based approach for operators to allow users to define their own parallelizable symbolic operators.
