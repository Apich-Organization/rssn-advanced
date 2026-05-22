# Module Review: `parallel`

## 1. Performance Issues (High Severity)

### 1.1 Unnecessary Arena Cloning
The `parallel_evaluate` function takes a reference to `DagArena` and clones it before wrapping it in an `Arc`:
```rust
pub fn parallel_evaluate(arena: &DagArena, chunks: Vec<Vec<DagNodeId>>, variables: &[f64]) -> f64 {
    let arc = Arc::new(arena.clone());
    parallel_evaluate_shared(&arc, chunks, variables)
}
```
Since `DagArena` holds a `Vec` of 80-byte `DagNode`s, this clone is extremely expensive (80MB for 1 million nodes). For a parallel solver that might be called frequently, this overhead is unacceptable.

### 1.2 Allocation Bottleneck in Evaluator
Similar to the AST and Heuristic modules, the `evaluate_node` function performs a heap allocation for every operator node encountered:
```rust
let split_at = values.len().saturating_sub(arity);
let child_vals: Vec<f64> = values.drain(split_at..).collect();
apply_op(op, &child_vals)
```
In a large expression, this results in millions of small allocations, likely making the "parallel" evaluator slower than a properly optimized single-threaded evaluator that uses a single value stack.

### 1.3 `Arc` Overhead for Small Tasks
The solver creates a new `Arc` for the variable bindings and clones the arena `Arc` once per chunk. While better than cloning the whole arena, the cost of these atomic operations and the overhead of dispatching small chunks to the `dtact` runtime can outweigh the benefits of parallelism for small to medium-sized expressions.

## 2. Correctness & Numerical Issues

### 2.1 Silent Masking of Division by Zero
The `apply_op` function for division returns `0.0` when the divisor is below `EPSILON`:
```rust
if rhs.abs() < f64::EPSILON { 0.0 } else { lhs / rhs }
```
This is mathematically incorrect and inconsistent with the JIT module (which traps) and standard floating-point behavior (which returns `Infinity` or `NaN`). It also uses the dangerous `EPSILON` threshold, leading to precision loss.

## 3. Engineering Standards

### 3.1 Redundant ThreadLocalState
The `ThreadLocalState` with 128-byte alignment is a good idea for avoiding false sharing, but it is currently only used to track a "steps count," which is not a performance-critical path. This adds complexity without addressing the real false-sharing risks in the deduplication or symbol registry.

## 4. Suggestions
- Change the API to require `Arc<DagArena>` or use a more efficient way to share the arena without copying.
- Optimize `evaluate_node` to use a single pre-allocated `Vec<f64>` as a value stack, passing slices to `apply_op`.
- Use exact zero checks (`== 0.0`) and consistent error handling (e.g., returning `NaN`) for division by zero.
- Re-evaluate the sharding strategy for the Hotspot table (see `storage` review) as it is a more likely source of false sharing than the evaluation steps count.
