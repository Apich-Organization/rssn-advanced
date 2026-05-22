# Module Review: `runtime` (Post-Upgrade)

## 1. Performance & Memory

### 1.1 Lock-Free Fan-Out
The `parallel_for_each` utility is now lock-free, using a disjoint-index write pattern that eliminates the previous global `Mutex` bottleneck. This allows the `dtact` fiber pool to scale linearly with core counts.

### 1.2 Allocation-Lite Task Spawn
`TaskEnvelope` reduces the task spawn cost to a single heap allocation, removing the redundant "double-boxing" identified in the previous review.

## 2. Dead Code & Functionality

### 2.1 Unpropagated Task Failure
While `parallel_for_each` now catches panics (returning `None`), it still simply flattens the result vector. This means the caller is never notified that a sub-task failed, which could lead to silent data corruption in large parallel computations.

## 3. Extensibility

### 3.1 Opaque Runtime
The `dtact` runtime is entirely hidden. Users cannot provide their own scheduler or configure the fiber pool (e.g. for NUMA affinity) from the symbolic engine's API.

## 4. Suggestions
- Change `parallel_for_each` to return `Vec<Option<T>>` or a similar structure that lets the caller handle sub-task failures explicitly.
- Expose basic `dtact` configuration knobs through the `ensure_runtime` initialization path.
