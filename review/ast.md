# Module Review: `ast` (Phase 3 Audit)

## 1. Performance & Memory

### 1.1 Boxed Slice Overhead
`AstChildList::Many` uses `Box<[RelPtr<AstNode>]>`.
- **Sharp Question:** We optimized the nodes to be compact, but `Many` still spills to the heap. If an expression has thousands of variadic children, aren't we just moving the allocation pressure from the "node" level to the "child list" level? Why not use a side-pool in the `AstProjection` similar to the `PackedArenaImage`?

## 2. Extensibility

### 2.1 The "Giant Match" Anti-Pattern
`ast_to_dag` and `dag_to_ast` use massive `match` statements over `OpKind`.
- **Sharp Question:** If we add 50 new operators, will this function become a 2000-line maintenance nightmare? Is there no way to register "Conversion Handlers" for new symbol kinds, or are we committed to a monolithic architecture?

## 3. Design Integrity

### 3.1 Unused `RelPtr<T, i64>`
The `i64` variant of relative pointers is implemented but never used.
- **Sharp Question:** Why maintain the complexity of a generic `RelPtr<T, O>` and an unused `i64` implementation if our arena size is strictly bounded by `u32` (4GB)? Is this "just-in-case" engineering or a signal for a future feature that hasn't arrived?
