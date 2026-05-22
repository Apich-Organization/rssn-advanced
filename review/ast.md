# Module Review: `ast` (Post-Upgrade)

## 1. Performance & Memory

### 1.1 `AstNode` Compaction
The transition to using `f64` instead of `Option<f64>` in `AstNode` successfully reduced the node size. The `dag_to_ast` conversion now uses a shared `child_pool`, which is a major performance win as it eliminates the per-node heap allocation.

## 2. Dead Code & Functionality

### 2.1 Missing `AstChildList::Many` Usage in Rebuild
While `AstChildList::Many` uses `Box<[RelPtr<AstNode>]>` to save space, the `backpatch_children` function still creates a temporary `Vec` and then converts it to `Box`. This is a minor allocation, but it could be optimized by using a pre-allocated pool for the `RelPtr`s as well.

## 3. Extensibility

### 3.1 Closed Conversion Path
The `dag_to_ast` and `ast_to_dag` functions are strictly tied to the internal `SymbolKind` and `OpKind`. If a user wants to extend the AST with their own metadata or node types (e.g. for specialized compiler passes), they must modify the core `ast` module. There is no support for a "Visitor" or "Generic Tree" pattern that would allow external extensions.

## 4. Suggestions
- Implement a more generic traversal or visitor pattern for the `AstProjection` to allow external tools to process the AST without modifying the library.
- Consider moving the `child_pool` pattern to the `ast_to_dag` side as well (it currently uses `Vec<DagNodeId>` per frame).
