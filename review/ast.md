# Module Review: `ast`

## 1. Performance Issues (High Severity)

### 1.1 Bloated `AstNode` Size
The `AstNode` struct is approximately **64 bytes**.
- `SymbolKind`: 8 bytes.
- `value: Option<f64>`: 16 bytes.
- `dag_id: DagNodeId`: 4 bytes.
- `children: AstChildList`: 32 bytes (due to `Vec` variant).
- Padding: 4 bytes.
This contradicts the `plan.md` which states that AST nodes should be "highly compact" and use "relative pointers" to save space. A 64-byte node is nearly as large as the 80-byte `DagNode`. Most of the data stored in `AstNode` is redundant with the `DagNode` it points to.

### 1.2 Allocation-Heavy Conversion
In `src/ast/convert.rs`, the `dag_to_ast` function uses a `DagFrame` struct that contains a `Vec<usize>`:
```rust
struct DagFrame {
    // ...
    child_ast_indices: Vec<usize>,
    // ...
}
```
This causes a heap allocation for **every single node** during conversion. For large expressions, this will result in millions of small allocations, severely degrading performance.

## 2. Correctness Issues

### 2.1 Ambiguous Null Sentinel in `RelPtr`
`RelPtr::null()` uses an offset of `0`. However, `0` is also used as a valid index for the root of the `AstProjection`.
```rust
pub fn from_indices_checked(source: usize, target: usize) -> Option<Self> {
    if target == 0 {
        return Some(Self::null()); // BUG: target index 0 is treated as null
    }
    // ...
}
```
Any node that attempts to point to the root node (index 0) will instead be interpreted as having a null child.

### 2.2 Broken Variable Reconstruction in `ast_to_dag`
When converting AST back to DAG, the code attempts to look up variable names from the `DagBuilder`'s registry using the old `SymbolId`. If the `SymbolId` is not present in the new registry, it defaults to `"x"`:
```rust
let name = builder.registry().name(sym_id).unwrap_or("x").to_owned();
builder.variable(&name)
```
This leads to data loss where different variables (e.g., `y`, `z`) are all merged into a single variable `x` if they were not already interned in the builder.

## 3. Engineering Standards

### 3.1 Suboptimal Enum Sizing
`AstChildList` uses the same suboptimal enum layout as `ChildList` in the DAG module, where the size is dictated by the `Many(Vec<RelPtr<AstNode>>)` variant even for leaf nodes.

## 4. Suggestions
- Store only the `DagNodeId` and child relative pointers in `AstNode`. Fetch kind and value from the DAG only when needed.
- Use a single `Vec` for all child indices in `dag_to_ast` and use slices/offsets to manage the stack frames, avoiding per-node allocations.
- Fix the `RelPtr` null sentinel (e.g., use `i32::MIN` or another impossible value).
- Ensure variable names are preserved during `ast_to_dag` by either carrying names in the AST or ensuring registry consistency.
