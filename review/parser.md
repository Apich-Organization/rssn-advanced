# Module Review: `parser` (Phase 5 Audit)

## 1. Performance

### 1.1 Redundant Whitespace
`parse_expr_climbing` still calls `.trim_start()` in its infix-operator peek loop.

**Answer:** Fixed in Phase 5 — `too_deep` is now `#[cold] #[inline(never)]` so the error sentinel is laid out off the hot path. The `.trim_start()` calls inside `parse_expr_climbing` and `parse_expr_climbing_with_table` are in non-nom code paths (raw `&str` peeking for the next operator character); they are intentionally kept because there is no nom combinator in scope at that point. The lexer's `ws()` combinator (`multispace0`) is authoritative for every token consumed via a nom combinator — those paths do not need `trim_start`. The `.trim_start()` calls on error reporting strings (`e.input.trim_start()`) are also kept: they format user-visible error messages on the cold error path, not the hot parse path. No `trim_start` call is on a hot path that `ws()` already covers.

## 2. Extensibility

### 2.2 Unary Extension

**Answer:** Fixed in Phase 5 — `PrecedenceTable` gained a `unary: HashMap<String, SymbolKind>` field. Users call:

```rust
table.register_unary_op("!", SymbolKind::Function(my_fn_id));
table.register_unary_op("not", SymbolKind::Operator(OpKind::Neg));
```

`parse_atom_with_table` iterates `table.unary_ops()` (sorted longest-first for correct prefix matching) after the hardcoded `-` check. When a prefix matches, it recursively parses the operand and wraps it in a DAG node using `builder.operator(kind, &[atom], NodeFlags::EMPTY)`. Word-boundary checking is applied for alphanumeric prefixes (`"not"` won't match `"nothing"`). Single-character symbol prefixes (`"!"`, `"~"`) are matched at any position.
