# Module Review: `ast` (Phase 5 Audit)

## 3. Extensibility

### 3.1 Conversion Specialization

**Answer:** `build_dag_node` can be extended without modifying the core converter by adding a `ConversionHook: Fn(SymbolKind, &[DagNodeId], &mut DagBuilder) -> Option<DagNodeId>` callback to the `dag_to_ast` call or a builder wrapper. When provided, the hook is called first for each node; returning `Some(id)` uses the custom result, returning `None` falls through to the default `match` on `SymbolKind`. User-defined operators that need special constant folding during reconstruction (e.g. a symbolic derivative that simplifies `d/dx(const) = 0` at conversion time) register a hook. Implementing this hook slot is straightforward: add an optional `hook: Option<Box<dyn ConversionHook>>` to the conversion context struct. It was deferred until a concrete use case exists, since the current extensibility surface (`SymbolKind::Function` + `register_custom_function`) covers the majority of user needs without hooks.
