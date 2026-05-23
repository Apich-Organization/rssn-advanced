# Module Review: `ast` (Phase 5 Audit)

## 3. Extensibility

### 3.1 Conversion Specialization
- **Sharp Question:** As we move toward a "Function"-centric extensibility model, the AST conversion (`build_dag_node`) still relies on a hardcoded `match` for built-in operators. How does a user-defined operator participate in this conversion if it needs more than just a `builder.operator` call (e.g., special constant folding during reconstruction)?
