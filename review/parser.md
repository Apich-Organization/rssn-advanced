# Module Review: `parser` (Phase 5 Audit)

## 1. Performance

### 1.1 Redundant Whitespace
`parse_expr_climbing` still calls `.trim_start()` in a loop.
- **Sharp Question:** We have a lexer module. Why isn't the lexer responsible for whitespace stripping? Why is our parser manually trimming strings while traversing the precedence tree?

## 2. Extensibility

### 2.2 Unary Extension
- **Sharp Question:** We have a `PrecedenceTable` for infix operators, but `parse_atom` has a hardcoded `-` handler. How does a user register a custom unary operator (e.g., `!x`, `~x`) or a custom prefix/postfix operator without modifying `expr.rs`?
