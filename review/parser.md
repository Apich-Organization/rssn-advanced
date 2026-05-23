# Module Review: `parser` (Phase 4 Audit)

## 1. Performance

### 1.1 Redundant Whitespace
`parse_expr_climbing` still calls `.trim_start()` in a loop.
- **Sharp Question:** We have a lexer module. Why isn't the lexer responsible for whitespace stripping? Why is our parser manually trimming strings while traversing the precedence tree?

## 2. Extensibility

### 2.1 Static Precedence
- **Sharp Question:** Users can now register custom rewrite rules and JIT functions, but they can't use them in strings because the parser's operator set is hardcoded. How does a user parse `sin(x) dot y` if `dot` is a custom operator? Is the parser becoming the bottleneck for high-level usability?
