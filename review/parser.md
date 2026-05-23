# Module Review: `parser` (Phase 3 Audit)

## 1. Performance

### 1.1 The "Trim" Tax
`parse_expr_climbing` calls `.trim_start()` on the input in every loop iteration.
- **Sharp Question:** If we are parsing a 1MB expression string, how many millions of times are we re-scanning the same leading whitespace? Why not use a proper lexer that consumes whitespace once and for all?

## 2. Extensibility

### 2.1 Static Precedence
Precedence is hardcoded in a `match` on `char`.
- **Sharp Question:** How does a user add a new infix operator (e.g. `xor` or `dot`) with custom precedence? Is our parser's grammar "fixed" at compile-time?

## 3. Correctness

### 3.1 Unary Minus Precedence
Unary minus is handled in `parse_atom` with a hardcoded precedence of `4`.
- **Sharp Question:** Is this correct for all expressions? Does `-x^y` parse as `-(x^y)` or `(-x)^y`? Are we sure our hardcoded "precedence climbing" matches standard mathematical conventions for all edge cases?
