# Module Review: `parser`

## 1. Performance Issues (High Severity)

### 1.1 Unbounded Recursion for Right-Associative Operators
While parenthesis nesting is capped at `MAX_PAREN_DEPTH`, right-associative operators (like `^`) cause direct recursion in `parse_expr_climbing`:
```rust
let next_min_prec = if op_right_associative(op_char) {
    op_prec
} else {
    op_prec + 1
};
let (rem_after_rhs, rhs) = parse_expr_climbing(rem, builder, next_min_prec, depth)?;
```
A long chain of exponentiations (e.g., `x^y^z^...`) will bypass the parenthesis depth check and can overflow the OS stack.

### 1.2 Redundant Whitespace Parsing
The `ws` combinator strips whitespace both before and after every token. This leads to redundant checks and character scanning, as the "after" whitespace of one token is the "before" whitespace of the next.

## 2. Correctness & Functionality Issues

### 2.1 Missing Function Support
The parser does not support function calls (e.g., `sin(x)`), even though the rest of the engine (DAG, JIT, Heuristic) supports `SymbolKind::Function`. This makes the library unusable for any expression involving functions.

### 2.2 Limited Operator Set
The parser only supports `+`, `-`, `*`, `/`, `^`, and unary `-`. Other common symbolic operators or custom operators mentioned in the plan are not implemented.

## 3. Engineering Standards

### 3.1 Error Message Quality
Parse errors rely on `nom::error::ErrorKind`, which produces generic messages like "Parser failed: Fail" or "Parser failed: Alpha". These are not helpful for end-users trying to debug a complex symbolic expression.

## 4. Suggestions
- Implement a global recursion depth counter to protect against deep right-associative chains.
- Optimize whitespace handling to only strip leading or trailing whitespace per token.
- Implement function call parsing (e.g., `identifier '(' args ')'`).
- Improve error reporting by using custom error types that provide descriptive messages for common syntax errors.
