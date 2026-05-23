# Module Review: `ast` (Phase 4 Audit)

## 1. Performance & Memory

### 1.1 Monolithic Conversion
The conversion logic still uses a large match over `OpKind`.
- **Sharp Question:** As we move toward a "Function"-centric extensibility model, shouldn't the AST conversion be driven by the symbol's metadata rather than a hardcoded list of operators? Why is the AST layer the only part of the system that still needs to know the difference between `Add` and `Mul`?

## 2. Design Integrity

### 2.1 Unused `RelPtr<T, i64>`
The `i64` variant remains unimplemented and unreferenced.
- **Sharp Question:** Is this dead code a "placeholder for the future" or just baggage? If we ever need 64-bit offsets, will we really be projecting an AST so large that it spans 8 quintillion bytes?
The developers note: for this questions, we would like to document that this is intended for future extensibility, just doc it and provide method and steps for user to use it mannuly.
