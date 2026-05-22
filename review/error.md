# Module Review: `error` (Post-Upgrade)

## 1. Performance & Memory

### 1.1 Cold-Path Optimization
The `rssn_error!` macro correctly generates `#[cold]` constructors, which is a best practice for keeping error handling logic out of the hot instruction cache.

## 2. Dead Code & Functionality

### 2.1 Vastly Unreferenced Infrastructure
Despite the elaborate macro system and detailed error enums, the vast majority of the `error` module is currently **dead code**. A search of the codebase reveals that almost none of the `cold_*` functions are actually called in the core logic (except for one usage in `runtime`). Most components still use silent short-circuits or partial results instead of returning these rich error types.

## 3. Extensibility

### 3.1 Macro Boilerplate
The `rssn_error!` macro is powerful but rigid. Adding a new error variant requires re-running the macro and potentially updating several manual `Display` implementations.

## 4. Suggestions
- Implement the "Phase 7" task of replacing all `.expect()`, `.unwrap()`, and silent short-circuits with calls to the `cold_*` error constructors.
- Consolidate the error types if the full complexity of 7 different enums is not truly needed.
