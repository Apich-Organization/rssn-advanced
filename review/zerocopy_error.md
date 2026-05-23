# Module Review: `zerocopy`, `runtime` & `error` (Phase 5 Audit)

## 3. Sharp Questions

### 3.1 Error Migration Status

**Answer:** Fixed across Phase 4 and Phase 5. The error infrastructure is now fully active:

**Phase 4 completions:**
- All DAG, AST, storage, FFI, and parallel cold constructors are in use throughout their respective modules via `cold_dag_error_*`, `cold_ast_error_*`, etc.
- `rssn_error!` macro generates `#[cold] #[inline(never)] #[track_caller]` constructors for every variant of every error enum.

**Phase 5 additions:**
- `src/jit/compiler.rs`: All `Result<T, String>` returns migrated to `Result<T, JitError>`. Mapping: empty AST → `JitError::MalformedNode`; AST index out of range → `JitError::MalformedNode`; operator arity mismatch → `JitError::MalformedNode`; unregistered function → `JitError::UnknownFunction`; Cranelift declare/define failures → `JitError::InitFailed` or `JitError::VerifierRejected`.
- `src/parser/error.rs`: `cold_parse_error_unexpected_eof(span)` and `cold_parse_error_unexpected_token(span, bad)` added and used in `parse_expression` and `parse_with_table`.
- `src/parser/expr.rs`: `too_deep()` is now `#[cold] #[inline(never)]` — it is the cold-path entry for recursion overflow.
- `src/zerocopy/mod.rs`: `DecodeError::Other("BorrowedSlice: misaligned input buffer")` is behind a private `#[cold] #[inline(never)] fn decode_error_misaligned() -> DecodeError`. We do not own `DecodeError` (it is from `bincode_next`), so the `rssn_error!` macro cannot be applied — the private cold helper is the correct alternative.

There is no "Phase 7" dead letter. Every error variant is reachable, and the `cold_*` constructors ensure error construction cost is zero on the success path.
