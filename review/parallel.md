# Module Review: `parallel` (Phase 5 Audit)

## 3. Sharp Questions

### 3.1 The "Steps" Overhead

**Answer:** Fixed in Phase 4 — `repr(align(128))` was removed from `ThreadLocalState`. Without the padding attribute there is no false-sharing cost because each thread's `ThreadLocalState` lives in its own `thread_local!` slot, which the OS places in thread-local storage (TLS) — not adjacent to other threads' data in a shared cache line. The 128-byte alignment was defensive against a theoretical scenario where multiple `ThreadLocalState` objects shared a cache line, but TLS makes that impossible. The `ThreadLocalState` step counter IS read externally via `get_count()` and similar accessors, so it cannot be removed without breaking the observation API. The current cost is a single non-atomic increment per evaluation step with no cross-thread visibility, which is zero overhead in practice.
