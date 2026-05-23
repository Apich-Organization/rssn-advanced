# Module Review: `ffi` (Phase 3 Audit)

## 1. Security

### 1.1 The "Time Bomb" v1 Async API
`rssn_dag_simplify_async` (v1) still captures a raw pointer and returns immediately.
- **Sharp Question:** We have a known Use-After-Free vulnerability in our public API. Why is it still here? Is "v1 compatibility" worth a potential security breach in a library designed for industrial use?

## 2. Design Integrity

### 2.1 Opaque Handle Inconsistency
We have `*mut DagBuilder`, `*mut RssnJitContext`, and `*mut RssnAsyncHandle`.
- **Sharp Question:** Why do some handles use `Box::into_raw` and others use `Arc::into_raw` (if they do)? Is there a unified ownership model for our C API, or is it just a collection of various `void*` stubs?

## 3. Extensibility

### 3.1 Hardcoded Config Defaults
`rssn_dag_simplify` and its async counterparts hardcode `HeuristicConfig::default()` and `SearchStrategy::Greedy`.
- **Sharp Question:** How does a C user tune the search timeout, depth, or strategy? Why are we exposing a "heuristic engine" but stripping away all its "knobs" at the FFI boundary?
