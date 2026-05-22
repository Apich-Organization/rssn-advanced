# Module Review: `zerocopy` (Post-Upgrade)

## 1. Performance & Memory

### 1.1 True Zero-Copy Restoration
The `decode_zerocopy_raw` function, combined with `MmapBuffer` alignment guarantees, now enables true zero-copy restoration of DAG arenas from disk. This is a major improvement over the previous version which performed a full heap copy for alignment.

## 2. Dead Code & Functionality

### 2.1 Unused `RelPtr::null_i64`
While the `i32` version of `RelPtr` is used extensively in the AST, the `i64` variant and its associated `null_i64` constant are currently **dead code**. No part of the system currently requires the larger 64-bit offsets.

## 3. Extensibility

### 3.1 Manual `Pod` Marker
The `Pod` trait remains a manual `unsafe` marker. There is no proc-macro support for verifying that structs (like `PackedDagNode`) remain POD-safe as they are modified by future developers.

## 4. Suggestions
- Remove the `i64` variant of `RelPtr` if it is not expected to be used, or implement a usage for extremely large "out-of-core" AST projections.
- Introduce a `bytemuck`-style derive macro for `Pod` to prevent safety regressions.
