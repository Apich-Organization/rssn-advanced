# Module Review: `zerocopy` & `error` (Phase 3 Audit)

## 1. Safety

### 1.1 The manual `Pod` Trap
`PackedDagNode` and `BorrowedSlice` implement `Pod` manually.
- **Sharp Question:** If a developer adds a `bool` field to `PackedDagNode`, they introduce 7 bytes of undefined padding. How many days of debugging will it take to find the resulting memory corruption? Why are we not using a verified `Pod` derive macro?

## 2. Implementation Integrity

### 2.1 The Ghost Error System
We have an elaborate `rssn_error!` macro system and dozens of `cold_*` functions.
- **Sharp Question:** Why are we still using `u32::MAX` sentinels and silent short-circuits in our core logic when we have a purpose-built error system? Is our error handling architecture just "documentation" or is it intended to be used?

## 3. Performance

### 3.1 The "Box" in `TaskEnvelope`
`TaskEnvelope` reduces spawns to a single allocation.
- **Sharp Question:** Why are we allocating at all for every fiber task? Could we use a slab-allocated or pool-allocated task buffer to achieve truly "advanced" parallel performance?
