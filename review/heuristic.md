# Module Review: `heuristic` (Phase 4 Audit)

## 1. Sharp Questions

### 1.1 Registry Serialization
- **Sharp Question:** If a user registers 100 rules at runtime, how do those rules get serialized into a `PackedArenaImage` for disk caching? Is our "custom extensibility" only valid for the current process, or is there a plan for Rule persistence?
