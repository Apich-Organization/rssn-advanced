# Module Review: `heuristic` (Phase 5 Audit)

## 4. Sharp Questions

### 4.1 Registry Serialization

**Answer:** Closures cannot be serialized. The `rule_set_fingerprint()` (Phase 4) and `PackedArenaImage::rule_fingerprint` field give serialization SAFETY without closure persistence: if the fingerprint mismatches on load (because rules changed between the process that wrote the cache and the process that reads it), the CANONICAL bits are cleared and heuristic derivation re-runs from scratch. The derived results are deterministic given the same rules, so correctness is maintained. Users who need rule persistence across processes should store rule names as strings and reconstruct closures at startup — the `register_named()` API in `RuleRegistry` supports this pattern: name → closure mapping can be re-established from a static lookup table keyed by name. This is the same approach taken by Lua, Python pickle (for named functions), and most rule-engine frameworks.
