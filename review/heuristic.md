# Module Review: `heuristic`

## 1. Performance Issues (High Severity)

### 1.1 Per-Node Allocation in Rewrite Loop
The `HeuristicEngine::rewrite_iterative` function contains an allocation bottleneck:
```rust
let Some(frame) = stack.pop() else { break };
let split_at = values.len().saturating_sub(frame.arity);
let new_children: Vec<DagNodeId> = values.drain(split_at..).collect();
let rebuilt = rebuild_or_match(builder, frame.kind, &new_children, frame.node_id);
```
Collecting children into a `Vec<DagNodeId>` for **every single node** in the rewrite path causes massive heap fragmentation and slows down the simplification process significantly.

### 1.2 Inefficient Identity Pattern Matching
The `patterns::try_apply` function and its subordinates (like `add_zero`) repeatedly perform arena lookups and kind checks for every node in the graph, regardless of whether the node or its children have changed. This "cold" matching approach is $O(N)$ even for a fully simplified graph.

### 1.3 Linearization of Additive Chains
In `approximate_simplify`, pruning an additive chain results in a left-associative binary tree:
```rust
let mut acc = kept[0];
for &term in &kept[1..] {
    acc = builder.add(acc, term);
}
```
If a sum originally has thousands of terms (e.g., in a flattened variadic representation), this pass transforms it into a tree thousands of levels deep. This can cause stack overflows in other parts of the system (like the parser or serializer) and contradicts the goal of "handling symbol explosion."

## 2. Engineering Standards

### 2.1 Fragile Pattern Matching
The pattern matching is hardcoded and limited to basic identities. There is no support for more complex algebraic identities (e.g., distributivity) or associative merging (`(a+b)+c -> a+b+c`) which are often necessary to prevent symbol explosion.

## 3. Suggestions
- Pass slices of the `values` stack to `rebuild_or_match` to avoid per-node `Vec` allocations.
- Implement a "dirty" flag or use the `NodeFlags::CANONICAL` bit to skip pattern matching on nodes known to be already simplified.
- Use a variadic `add` builder or a balanced tree construction for large sums in `approximate_simplify`.
- Expand the pattern matching engine to handle common algebraic simplifications like constant merging in products/sums.
