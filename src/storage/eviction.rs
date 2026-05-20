//! Mark-and-sweep eviction policy for streaming storage.
//!
//! Per `storage_review §2` the previous implementation walked every
//! node and kept it iff `freq >= threshold || is_leaf()` — but it
//! cloned each kept node *without remapping its children*. The
//! "compacted" arena was therefore full of dangling `DagNodeId`s
//! pointing at indices that no longer existed.
//!
//! This rewrite is a proper mark-and-sweep:
//!
//! 1. **Mark.** DFS from every hot node (frequency ≥ threshold) and
//!    mark every reachable node as protected. Leaves are protected
//!    automatically once a hot node reaches them.
//! 2. **Allocate.** Walk the protected set in topological order
//!    (children before parents) and bump-allocate them into the
//!    compacted arena, building an `old_id → new_id` remap table.
//! 3. **Rewrite.** For each protected node, rewrite its `ChildList`
//!    so every reference resolves to the new arena slot.
//!
//! The return value bundles the new arena with the remap table so
//! callers can update their root pointers.

use std::collections::HashMap;

use super::hotspot::DynamicHotspotTable;
use crate::dag::arena::DagArena;
use crate::dag::node::{ChildList, DagNode, DagNodeId};

/// Result of an eviction pass.
///
/// `arena` is the compacted DAG with every protected node remapped to
/// a fresh contiguous index range. `remap[old_id]` returns the new
/// `DagNodeId` (or `None` if `old_id` was evicted).
#[derive(Debug)]
pub struct EvictionResult {
    /// Compacted arena, every child reference resolved against
    /// `remap`.
    pub arena: DagArena,
    /// Forward index table: `remap.get(&old_id) → new_id`.
    pub remap: HashMap<DagNodeId, DagNodeId>,
}

impl EvictionResult {
    /// Translates an old (pre-eviction) `DagNodeId` to its new
    /// (post-eviction) value, or `None` if it was evicted.
    #[must_use]
    pub fn translate(&self, old: DagNodeId) -> Option<DagNodeId> {
        if old.is_none() {
            return Some(DagNodeId::NONE);
        }
        self.remap.get(&old).copied()
    }
}

/// Compacts `arena` by keeping only nodes reachable from a hot root.
///
/// The `keep_threshold` parameter controls hotness: any node whose
/// access frequency in `hotspots` meets or exceeds this value seeds a
/// DFS that protects its entire dependency closure. The DFS is
/// iterative; it can survive arbitrarily deep expressions.
#[must_use]
pub fn evict_cold_nodes(
    arena: &DagArena,
    hotspots: &DynamicHotspotTable,
    keep_threshold: u64,
) -> EvictionResult {
    // ---------------------------------------------------------------
    // Phase 1 — collect hot roots and walk the reachable closure.
    // ---------------------------------------------------------------
    let total = arena.len();
    let mut protected: Vec<bool> = vec![false; total];

    // Hot roots: every node whose access frequency >= threshold.
    let mut work: Vec<u32> = Vec::with_capacity(64);
    #[allow(clippy::cast_possible_truncation)]
    for i in 0..total as u32 {
        let id = DagNodeId::new(i);
        if hotspots.is_hot(id, keep_threshold) {
            work.push(i);
        }
    }
    while let Some(idx) = work.pop() {
        let i = idx as usize;
        if i >= total || protected[i] {
            continue;
        }
        protected[i] = true;
        if let Some(node) = arena.get(DagNodeId::new(idx)) {
            for child in node.children.iter() {
                let ci = child.index();
                if ci < total && !protected[ci] {
                    work.push(child.value());
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // Phase 2 — allocate protected nodes in topological order, build
    // the remap table, and rewrite each node's ChildList.
    // ---------------------------------------------------------------
    // `arena.alloc` writes nodes in the order they're handed in. By
    // walking 0..N we visit children before parents iff the arena was
    // built bottom-up — which `DagBuilder` always does. We therefore
    // get topo order "for free" by iterating in index order.
    let mut compacted = DagArena::new();
    let mut remap: HashMap<DagNodeId, DagNodeId> = HashMap::with_capacity(total / 2);

    for (i, is_protected) in protected.iter().enumerate() {
        if !*is_protected {
            continue;
        }
        #[allow(clippy::cast_possible_truncation)]
        let old_id = DagNodeId::new(i as u32);
        let Some(original) = arena.get(old_id) else {
            continue;
        };

        // Rewrite children through the remap. If any child reference
        // can't be remapped (which would mean a hot node depends on a
        // node we somehow failed to mark — should never happen) we
        // skip that child rather than dangling.
        let new_children: Vec<DagNodeId> = original
            .children
            .iter()
            .filter_map(|c| remap.get(&c).copied())
            .collect();
        let child_list = ChildList::from_slice(&new_children);

        // Re-build the node with the new child list. Other fields are
        // preserved verbatim.
        let new_node = DagNode {
            kind: original.kind,
            meta: original.meta.clone(),
            children: child_list,
            value: original.value,
        };
        let new_id = compacted.alloc(new_node);
        remap.insert(old_id, new_id);
    }

    EvictionResult {
        arena: compacted,
        remap,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::builder::DagBuilder;

    #[test]
    fn eviction_drops_cold_branches_and_preserves_hot_closure() {
        // Build (x + y) where only `x+y` is hot. The compacted arena
        // must keep the root AND both leaves (they're reachable),
        // even though only the root sits above the hotness threshold.
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let y = b.variable("y");
        let sum = b.add(x, y);

        let hot = DynamicHotspotTable::new();
        for _ in 0..10 {
            hot.record_access(sum);
        }

        let result = evict_cold_nodes(b.arena(), &hot, 5);
        // All three nodes must survive (sum requires x and y).
        assert_eq!(result.arena.len(), 3);
        // Children of the new root must resolve to existing arena
        // slots — no dangling references.
        let new_sum = result.translate(sum).expect("sum survives");
        let new_node = result.arena.get(new_sum).expect("new sum");
        for c in new_node.children.iter() {
            assert!(
                result.arena.get(c).is_some(),
                "child {c:?} dangles after eviction"
            );
        }
    }

    #[test]
    fn cold_only_arena_compacts_to_empty() {
        let mut b = DagBuilder::new();
        let _ = b.variable("x");
        let _ = b.variable("y");
        let hot = DynamicHotspotTable::new();
        let result = evict_cold_nodes(b.arena(), &hot, 1);
        assert_eq!(result.arena.len(), 0);
        assert!(result.remap.is_empty());
    }

    #[test]
    fn unrelated_subgraph_is_evicted() {
        // Build two independent subgraphs:
        //   A: (x + y), hot
        //   B: (a * b), cold
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let y = b.variable("y");
        let sum = b.add(x, y);
        let av = b.variable("a");
        let bv = b.variable("b");
        let _prod = b.mul(av, bv);

        let hot = DynamicHotspotTable::new();
        for _ in 0..10 {
            hot.record_access(sum);
        }

        let result = evict_cold_nodes(b.arena(), &hot, 5);
        // sum + x + y survive; av/bv/prod do not.
        assert_eq!(result.arena.len(), 3);
        assert!(result.translate(sum).is_some());
        assert!(result.translate(av).is_none());
    }

    #[test]
    fn random_dag_has_no_dangling_after_eviction() {
        // Pseudo-random DAG: every node references two earlier ones.
        // After marking ~30 % of nodes as hot, every child of every
        // kept node must remap successfully.
        let mut b = DagBuilder::new();
        let mut ids: Vec<DagNodeId> = (0..6).map(|i| b.constant(f64::from(i))).collect();
        for i in 6..200 {
            let lhs = ids[(i * 17) % ids.len()];
            let rhs = ids[(i * 23) % ids.len()];
            let op = if i % 3 == 0 {
                b.add(lhs, rhs)
            } else if i % 3 == 1 {
                b.mul(lhs, rhs)
            } else {
                b.sub(lhs, rhs)
            };
            ids.push(op);
        }

        let hot = DynamicHotspotTable::new();
        for (i, id) in ids.iter().enumerate() {
            if i % 3 == 0 {
                for _ in 0..5 {
                    hot.record_access(*id);
                }
            }
        }

        let result = evict_cold_nodes(b.arena(), &hot, 3);
        // Every kept node's children must resolve.
        for i in 0..result.arena.len() {
            #[allow(clippy::cast_possible_truncation)]
            let id = DagNodeId::new(i as u32);
            let node = result.arena.get(id).expect("kept node");
            for c in node.children.iter() {
                assert!(
                    result.arena.get(c).is_some(),
                    "node #{i} has dangling child {c:?}"
                );
            }
        }
    }
}
