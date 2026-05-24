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

use super::hotspot::DynamicHotspotTable;
use crate::dag::arena::DagArena;
use crate::dag::node::{ChildList, DagNode, DagNodeId};

// =========================================================================
// CompactRemap — hierarchical rank-based translation table (Part 2b)
// =========================================================================

/// A memory-compact remap table using a hierarchical popcount rank structure.
///
/// For N old slots:
/// - `blocks`: 1 bit per slot → N/8 bytes (16 KB for 1 M nodes).
/// - `block_prefix`: 1 u32 per 64-slot block → N/16 bytes (64 KB for 1 M nodes).
///
/// Total ≈ 80 KB for 1 M nodes, a 50× reduction vs `Vec<DagNodeId>` (4 MB).
/// `translate` is O(1) with two array accesses and one `count_ones()`.
#[derive(Debug)]
pub struct CompactRemap {
    /// Bitset: bit `i` is set iff slot `i` survived eviction.
    blocks: Vec<u64>,
    /// Prefix popcount: `block_prefix[w]` = number of surviving slots in
    /// words `0..w` (exclusive). Length = `blocks.len()`.
    block_prefix: Vec<u32>,
    /// Total number of surviving slots.
    n_kept: u32,
}

impl CompactRemap {
    /// Builds a `CompactRemap` from an iterator of old slot indices that survived.
    ///
    /// `n_slots` is the total number of old arena slots (surviving + evicted).
    /// `kept` yields each surviving old slot index exactly once, in any order.
    #[must_use]
    pub fn build(n_slots: usize, kept: impl Iterator<Item = usize>) -> Self {
        let n_words = n_slots.div_ceil(64);
        let mut blocks = vec![0u64; n_words];
        let mut n_kept = 0u32;
        for idx in kept {
            let word = idx / 64;
            let bit = idx % 64;
            if word < n_words {
                blocks[word] |= 1u64 << bit;
                n_kept += 1;
            }
        }
        // Build exclusive-prefix popcount table.
        let mut block_prefix = Vec::with_capacity(n_words);
        let mut running = 0u32;
        for &w in &blocks {
            block_prefix.push(running);
            running = running.saturating_add(w.count_ones());
        }
        Self {
            blocks,
            block_prefix,
            n_kept,
        }
    }

    /// Translates an old (pre-eviction) `DagNodeId` to its new (post-eviction) id.
    ///
    /// Returns `None` if `old_id` was evicted (bit not set in bitset).
    #[must_use]
    pub fn translate(&self, old_id: DagNodeId) -> Option<DagNodeId> {
        if old_id.is_none() {
            return Some(DagNodeId::NONE);
        }
        let idx = old_id.index();
        let word = idx / 64;
        let bit = idx % 64;
        let block = *self.blocks.get(word)?;
        if block & (1u64 << bit) == 0 {
            return None; // evicted
        }
        // rank = number of set bits up to and including position `bit` in word `word`.
        let prefix = self.block_prefix[word];
        // Mask: all bits at positions 0..=bit (inclusive).
        let mask = (1u64 << bit).wrapping_shl(1).wrapping_sub(1) | (1u64 << bit);
        let rank = prefix + (block & mask).count_ones();
        // rank is 1-based (the surviving node occupies arena slot rank-1).
        Some(DagNodeId::new(rank - 1))
    }

    /// Total number of surviving slots.
    #[must_use]
    pub const fn n_kept(&self) -> u32 {
        self.n_kept
    }
}

/// Result of an eviction pass.
///
/// `arena` is the compacted DAG with every protected node remapped to
/// a fresh contiguous index range.
///
/// `remap` is a [`CompactRemap`] bitset-based translation table.
/// For 1 M nodes it occupies ≈ 80 KB, a 50× reduction vs a flat
/// `Vec<DagNodeId>` (4 MB). Use [`EvictionResult::translate`] for lookups.
#[derive(Debug)]
pub struct EvictionResult {
    /// Compacted arena, every child reference resolved against `remap`.
    pub arena: DagArena,
    /// Compact bitset remap: surviving old slot → new slot id.
    pub remap: CompactRemap,
}

impl EvictionResult {
    /// Translates an old (pre-eviction) `DagNodeId` to its new
    /// (post-eviction) value, or `None` if it was evicted.
    #[must_use]
    pub fn translate(&self, old: DagNodeId) -> Option<DagNodeId> {
        self.remap.translate(old)
    }
}

/// Trait for pluggable node eviction policies.
///
/// Implement this to define custom hotness criteria (e.g. LRU, priority-based,
/// or time-decay). The `is_hot` method is called once per node during the mark
/// phase; returning `true` seeds a DFS that protects the node's entire
/// dependency closure.
pub trait EvictionPolicy {
    /// Returns `true` if `id` should be retained (protected from eviction).
    fn is_hot(
        &self,
        hotspots: &super::hotspot::DynamicHotspotTable,
        id: crate::dag::node::DagNodeId,
    ) -> bool;
}

/// Frequency-threshold eviction policy: protect any node whose access count
/// meets or exceeds `threshold`.
pub struct FrequencyPolicy {
    /// Minimum access count to be considered "hot".
    pub threshold: u64,
}

impl FrequencyPolicy {
    /// Creates a new `FrequencyPolicy` with the given threshold.
    #[must_use]
    pub fn new(threshold: u64) -> Self {
        Self { threshold }
    }
}

impl EvictionPolicy for FrequencyPolicy {
    fn is_hot(
        &self,
        hotspots: &super::hotspot::DynamicHotspotTable,
        id: crate::dag::node::DagNodeId,
    ) -> bool {
        hotspots.is_hot(id, self.threshold)
    }
}

/// Protects nodes that contribute at least `min_ratio` of total accesses.
/// Provides decay-like behaviour: a frequently-accessed node that hasn't
/// been touched recently will see its ratio fall as other nodes are accessed.
pub struct RecencyWeightedPolicy {
    /// Minimum raw access count.
    pub min_freq: u64,
    /// node.freq / total_accesses must be >= this to be considered hot.
    pub min_ratio: f64,
}

impl RecencyWeightedPolicy {
    /// Creates a new `RecencyWeightedPolicy`.
    #[must_use]
    pub fn new(min_freq: u64, min_ratio: f64) -> Self {
        Self {
            min_freq,
            min_ratio,
        }
    }
}

impl EvictionPolicy for RecencyWeightedPolicy {
    fn is_hot(
        &self,
        hotspots: &super::hotspot::DynamicHotspotTable,
        id: crate::dag::node::DagNodeId,
    ) -> bool {
        let freq = hotspots.get_frequency(id);
        if freq < self.min_freq {
            return false;
        }
        let total = hotspots.total_accesses();
        if total == 0 {
            return false;
        }
        (freq as f64 / total as f64) >= self.min_ratio
    }
}

/// Composes two policies with AND or OR logic.
pub struct CompositePolicy<A: EvictionPolicy, B: EvictionPolicy> {
    /// First policy.
    pub primary: A,
    /// Second policy.
    pub secondary: B,
    /// `true` = AND (both must be hot), `false` = OR (either must be hot).
    pub require_both: bool,
}

impl<A: EvictionPolicy, B: EvictionPolicy> CompositePolicy<A, B> {
    /// Creates a policy that requires **both** `a` and `b` to consider a node hot.
    #[must_use]
    pub fn and(a: A, b: B) -> Self {
        Self {
            primary: a,
            secondary: b,
            require_both: true,
        }
    }
    /// Creates a policy that requires **either** `a` or `b` to consider a node hot.
    #[must_use]
    pub fn or(a: A, b: B) -> Self {
        Self {
            primary: a,
            secondary: b,
            require_both: false,
        }
    }
}

impl<A: EvictionPolicy, B: EvictionPolicy> EvictionPolicy for CompositePolicy<A, B> {
    fn is_hot(
        &self,
        hotspots: &super::hotspot::DynamicHotspotTable,
        id: crate::dag::node::DagNodeId,
    ) -> bool {
        if self.require_both {
            self.primary.is_hot(hotspots, id) && self.secondary.is_hot(hotspots, id)
        } else {
            self.primary.is_hot(hotspots, id) || self.secondary.is_hot(hotspots, id)
        }
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
    evict_nodes_with_policy(arena, hotspots, FrequencyPolicy::new(keep_threshold))
}

/// Compacts `arena` using a caller-supplied eviction policy.
///
/// The `policy` determines which nodes are "hot" (and thus protected from eviction).
/// Use [`FrequencyPolicy`] for the classic frequency-threshold behavior, or
/// provide your own [`EvictionPolicy`] implementation for custom criteria
/// (e.g. LRU, priority-based, or time-decay weighting).
#[must_use]
pub fn evict_nodes_with_policy<P: EvictionPolicy>(
    arena: &DagArena,
    hotspots: &DynamicHotspotTable,
    policy: P,
) -> EvictionResult {
    evict_nodes_budgeted_with_policy(arena, hotspots, policy, usize::MAX)
}

/// Like `evict_cold_nodes` but limits compaction to at most `budget` protected nodes.
/// The mark phase always runs fully; only the sweep is bounded.
/// Pass `usize::MAX` for unlimited (equivalent to the original).
#[must_use]
pub fn evict_cold_nodes_budgeted(
    arena: &DagArena,
    hotspots: &DynamicHotspotTable,
    keep_threshold: u64,
    budget: usize,
) -> EvictionResult {
    evict_nodes_budgeted_with_policy(
        arena,
        hotspots,
        FrequencyPolicy::new(keep_threshold),
        budget,
    )
}

/// Like [`evict_nodes_with_policy`] but limits the sweep to at most `budget` protected nodes.
/// The mark phase always runs fully; only the sweep is bounded.
#[must_use]
pub fn evict_nodes_budgeted_with_policy<P: EvictionPolicy>(
    arena: &DagArena,
    hotspots: &DynamicHotspotTable,
    policy: P,
    budget: usize,
) -> EvictionResult {
    let total = arena.len();
    let mut protected: Vec<bool> = vec![false; total];

    let mut work: Vec<u32> = Vec::with_capacity(64);
    #[allow(clippy::cast_possible_truncation)]
    for i in 0..total as u32 {
        let id = DagNodeId::new(i);
        if policy.is_hot(hotspots, id) {
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

    // Sweep phase: allocate surviving nodes into the compacted arena.
    // `flat_remap[old_index]` holds the new DagNodeId (or NONE) so that
    // child references can be rewritten during this pass. A CompactRemap
    // is built afterwards from the set of surviving old indices.
    let mut compacted = DagArena::new();
    let mut flat_remap: Vec<DagNodeId> = vec![DagNodeId::NONE; total];
    let mut kept_indices: Vec<usize> = Vec::new();
    let mut count = 0usize;

    for (i, is_protected) in protected.iter().enumerate() {
        if !*is_protected {
            continue;
        }
        if count >= budget {
            break;
        }
        count += 1;
        #[allow(clippy::cast_possible_truncation)]
        let old_id = DagNodeId::new(i as u32);
        let Some(original) = arena.get(old_id) else {
            continue;
        };

        let new_children: Vec<DagNodeId> = original
            .children
            .iter()
            .filter_map(|c| {
                let mapped = flat_remap
                    .get(c.index())
                    .copied()
                    .unwrap_or(DagNodeId::NONE);
                if mapped.is_none() { None } else { Some(mapped) }
            })
            .collect();
        let child_list = ChildList::from_slice(&new_children);

        let new_node = DagNode {
            kind: original.kind,
            meta: original.meta.clone(),
            children: child_list,
        };
        let new_id = compacted.alloc(new_node);
        flat_remap[old_id.index()] = new_id;
        kept_indices.push(i);
    }

    // Post-pass: clear CANONICAL bits from nodes whose children are absent.
    // When the budget cuts the sweep early, a node may be copied into the
    // compacted arena but some of its children may have been excluded (their
    // old ids map to NONE in `flat_remap`). The CANONICAL flag would then be a
    // lie — the subtree is incomplete.
    let compacted_len = compacted.len();
    let mut to_clear: Vec<DagNodeId> = Vec::new();
    for i in 0..compacted_len {
        #[allow(clippy::cast_possible_truncation)]
        let id = DagNodeId::new(i as u32);
        if let Some(node) = compacted.get(id) {
            if node.meta.flags.is_canonical() {
                let children_valid = node
                    .children
                    .iter()
                    .all(|c| !c.is_none() && c.index() < compacted_len);
                if !children_valid {
                    to_clear.push(id);
                }
            }
        }
    }
    for id in to_clear {
        if let Some(node) = compacted.get_mut(id) {
            node.meta.flags = node.meta.flags.without_canonical();
        }
    }

    // Build the compact remap from the list of surviving old slot indices.
    let remap = CompactRemap::build(total, kept_indices.into_iter());

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
        // CompactRemap has no surviving slots — all old ids translate to None.
        assert_eq!(result.remap.n_kept(), 0);
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

    #[test]
    fn recency_weighted_policy_protects_dominant_nodes() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let y = b.variable("y");
        let sum = b.add(x, y);

        let hot = DynamicHotspotTable::new();
        // x+y gets 90 accesses, x gets 10 — ratio 0.9 vs 0.1 total 100.
        for _ in 0..90 {
            hot.record_access(sum);
        }
        for _ in 0..10 {
            hot.record_access(x);
        }

        // min_freq=5, min_ratio=0.5 → only sum qualifies (ratio 0.9 >= 0.5).
        let policy = RecencyWeightedPolicy::new(5, 0.5);
        let result = evict_nodes_with_policy(b.arena(), &hot, policy);
        // sum + its children (x, y) must survive.
        assert_eq!(result.arena.len(), 3);
        assert!(result.translate(sum).is_some());
    }

    #[test]
    fn composite_and_policy_requires_both_conditions() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let y = b.variable("y");
        let sum = b.add(x, y);

        let hot = DynamicHotspotTable::new();
        for _ in 0..10 {
            hot.record_access(sum);
        }

        // AND: freq >= 5 AND ratio >= 0.05. sum satisfies both; x/y have freq=0.
        let policy =
            CompositePolicy::and(FrequencyPolicy::new(5), RecencyWeightedPolicy::new(1, 0.05));
        let result = evict_nodes_with_policy(b.arena(), &hot, policy);
        assert!(result.translate(sum).is_some());
        // x and y have 0 accesses so they're cold in the AND policy,
        // but they're transitively reachable from sum, so they survive.
        assert_eq!(result.arena.len(), 3);
    }

    #[test]
    fn budgeted_eviction_respects_limit() {
        let mut b = DagBuilder::new();
        // 6 constants — all will be hot.
        for i in 0..6 {
            let _ = b.constant(f64::from(i));
        }
        let hot = DynamicHotspotTable::new();
        for i in 0..6_u32 {
            for _ in 0..10 {
                hot.record_access(DagNodeId::new(i));
            }
        }
        // Budget of 3: only 3 nodes should be swept.
        let result = evict_cold_nodes_budgeted(b.arena(), &hot, 5, 3);
        assert_eq!(result.arena.len(), 3);
    }
}
