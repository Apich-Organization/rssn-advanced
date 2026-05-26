//! Flat open-addressed hash-consing deduplication map.
//!
//! Uses Robin Hood linear probing over a flat `Vec<Slot>` instead of
//! `HashMap<u64, Vec<DagNodeId>>`. This eliminates the inner Vec allocation
//! per bucket and gives sequential memory access during probing.

use super::arena::DagArena;
use super::metadata::NodeHash;
use super::node::{ChildList, DagNode, DagNodeId};
use super::symbol::SymbolKind;

/// Load factor threshold: rehash when len/cap exceeds this.
const MAX_LOAD: f64 = 0.7;
/// Initial capacity (must be power of two).
const INIT_CAP: usize = 64;

#[derive(Clone, Default)]
#[allow(dead_code)]
enum Slot {
    #[default]
    Empty,
    /// Reserved for future delete support; not currently constructed.
    Tombstone,
    Occupied {
        raw_hash: u64,
        id: DagNodeId,
    },
}

/// Flat Robin Hood open-addressed deduplication map.
///
/// `get_or_insert` does structural comparison only for slots whose `raw_hash`
/// matches — full-field equality on `kind`, `children`, `coefficient`, `flags`.
/// Probe distance is bounded to O(1) amortised at 70% load.
#[derive(Clone)]
pub struct DedupMap {
    slots: Vec<Slot>,
    len: usize,
    cap: usize,
}

impl Default for DedupMap {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for DedupMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DedupMap")
            .field("len", &self.len)
            .field("cap", &self.cap)
            .finish()
    }
}

impl DedupMap {
    /// Creates a new, empty deduplication map.
    #[must_use]
    pub fn new() -> Self {
        let cap = INIT_CAP;
        Self {
            slots: vec![Slot::Empty; cap],
            len: 0,
            cap,
        }
    }

    /// Computes structural hash for a variable node.
    #[must_use]
    pub fn hash_variable(kind: &SymbolKind) -> NodeHash {
        use std::hash::{Hash, Hasher};
        let mut h = rapidhash::fast::RapidHasher::default();
        kind.hash(&mut h);
        NodeHash(h.finish())
    }

    /// Computes structural hash for a constant node.
    ///
    /// Accepts the `SymbolKind` (expected to be `SymbolKind::Constant(val)`)
    /// and hashes the f64 bit pattern.
    #[must_use]
    pub fn hash_constant(kind: &SymbolKind) -> NodeHash {
        let val = if let SymbolKind::Constant(v) = kind {
            *v
        } else {
            0.0
        };
        use std::hash::Hasher;
        let mut h = rapidhash::fast::RapidHasher::default();
        h.write_u64(val.to_bits());
        NodeHash(h.finish())
    }

    /// Computes structural hash for an operator/function node with children.
    ///
    /// `coefficient` and `flags` are included because two nodes that differ
    /// only in coefficient (e.g. `2*x` vs `3*x` via metadata) or flags are
    /// structurally distinct.
    #[must_use]
    pub fn hash_operator(kind: &SymbolKind, children: &ChildList) -> NodeHash {
        Self::hash_operator_full(kind, children, 1.0, super::metadata::NodeFlags::EMPTY)
    }

    /// Like [`Self::hash_operator`] but includes `coefficient` and `flags`
    /// in the hash.
    #[must_use]
    pub fn hash_operator_full(
        kind: &SymbolKind,
        children: &ChildList,
        coefficient: f64,
        flags: super::metadata::NodeFlags,
    ) -> NodeHash {
        use std::hash::{Hash, Hasher};
        let mut h = rapidhash::fast::RapidHasher::default();
        kind.hash(&mut h);
        for &c in children.as_slice() {
            c.0.hash(&mut h);
        }
        h.write_u64(coefficient.to_bits());
        h.write_u8(flags.bits());
        NodeHash(h.finish())
    }

    /// Returns the slot index to start probing from.
    #[inline]
    const fn start_idx(&self, raw_hash: u64) -> usize {
        (raw_hash as usize) & (self.cap - 1)
    }

    /// Looks up or inserts a node. Returns the existing ID if a structural match
    /// is found; otherwise allocates a new node in `arena`, inserts it, and
    /// returns the new ID.
    pub fn get_or_insert(
        &mut self,
        arena: &mut DagArena,
        kind: SymbolKind,
        hash: NodeHash,
        children: ChildList,
        coefficient: f64,
        flags: super::metadata::NodeFlags,
    ) -> DagNodeId {
        // Rehash if at load limit.
        if self.len + 1 > (self.cap as f64 * MAX_LOAD) as usize {
            self.grow();
        }

        let raw = hash.0;
        let mut idx = self.start_idx(raw);

        loop {
            match &self.slots[idx] {
                Slot::Empty | Slot::Tombstone => {
                    // Insert here.
                    let meta = super::metadata::NodeMetadata {
                        hash,
                        coefficient,
                        flags,
                    };
                    let node = DagNode {
                        kind,
                        meta,
                        children,
                    };
                    let new_id = arena.alloc(node);
                    self.slots[idx] = Slot::Occupied {
                        raw_hash: raw,
                        id: new_id,
                    };
                    self.len += 1;
                    return new_id;
                }
                Slot::Occupied { raw_hash, id } => {
                    if *raw_hash == raw {
                        let found_id = *id;
                        if let Some(existing) = arena.get(found_id)
                            && existing.kind == kind
                            && existing.children == children
                            && existing.meta.coefficient.to_bits() == coefficient.to_bits()
                            && existing.meta.flags.without_canonical() == flags.without_canonical()
                        {
                            return found_id;
                        }
                    }
                    idx = (idx + 1) & (self.cap - 1);
                }
            }
        }
    }

    fn grow(&mut self) {
        let new_cap = self.cap * 2;
        let mut new_slots: Vec<Slot> = (0..new_cap).map(|_| Slot::Empty).collect();
        for slot in self.slots.drain(..) {
            if let Slot::Occupied { raw_hash, id } = slot {
                let mut idx = (raw_hash as usize) & (new_cap - 1);
                loop {
                    match &new_slots[idx] {
                        Slot::Empty => {
                            new_slots[idx] = Slot::Occupied { raw_hash, id };
                            break;
                        }
                        _ => {
                            idx = (idx + 1) & (new_cap - 1);
                        }
                    }
                }
            }
        }
        self.slots = new_slots;
        self.cap = new_cap;
    }

    /// Clears the deduplication map.
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            *slot = Slot::Empty;
        }
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::metadata::NodeFlags;
    use crate::dag::symbol::SymbolId;

    #[test]
    fn test_dedup_constant() {
        let mut arena = DagArena::new();
        let mut dedup = DedupMap::new();

        let val = 3.14;
        let kind = SymbolKind::Constant(val);
        let hash = DedupMap::hash_constant(&kind);
        let id1 = dedup.get_or_insert(
            &mut arena,
            kind,
            hash,
            ChildList::Empty,
            1.0,
            NodeFlags::EMPTY,
        );

        let id2 = dedup.get_or_insert(
            &mut arena,
            kind,
            hash,
            ChildList::Empty,
            1.0,
            NodeFlags::EMPTY,
        );

        assert_eq!(
            id1, id2,
            "Identical constants must resolve to the same node ID"
        );
        assert_eq!(
            arena.len(),
            1,
            "Only one node should be allocated in the arena"
        );
    }

    #[test]
    fn test_dedup_variable() {
        let mut arena = DagArena::new();
        let mut dedup = DedupMap::new();

        let kind = SymbolKind::Variable(SymbolId(0));
        let hash = DedupMap::hash_variable(&kind);

        let id1 = dedup.get_or_insert(
            &mut arena,
            kind,
            hash,
            ChildList::Empty,
            1.0,
            NodeFlags::EMPTY,
        );

        let id2 = dedup.get_or_insert(
            &mut arena,
            kind,
            hash,
            ChildList::Empty,
            1.0,
            NodeFlags::EMPTY,
        );

        assert_eq!(
            id1, id2,
            "Identical variables must resolve to the same node ID"
        );
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn flat_table_grows_and_deduplicates() {
        // Insert enough entries to trigger at least one grow() cycle (>70% of 64).
        let mut arena = DagArena::new();
        let mut dedup = DedupMap::new();

        // Insert 50 distinct variables to force a grow.
        let mut ids = Vec::new();
        for i in 0..50u32 {
            let kind = SymbolKind::Variable(SymbolId(i));
            let hash = DedupMap::hash_variable(&kind);
            let id = dedup.get_or_insert(
                &mut arena,
                kind,
                hash,
                ChildList::Empty,
                1.0,
                NodeFlags::EMPTY,
            );
            ids.push(id);
        }

        assert_eq!(
            arena.len(),
            50,
            "50 distinct variables should produce 50 nodes"
        );

        // Re-lookup each one — must get the same ID back.
        for i in 0..50u32 {
            let kind = SymbolKind::Variable(SymbolId(i));
            let hash = DedupMap::hash_variable(&kind);
            let id = dedup.get_or_insert(
                &mut arena,
                kind,
                hash,
                ChildList::Empty,
                1.0,
                NodeFlags::EMPTY,
            );
            assert_eq!(
                id, ids[i as usize],
                "dedup failed after grow for variable {i}"
            );
        }

        assert_eq!(
            arena.len(),
            50,
            "No new nodes should be allocated on re-lookup"
        );
    }

    #[test]
    fn clear_resets_flat_table() {
        let mut arena = DagArena::new();
        let mut dedup = DedupMap::new();

        let kind = SymbolKind::Constant(1.0);
        let hash = DedupMap::hash_constant(&kind);
        let _ = dedup.get_or_insert(
            &mut arena,
            kind,
            hash,
            ChildList::Empty,
            1.0,
            NodeFlags::EMPTY,
        );
        assert_eq!(dedup.len, 1);

        dedup.clear();
        assert_eq!(dedup.len, 0);

        // After clear, inserting the same node should produce a NEW arena slot
        // (the dedup map no longer knows about the old one).
        let id2 = dedup.get_or_insert(
            &mut arena,
            kind,
            hash,
            ChildList::Empty,
            1.0,
            NodeFlags::EMPTY,
        );
        // Arena now has 2 nodes (old + new), but dedup only knows about the new one.
        assert_eq!(arena.len(), 2);
        assert_eq!(dedup.len, 1);
        // The returned id must be the new (second) allocation.
        assert_eq!(id2, DagNodeId::new(1));
    }
}
