//! Hash-consing and structural deduplication.
//!
//! Ensures that structurally identical sub-expressions share the same
//! `DagNodeId`. Uses `rapidhash`-based structural hashing to key a
//! deduplication map.

use std::collections::HashMap;

use super::arena::DagArena;
use super::metadata::NodeHash;
use super::node::{ChildList, DagNode, DagNodeId};
use super::symbol::SymbolKind;

/// Hash-consing map for structural deduplication of nodes in a `DagArena`.
#[derive(Debug, Clone, Default)]
pub struct DedupMap {
    map: HashMap<u64, Vec<DagNodeId>>,
}

impl DedupMap {
    /// Creates a new, empty deduplication map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Computes structural hash for a variable node.
    #[must_use]
    pub fn hash_variable(kind: &SymbolKind) -> NodeHash {
        let mut hasher = rapidhash::fast::RapidHasher::default();
        use std::hash::Hash;
        kind.hash(&mut hasher);
        NodeHash(std::hash::Hasher::finish(&hasher))
    }

    /// Computes structural hash for a constant node.
    #[must_use]
    pub fn hash_constant(val: f64) -> NodeHash {
        let mut hasher = rapidhash::fast::RapidHasher::default();
        // Use bits to hash f64 cleanly
        let bits = val.to_bits();
        use std::hash::Hasher;
        hasher.write_u64(bits);
        NodeHash(hasher.finish())
    }

    /// Computes structural hash for an operator/function node with children.
    ///
    /// `coefficient` and `flags` are included because two nodes that differ
    /// only in coefficient (e.g. `2*x` vs `3*x` via metadata) or flags are
    /// structurally distinct — omitting them caused guaranteed hash collisions
    /// that forced a full O(N) bucket scan on every dedup lookup.
    #[must_use]
    pub fn hash_operator(kind: &SymbolKind, children: &ChildList) -> NodeHash {
        Self::hash_operator_full(kind, children, 1.0, super::metadata::NodeFlags::EMPTY)
    }

    /// Like [`Self::hash_operator`] but includes `coefficient` and `flags`
    /// in the hash. Call this from builder methods that set non-default
    /// metadata on operator nodes.
    #[must_use]
    pub fn hash_operator_full(
        kind: &SymbolKind,
        children: &ChildList,
        coefficient: f64,
        flags: super::metadata::NodeFlags,
    ) -> NodeHash {
        let mut hasher = rapidhash::fast::RapidHasher::default();
        use std::hash::Hash;
        use std::hash::Hasher;
        kind.hash(&mut hasher);
        for &child in children.as_slice() {
            child.0.hash(&mut hasher);
        }
        hasher.write_u64(coefficient.to_bits());
        hasher.write_u8(flags.bits());
        NodeHash(hasher.finish())
    }

    /// Checks if a matching node already exists in the arena.
    /// If so, returns its ID. Otherwise, allocates it in the arena,
    /// inserts it into the deduplication map, and returns the new ID.
    pub fn get_or_insert(
        &mut self,
        arena: &mut DagArena,
        kind: SymbolKind,
        hash: NodeHash,
        children: ChildList,
        value: Option<f64>,
        coefficient: f64,
        flags: super::metadata::NodeFlags,
    ) -> DagNodeId {
        let bucket = self.map.entry(hash.0).or_default();
        
        // Linear scan in the hash bucket to handle collisions
        for &id in bucket.iter() {
            if let Some(existing) = arena.get(id) {
                if existing.kind == kind
                    && existing.children == children
                    && existing.value == value
                    && existing.meta.coefficient.to_bits() == coefficient.to_bits()
                    && existing.meta.flags == flags
                {
                    return id;
                }
            }
        }

        // Not found, construct and allocate
        let meta = super::metadata::NodeMetadata {
            hash,
            coefficient,
            arity: children.len() as u16,
            flags,
        };

        let node = DagNode {
            kind,
            meta,
            children,
            value,
        };

        let new_id = arena.alloc(node);
        bucket.push(new_id);
        new_id
    }

    /// Clears the deduplication map.
    pub fn clear(&mut self) {
        self.map.clear();
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
        let hash = DedupMap::hash_constant(val);
        let id1 = dedup.get_or_insert(
            &mut arena,
            SymbolKind::Constant,
            hash,
            ChildList::Empty,
            Some(val),
            1.0,
            NodeFlags::EMPTY,
        );

        let id2 = dedup.get_or_insert(
            &mut arena,
            SymbolKind::Constant,
            hash,
            ChildList::Empty,
            Some(val),
            1.0,
            NodeFlags::EMPTY,
        );

        assert_eq!(id1, id2, "Identical constants must resolve to the same node ID");
        assert_eq!(arena.len(), 1, "Only one node should be allocated in the arena");
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
            None,
            1.0,
            NodeFlags::EMPTY,
        );

        let id2 = dedup.get_or_insert(
            &mut arena,
            kind,
            hash,
            ChildList::Empty,
            None,
            1.0,
            NodeFlags::EMPTY,
        );

        assert_eq!(id1, id2, "Identical variables must resolve to the same node ID");
        assert_eq!(arena.len(), 1);
    }
}

