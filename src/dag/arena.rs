//! Typed arena allocator for DAG nodes.
//!
//! `DagArena` provides bump-allocated storage for `DagNode` values.
//! Nodes are stored in a contiguous vector and referenced via `DagNodeId`.

use super::node::{DagNode, DagNodeId};
use bincode_next::{Decode, Encode};

/// A memory arena for `DagNode` allocation.
///
/// It stores all nodes contiguously, enabling quick access by index,
/// Cache-friendly sequential traversals, and efficient bulk serialization.
#[derive(Debug, Clone, Encode, Decode)]
pub struct DagArena {
    nodes: Vec<DagNode>,
}

impl DagArena {
    /// Creates a new, empty arena.
    #[must_use]
    pub const fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Allocates a new node in the arena and returns its unique ID.
    pub fn alloc(&mut self, node: DagNode) -> DagNodeId {
        let index = self.nodes.len();
        self.nodes.push(node);
        #[allow(clippy::cast_possible_truncation)]
        DagNodeId(index as u32)
    }

    /// Retrieves a reference to the node with the given ID.
    ///
    /// Returns `None` if the ID is out of bounds or null.
    #[must_use]
    pub fn get(&self, id: DagNodeId) -> Option<&DagNode> {
        if id.is_none() {
            return None;
        }
        self.nodes.get(id.index())
    }

    /// Retrieves a mutable reference to the node with the given ID.
    ///
    /// Returns `None` if the ID is out of bounds or null.
    pub fn get_mut(&mut self, id: DagNodeId) -> Option<&mut DagNode> {
        if id.is_none() {
            return None;
        }
        self.nodes.get_mut(id.index())
    }

    /// Returns the total number of nodes allocated in this arena.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if no nodes have been allocated in this arena.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clears all nodes from the arena.
    pub fn clear(&mut self) {
        self.nodes.clear();
    }
}

impl Default for DagArena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::metadata::{NodeHash, NodeMetadata};

    #[test]
    fn arena_alloc_and_retrieve() {
        let mut arena = DagArena::new();
        assert!(arena.is_empty());

        let meta = NodeMetadata::leaf(NodeHash(42));
        let node = DagNode::constant(1.0, meta);
        let id = arena.alloc(node);

        assert_eq!(arena.len(), 1);
        assert!(!arena.is_empty());

        let retrieved = arena.get(id).unwrap();
        assert!(
            matches!(retrieved.kind, crate::dag::symbol::SymbolKind::Constant(v) if (v - 1.0).abs() < f64::EPSILON)
        );
        assert_eq!(retrieved.meta.hash, NodeHash(42));

        let retrieved_mut = arena.get_mut(id).unwrap();
        retrieved_mut.meta.hash = NodeHash(43);

        assert_eq!(arena.get(id).unwrap().meta.hash, NodeHash(43));
    }
}
