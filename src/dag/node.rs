//! DAG node representation.
//!
//! A `DagNode` is a vertex in the global expression graph. It owns its
//! metadata inline and stores children as compact arena indices.

use core::fmt;

use bincode_next::{Decode, Encode};

use super::metadata::NodeMetadata;
use super::symbol::SymbolKind;

/// An index into the `DagArena`, identifying a specific `DagNode`.
///
/// This is a lightweight handle (4 bytes) that can be freely copied
/// and compared. It is only valid within the arena that created it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Encode, Decode)]
pub struct DagNodeId(pub(crate) u32);

impl DagNodeId {
    /// Creates a new `DagNodeId` from a raw u32 index.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Returns the raw u32 index of this node ID.
    #[must_use]
    pub const fn value(self) -> u32 {
        self.0
    }

    /// A sentinel value representing "no node" / null.
    pub const NONE: Self = Self(u32::MAX);

    /// Returns the raw index value.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Returns `true` if this is the null sentinel.
    #[must_use]
    pub const fn is_none(self) -> bool {
        self.0 == u32::MAX
    }
}

impl fmt::Debug for DagNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            f.write_str("DagNodeId(NONE)")
        } else {
            write!(f, "DagNodeId({})", self.0)
        }
    }
}

impl fmt::Display for DagNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            f.write_str("∅")
        } else {
            write!(f, "#{}", self.0)
        }
    }
}

/// Maximum number of children stored inline (on the stack) before
/// spilling to the heap. 4 covers the vast majority of nodes
/// (binary ops = 2, ternary = 3, quaternary = 4).
const INLINE_CHILDREN: usize = 4;

/// A compact child list that stores up to [`INLINE_CHILDREN`] IDs inline.
///
/// For nodes with more children (e.g. variadic `+` or `*` chains),
/// the list spills to a heap-allocated `Vec`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Encode, Decode)]
pub enum ChildList {
    /// Zero children (leaf node).
    Empty,
    /// One child (unary operator).
    One([DagNodeId; 1]),
    /// Two children (binary operator — the most common case).
    Two([DagNodeId; 2]),
    /// Three children.
    Three([DagNodeId; 3]),
    /// Four children (inline limit).
    Four([DagNodeId; 4]),
    /// More than [`INLINE_CHILDREN`] children (heap-allocated).
    Many(Vec<DagNodeId>),
}

impl ChildList {
    /// Creates a child list from a slice of node IDs.
    #[must_use]
    pub fn from_slice(ids: &[DagNodeId]) -> Self {
        match ids.len() {
            0 => Self::Empty,
            1 => Self::One([ids[0]]),
            2 => Self::Two([ids[0], ids[1]]),
            3 => Self::Three([ids[0], ids[1], ids[2]]),
            4 => Self::Four([ids[0], ids[1], ids[2], ids[3]]),
            _ => Self::Many(ids.to_vec()),
        }
    }

    /// Returns the number of children.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::One(_) => 1,
            Self::Two(_) => 2,
            Self::Three(_) => 3,
            Self::Four(_) => INLINE_CHILDREN,
            Self::Many(v) => v.len(),
        }
    }

    /// Returns `true` if there are no children.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Returns the children as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[DagNodeId] {
        match self {
            Self::Empty => &[],
            Self::One(a) => a,
            Self::Two(a) => a,
            Self::Three(a) => a,
            Self::Four(a) => a,
            Self::Many(v) => v,
        }
    }

    /// Returns an iterator over the children.
    pub fn iter(&self) -> impl Iterator<Item = DagNodeId> + '_ {
        self.as_slice().iter().copied()
    }
}

/// A node in the global DAG.
///
/// Each `DagNode` represents a unique sub-expression. It contains:
/// - What **kind** of symbol it is (variable, constant, operator, function).
/// - **Metadata** (hash, coefficient, arity, flags).
/// - **Children** (references to other DAG nodes via `DagNodeId`).
///
/// For constant nodes, the value is stored inline in `kind` as
/// `SymbolKind::Constant(val)` — no separate `Option<f64>` field.
#[derive(Debug, Clone, Encode, Decode)]
pub struct DagNode {
    /// The classification of this node.
    pub kind: SymbolKind,
    /// Algebraic metadata (hash, coefficient, flags).
    pub meta: NodeMetadata,
    /// References to child nodes in the arena.
    pub children: ChildList,
}

impl DagNode {
    /// Creates a variable leaf node.
    #[must_use]
    pub fn variable(kind: SymbolKind, meta: NodeMetadata) -> Self {
        Self {
            kind,
            meta,
            children: ChildList::Empty,
        }
    }

    /// Creates a numeric constant leaf node.
    #[must_use]
    pub fn constant(val: f64, meta: NodeMetadata) -> Self {
        Self {
            kind: SymbolKind::Constant(val),
            meta,
            children: ChildList::Empty,
        }
    }

    /// Creates an operator node with the given children.
    #[must_use]
    pub fn operator(kind: SymbolKind, meta: NodeMetadata, children: ChildList) -> Self {
        Self {
            kind,
            meta,
            children,
        }
    }

    /// Returns `true` if this is a leaf node (no children).
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::metadata::{NodeFlags, NodeHash};
    use crate::dag::symbol::{OpKind, SymbolId};

    #[test]
    fn child_list_inline() {
        let ids = [DagNodeId(0), DagNodeId(1)];
        let cl = ChildList::from_slice(&ids);
        assert_eq!(cl.len(), 2);
        assert_eq!(cl.as_slice(), &ids);
    }

    #[test]
    fn child_list_many() {
        let ids: Vec<DagNodeId> = (0..10).map(DagNodeId).collect();
        let cl = ChildList::from_slice(&ids);
        assert_eq!(cl.len(), 10);
        assert_eq!(cl.as_slice(), &ids);
    }

    #[test]
    fn variable_node() {
        let meta = NodeMetadata::leaf(NodeHash(100));
        let node = DagNode::variable(SymbolKind::Variable(SymbolId(0)), meta);
        assert!(node.is_leaf());
        assert!(!matches!(node.kind, SymbolKind::Constant(_)));
    }

    #[test]
    fn constant_node() {
        let meta = NodeMetadata::leaf(NodeHash(200));
        let node = DagNode::constant(3.14, meta);
        assert!(node.is_leaf());
        assert!(matches!(node.kind, SymbolKind::Constant(v) if (v - 3.14).abs() < f64::EPSILON));
    }

    #[test]
    fn operator_node() {
        let meta = NodeMetadata::operator(NodeHash(300), NodeFlags::commutative_associative());
        let children = ChildList::from_slice(&[DagNodeId(0), DagNodeId(1)]);
        let node = DagNode::operator(SymbolKind::Operator(OpKind::Add), meta, children);
        assert!(!node.is_leaf());
        assert_eq!(node.children.len(), 2);
        assert!(node.meta.flags.is_commutative());
    }
}
