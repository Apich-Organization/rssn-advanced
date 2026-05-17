//! AST projection tree built from DAG subgraphs.
//!
//! `AstProjection` holds a flat, cache-friendly array of `AstNode` elements.
//! Instead of using recursive boxed structures, child nodes reference each other
//! using relative pointers (`RelPtr<AstNode>`).

use bincode_next::{Decode, Encode};

use super::pointer::RelPtr;
use crate::dag::node::DagNodeId;
use crate::dag::symbol::SymbolKind;

/// Compact representation of children inside a stack-local AST node.
///
/// Children are represented as relative pointers to other positions
/// within the contiguous AST projection buffer.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Encode, Decode)]
pub enum AstChildList {
    /// Zero children (leaf node).
    Empty,
    /// One child (unary operator).
    One(RelPtr<AstNode>),
    /// Two children (binary operator — most common).
    Two([RelPtr<AstNode>; 2]),
    /// Three children.
    Three([RelPtr<AstNode>; 3]),
    /// Four children.
    Four([RelPtr<AstNode>; 4]),
    /// Variadic children (heap spilled).
    Many(Vec<RelPtr<AstNode>>),
}

impl AstChildList {
    /// Returns the number of child pointers.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::One(_) => 1,
            Self::Two(_) => 2,
            Self::Three(_) => 3,
            Self::Four(_) => 4,
            Self::Many(v) => v.len(),
        }
    }

    /// Returns `true` if this list has no children.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Returns the children as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[RelPtr<AstNode>] {
        match self {
            Self::Empty => &[],
            Self::One(ptr) => std::slice::from_ref(ptr),
            Self::Two(arr) => arr,
            Self::Three(arr) => arr,
            Self::Four(arr) => arr,
            Self::Many(v) => v,
        }
    }
}

/// An AST node inside the stack-local projection buffer.
///
/// Highly compact and memory aligned:
/// - Stores its classification (`kind`) and constant value (`value`).
/// - Tracks the original global `DagNodeId` for metadata lookups.
/// - Connects to child nodes using relative pointers (`children`).
#[derive(Debug, Clone, Encode, Decode)]
pub struct AstNode {
    /// The kind/classification of this symbol.
    pub kind: SymbolKind,
    /// Numeric value if this is a constant node.
    pub value: Option<f64>,
    /// Index reference back to the global DAG node.
    pub dag_id: DagNodeId,
    /// Relative pointer offsets to child nodes within the buffer.
    pub children: AstChildList,
}

/// A contiguous array of AST nodes representing a stack-local projection tree.
///
/// Because it uses relative pointers, the entire tree can be cloned,
/// serialized, or iterated over in linear memory with zero pointer patching.
#[derive(Debug, Clone, Encode, Decode, Default)]
pub struct AstProjection {
    /// Contiguous buffer of nodes. Index 0 is typically the root of the projection.
    pub nodes: Vec<AstNode>,
}

impl AstProjection {
    /// Creates a new, empty AST projection buffer.
    #[must_use]
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Accesses the root node of the AST projection.
    #[must_use]
    pub fn root(&self) -> Option<&AstNode> {
        self.nodes.first()
    }

    /// Resolves a relative pointer starting from a given source node index.
    #[must_use]
    pub fn resolve(&self, source_idx: usize, ptr: RelPtr<AstNode>) -> Option<&AstNode> {
        ptr.resolve(source_idx).and_then(|idx| self.nodes.get(idx))
    }

    /// Clears the projection buffer.
    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    /// Returns the total number of nodes in the projection.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the projection is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}
