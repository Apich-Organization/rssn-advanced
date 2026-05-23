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
    /// Variadic children (heap spilled). Box<[T]> saves 8 bytes vs Vec<T>
    /// (16-byte fat pointer vs 24-byte triple), halving the enum's large-variant cost.
    Many(Box<[RelPtr<AstNode>]>),
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
            Self::Many(v) => v.len(), // Box<[T]> has .len()
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
            Self::Many(v) => v.as_ref(),
        }
    }
}

/// An AST node inside the stack-local projection buffer.
///
/// Highly compact and memory aligned:
/// - Stores its classification (`kind`) and constant value (`value`).
/// - Tracks the original global `DagNodeId` for metadata lookups.
/// - Connects to child nodes using relative pointers (`children`).
///
/// `value` is meaningful only when `kind == SymbolKind::Constant`; for all
/// other node kinds it is `0.0`. Using `f64` instead of `Option<f64>` saves
/// 8 bytes per node (8 vs 16 bytes) (`ast_review §1.1`).
#[derive(Debug, Clone, Encode, Decode)]
pub struct AstNode {
    /// The kind/classification of this symbol.
    pub kind: SymbolKind,
    /// Numeric constant value. Meaningful only when `kind == SymbolKind::Constant`;
    /// guaranteed to be `0.0` for all other kinds.
    pub value: f64,
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

/// Trait for visiting nodes of an [`AstProjection`] tree in post-order.
///
/// Implement this trait to process AST nodes without modifying the library.
/// Call [`AstProjection::visit_post`] to drive the traversal.
///
/// The visitor is called in **post-order** (children before parent), which is
/// the natural order for bottom-up transformations (e.g. constant folding,
/// type inference).
pub trait PostOrderVisitor {
    /// Called once for each node in post-order traversal.
    ///
    /// `node` is the current node; `node_idx` is its index in the flat buffer.
    /// Return `true` to continue traversal, `false` to halt early.
    fn visit(&mut self, node: &AstNode, node_idx: usize) -> bool;
}

impl AstProjection {
    /// Drives a post-order traversal of this projection, calling
    /// [`PostOrderVisitor::visit`] for each node.
    ///
    /// Traversal is iterative (stack-based), so arbitrarily deep trees do
    /// not overflow the OS stack. The visitor can halt early by returning
    /// `false` from [`PostOrderVisitor::visit`].
    pub fn visit_post<V: PostOrderVisitor>(&self, visitor: &mut V) {
        if self.nodes.is_empty() {
            return;
        }
        // Iterative post-order using an explicit stack of (node_idx, visited).
        let mut stack: Vec<(usize, bool)> = vec![(0, false)];
        while let Some((idx, visited)) = stack.pop() {
            let Some(node) = self.nodes.get(idx) else { continue };
            if visited {
                if !visitor.visit(node, idx) {
                    return;
                }
            } else {
                // Push self again with visited=true, then push children.
                stack.push((idx, true));
                for ptr in node.children.as_slice().iter().rev() {
                    if let Some(child_idx) = ptr.resolve(idx) {
                        if child_idx < self.nodes.len() {
                            stack.push((child_idx, false));
                        }
                    }
                }
            }
        }
    }
}
