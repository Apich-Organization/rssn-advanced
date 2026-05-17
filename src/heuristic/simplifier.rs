//! Approximate simplification for graceful degradation.
//!
//! When exact simplification detects symbol explosion, this module
//! applies lossy but fast rewriting to keep computation tractable.

use crate::dag::arena::DagArena;
use crate::dag::node::DagNodeId;

/// Performs lossy approximate simplification to prevent symbol explosion.
///
/// When exact simplification is too expensive, this function prunes or folds
/// sub-trees based on the `aggressiveness` parameter (from 0.0 to 1.0).
pub fn approximate_simplify(
    arena: &mut DagArena,
    root: DagNodeId,
    aggressiveness: f64,
) -> DagNodeId {
    if aggressiveness < 0.1 {
        return root;
    }

    approximate_simplify_rec(arena, root, 0, aggressiveness)
}

fn approximate_simplify_rec(
    arena: &mut DagArena,
    root: DagNodeId,
    depth: usize,
    aggressiveness: f64,
) -> DagNodeId {
    // If the tree depth is excessive, gracefully degrade by folding deep subtrees to a constant 1.0
    if depth > 5 && aggressiveness > 0.5 {
        let meta = crate::dag::metadata::NodeMetadata::leaf(crate::dag::metadata::NodeHash(999));
        let node = crate::dag::node::DagNode::constant(1.0, meta);
        return arena.alloc(node);
    }

    // Extract children, kind, and meta to release borrow on arena before recursing
    let (children, kind, meta) = if let Some(node) = arena.get(root) {
        if node.is_leaf() {
            return root;
        }
        (node.children.iter().collect::<Vec<_>>(), node.kind, node.meta.clone())
    } else {
        return root;
    };

    let mut children_changed = false;
    let mut new_children = Vec::new();

    for child_id in children {
        let simplified = approximate_simplify_rec(arena, child_id, depth + 1, aggressiveness);
        if simplified != child_id {
            children_changed = true;
        }
        new_children.push(simplified);
    }

    if children_changed {
        let child_list = crate::dag::node::ChildList::from_slice(&new_children);
        let new_node = crate::dag::node::DagNode::operator(kind, meta, child_list);
        return arena.alloc(new_node);
    }

    root
}
