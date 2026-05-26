//! Commutativity-based expression splitting.
//!
//! Walks the DAG to collect commutative chains of additions,
//! then partitions them into N independent chunks for parallel evaluation.

use super::permission::SymbolPermissions;
use crate::dag::arena::DagArena;
use crate::dag::node::DagNodeId;
use crate::dag::symbol::{OpKind, SymbolKind};

/// Collects all leaves of a commutative addition chain.
///
/// Recursively flattens addition nodes where commutativity is supported.
pub fn collect_commutative_add_leaves(
    arena: &DagArena,
    node_id: DagNodeId,
    permissions: &SymbolPermissions,
    leaves: &mut Vec<DagNodeId>,
) {
    if let Some(node) = arena.get(node_id) {
        if node.kind == SymbolKind::Operator(OpKind::Add) {
            // Check if all variables in this node's children are commutative
            let is_comm = node.children.iter().all(|child_id| {
                if let Some(child_node) = arena.get(child_id) {
                    match child_node.kind {
                        SymbolKind::Variable(sym_id) => permissions.is_commutative(sym_id),
                        _ => true,
                    }
                } else {
                    true
                }
            });

            if is_comm {
                for child_id in node.children.iter() {
                    collect_commutative_add_leaves(arena, child_id, permissions, leaves);
                }
                return;
            }
        }
        leaves.push(node_id);
    }
}

/// Splits a commutative addition tree into up to `num_chunks` independent sub-lists of leaves.
#[must_use]
pub fn split_commutative_tree(
    arena: &DagArena,
    root: DagNodeId,
    permissions: &SymbolPermissions,
    num_chunks: usize,
) -> Vec<Vec<DagNodeId>> {
    let mut leaves = Vec::new();
    collect_commutative_add_leaves(arena, root, permissions, &mut leaves);

    if leaves.is_empty() {
        return Vec::new();
    }

    let n = leaves.len();
    let num_chunks = num_chunks.clamp(1, n);
    let chunk_size = n.div_ceil(num_chunks);

    leaves
        .chunks(chunk_size)
        .map(<[crate::dag::node::DagNodeId]>::to_vec)
        .collect()
}
