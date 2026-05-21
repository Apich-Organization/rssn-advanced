//! DAG ↔ AST bidirectional conversion routines.
//!
//! ## Iterative traversal
//!
//! The original implementation was recursive; for industrial-scale
//! expressions (the project's whole point) recursive walks blow the
//! stack at a few thousand-deep chains. `ast_review §2`,
//! `heuristic_review §2`, `jit_review §2`, `parallel_review §2`,
//! `parser_review §2` all flagged it.
//!
//! Both [`dag_to_ast`] and [`ast_to_dag`] now drive an explicit work
//! stack. Stack frames carry a cursor over the node's children so we
//! revisit each parent exactly twice — once to enqueue its children
//! and once to back-patch the relative-pointer child list. Depth is
//! limited by heap, not by OS stack.

use super::pointer::RelPtr;
use super::projection::{AstChildList, AstNode, AstProjection};
use crate::dag::arena::DagArena;
use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use crate::dag::symbol::{OpKind, SymbolKind};

/// A frame on the explicit traversal stack.
///
/// `cursor` advances from `0` to `arity`; when they're equal the frame
/// has finished and we back-patch.
struct DagFrame {
    dag_id: DagNodeId,
    ast_idx: usize,
    child_ast_indices: Vec<usize>,
    cursor: usize,
    arity: usize,
}

/// Converts a DAG subgraph starting at `root` into a stack-local
/// `AstProjection` without recursion.
///
/// Root lives at index 0 of the resulting projection.
///
/// If `root` (or any reachable node) references a missing arena slot
/// (i.e. corrupt input), the conversion short-circuits silently
/// rather than panicking — returns whatever projection it has built
/// so far. The previous `expect()`-based version would have aborted.
#[must_use]
pub fn dag_to_ast(arena: &DagArena, root: DagNodeId) -> AstProjection {
    let mut projection = AstProjection::new();
    if root.is_none() {
        return projection;
    }

    let mut stack: Vec<DagFrame> = Vec::with_capacity(64);
    if !push_dag_frame(arena, &mut projection, &mut stack, root) {
        return projection;
    }

    while let Some(top) = stack.last_mut() {
        // Pull next child id (if any) without holding a borrow across
        // the recursive push below.
        let next_child: Option<DagNodeId> = if top.cursor < top.arity {
            arena
                .get(top.dag_id)
                .and_then(|n| n.children.as_slice().get(top.cursor).copied())
                .inspect(|_| {
                    top.cursor += 1;
                })
        } else {
            None
        };

        if let Some(child) = next_child {
            // If the child push fails (dangling id), just skip it; the
            // parent will end up with one fewer child but the
            // projection stays internally consistent.
            let _ = push_dag_frame(arena, &mut projection, &mut stack, child);
        } else {
            // All children processed for this frame.
            let Some(frame) = stack.pop() else { break };
            backpatch_children(&mut projection, &frame);
            if let Some(parent) = stack.last_mut() {
                parent.child_ast_indices.push(frame.ast_idx);
            }
        }
    }

    projection
}

/// Returns `false` if `dag_id` doesn't resolve in `arena`. The caller
/// is expected to treat that as "skip this branch" rather than abort.
fn push_dag_frame(
    arena: &DagArena,
    projection: &mut AstProjection,
    stack: &mut Vec<DagFrame>,
    dag_id: DagNodeId,
) -> bool {
    let Some(node) = arena.get(dag_id) else {
        return false;
    };
    let ast_idx = projection.nodes.len();
    projection.nodes.push(AstNode {
        kind: node.kind,
        value: node.value,
        dag_id,
        children: AstChildList::Empty,
    });
    stack.push(DagFrame {
        dag_id,
        ast_idx,
        child_ast_indices: Vec::new(),
        cursor: 0,
        arity: node.children.len(),
    });
    true
}

fn backpatch_children(projection: &mut AstProjection, frame: &DagFrame) {
    let ptrs: Vec<RelPtr<AstNode>> = frame
        .child_ast_indices
        .iter()
        .map(|&child_idx| RelPtr::from_indices(frame.ast_idx, child_idx))
        .collect();
    let slot = &mut projection.nodes[frame.ast_idx];
    slot.children = match ptrs.len() {
        0 => AstChildList::Empty,
        1 => AstChildList::One(ptrs[0]),
        2 => AstChildList::Two([ptrs[0], ptrs[1]]),
        3 => AstChildList::Three([ptrs[0], ptrs[1], ptrs[2]]),
        4 => AstChildList::Four([ptrs[0], ptrs[1], ptrs[2], ptrs[3]]),
        _ => AstChildList::Many(ptrs),
    };
}

// =========================================================================
// AST → DAG
// =========================================================================

struct AstFrame {
    idx: usize,
    child_dag_ids: Vec<DagNodeId>,
    cursor: usize,
    arity: usize,
}

/// Merges an `AstProjection` back into the global DAG, re-deduplicating
/// every subexpression.
///
/// Returns the new `DagNodeId` of the merged root node, or
/// `DagNodeId::NONE` if the projection is empty / malformed
/// (e.g. relative-pointer underflow, child slot out of range).
/// The previous version panicked in these cases; Phase 7 made the
/// path panic-free.
pub fn ast_to_dag(ast: &AstProjection, builder: &mut DagBuilder) -> DagNodeId {
    let Some(root_node) = ast.nodes.first() else {
        return DagNodeId::NONE;
    };

    let mut stack: Vec<AstFrame> = Vec::with_capacity(64);
    stack.push(AstFrame {
        idx: 0,
        child_dag_ids: Vec::new(),
        cursor: 0,
        arity: root_node.children.len(),
    });

    let mut root_id = DagNodeId::NONE;

    while let Some(top) = stack.last_mut() {
        // Pull the next child slot, if any. If a relative pointer
        // resolves out of range, skip it; the parent ends up with
        // one fewer child.
        let next_child_idx: Option<usize> = if top.cursor < top.arity {
            let Some(node) = ast.nodes.get(top.idx) else {
                let _ = stack.pop();
                continue;
            };
            let cursor = top.cursor;
            top.cursor += 1;
            node.children
                .as_slice()
                .get(cursor)
                .and_then(|p| p.resolve(top.idx))
                .filter(|idx| *idx < ast.nodes.len())
        } else {
            None
        };

        if let Some(child_idx) = next_child_idx {
            let arity = ast
                .nodes
                .get(child_idx)
                .map(|n| n.children.len())
                .unwrap_or(0);
            stack.push(AstFrame {
                idx: child_idx,
                child_dag_ids: Vec::new(),
                cursor: 0,
                arity,
            });
        } else if top.cursor >= top.arity {
            let Some(frame) = stack.pop() else { break };
            let dag_id = build_dag_node(ast, builder, frame.idx, &frame.child_dag_ids);
            if let Some(parent) = stack.last_mut() {
                parent.child_dag_ids.push(dag_id);
            } else {
                root_id = dag_id;
            }
        }
    }

    root_id
}

fn build_dag_node(
    ast: &AstProjection,
    builder: &mut DagBuilder,
    idx: usize,
    child_ids: &[DagNodeId],
) -> DagNodeId {
    let Some(ast_node) = ast.nodes.get(idx) else {
        return DagNodeId::NONE;
    };
    match ast_node.kind {
        SymbolKind::Constant => builder.constant(ast_node.value.unwrap_or(0.0)),
        SymbolKind::Variable(sym_id) => {
            // Round-trip the variable through the builder's registry; if
            // the same SymbolId was already interned the name lookup
            // succeeds, otherwise we fall back to a synthetic name.
            let name = builder
                .registry()
                .name(sym_id)
                .unwrap_or("x")
                .to_owned();
            builder.variable(&name)
        }
        // Arity mismatches signal a malformed projection. Previously
        // we used `assert_eq!` which panicked; Phase 7 instead returns
        // `DagNodeId::NONE` so callers can detect and recover.
        SymbolKind::Operator(op) => match op {
            OpKind::Add if child_ids.len() == 2 => builder.add(child_ids[0], child_ids[1]),
            OpKind::Sub if child_ids.len() == 2 => builder.sub(child_ids[0], child_ids[1]),
            OpKind::Mul if child_ids.len() == 2 => builder.mul(child_ids[0], child_ids[1]),
            OpKind::Div if child_ids.len() == 2 => builder.div(child_ids[0], child_ids[1]),
            OpKind::Pow if child_ids.len() == 2 => builder.pow(child_ids[0], child_ids[1]),
            OpKind::Neg if child_ids.len() == 1 => builder.neg(child_ids[0]),
            _ => DagNodeId::NONE,
        },
        SymbolKind::Function(_) => {
            builder.operator(ast_node.kind, child_ids, crate::dag::metadata::NodeFlags::EMPTY)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bidirectional_conversion() {
        let mut builder = DagBuilder::new();

        // Build: (a + b) * 2.5
        let a = builder.variable("a");
        let b = builder.variable("b");
        let sum = builder.add(a, b);
        let coeff = builder.constant(2.5);
        let root = builder.mul(sum, coeff);

        let ast = dag_to_ast(builder.arena(), root);
        assert_eq!(ast.len(), 5);
        let root_ast = ast.root().expect("root present");
        assert_eq!(root_ast.kind, SymbolKind::Operator(OpKind::Mul));

        let mut new_builder = DagBuilder::new();
        new_builder.variable("a");
        new_builder.variable("b");
        let new_root = ast_to_dag(&ast, &mut new_builder);

        let new_node = new_builder.arena().get(new_root).expect("new root present");
        assert_eq!(new_node.kind, SymbolKind::Operator(OpKind::Mul));
        assert_eq!(new_builder.arena().len(), 5);
    }

    #[test]
    fn deep_chain_does_not_overflow_stack() {
        // Build a 5_000-deep left-associative chain `((((x+x)+x)+x)+x)...`.
        // The recursive convert would have overflowed at a few thousand;
        // the iterative one must finish in linear time.
        let mut builder = DagBuilder::new();
        let x = builder.variable("x");
        let mut acc = x;
        for _ in 0..5_000 {
            acc = builder.add(acc, x);
        }

        let ast = dag_to_ast(builder.arena(), acc);
        // Each `add` introduces one new AST node + one new leaf `x`. The
        // very first `x` is also a leaf, so we get 5000 adds + 5001
        // leaves = 10_001 nodes.
        assert_eq!(ast.len(), 10_001);

        let mut rebuild = DagBuilder::new();
        rebuild.variable("x");
        let new_root = ast_to_dag(&ast, &mut rebuild);
        assert!(!new_root.is_none());
    }
}
