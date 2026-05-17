//! DAG ↔ AST bidirectional conversion routines.

use super::pointer::RelPtr;
use super::projection::{AstChildList, AstNode, AstProjection};
use crate::dag::arena::DagArena;
use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use crate::dag::symbol::{OpKind, SymbolKind};

/// Converts a DAG subgraph starting at `root` into a stack-local `AstProjection`.
///
/// The resulting projection will have the root node positioned at index 0.
#[must_use]
pub fn dag_to_ast(arena: &DagArena, root: DagNodeId) -> AstProjection {
    let mut projection = AstProjection::new();
    if !root.is_none() {
        convert_dag_node(arena, root, &mut projection.nodes);
    }
    projection
}

fn convert_dag_node(arena: &DagArena, id: DagNodeId, nodes: &mut Vec<AstNode>) -> usize {
    let node = arena.get(id).expect("Invalid DAG node ID in conversion");
    let current_idx = nodes.len();

    // Push placeholder node to reserve the index position
    nodes.push(AstNode {
        kind: node.kind,
        value: node.value,
        dag_id: id,
        children: AstChildList::Empty,
    });

    // Recursively convert children and build relative pointers
    let mut child_ptrs = Vec::new();
    for &child_id in node.children.as_slice() {
        let child_idx = convert_dag_node(arena, child_id, nodes);
        child_ptrs.push(RelPtr::from_indices(current_idx, child_idx));
    }

    // Update node with correct relative child list
    nodes[current_idx].children = match child_ptrs.len() {
        0 => AstChildList::Empty,
        1 => AstChildList::One(child_ptrs[0]),
        2 => AstChildList::Two([child_ptrs[0], child_ptrs[1]]),
        3 => AstChildList::Three([child_ptrs[0], child_ptrs[1], child_ptrs[2]]),
        4 => AstChildList::Four([child_ptrs[0], child_ptrs[1], child_ptrs[2], child_ptrs[3]]),
        _ => AstChildList::Many(child_ptrs),
    };

    current_idx
}

/// Merges an `AstProjection` back into the global DAG, re-deduplicating all subexpressions.
///
/// Returns the new `DagNodeId` of the merged root node.
pub fn ast_to_dag(ast: &AstProjection, builder: &mut DagBuilder) -> DagNodeId {
    if ast.is_empty() {
        return DagNodeId::NONE;
    }
    convert_ast_node(ast, 0, builder)
}

fn convert_ast_node(projection: &AstProjection, idx: usize, builder: &mut DagBuilder) -> DagNodeId {
    let ast_node = &projection.nodes[idx];

    // Traverse children postorder
    let mut child_ids = Vec::new();
    for &child_ptr in ast_node.children.as_slice() {
        if let Some(child_idx) = child_ptr.resolve(idx) {
            let child_dag_id = convert_ast_node(projection, child_idx, builder);
            child_ids.push(child_dag_id);
        }
    }

    // Build the node in the DAG arena with dedup
    match ast_node.kind {
        SymbolKind::Constant => builder.constant(ast_node.value.unwrap_or(0.0)),
        SymbolKind::Variable(sym_id) => {
            let name = builder
                .registry()
                .name(sym_id)
                .unwrap_or("x")
                .to_owned();
            builder.variable(&name)
        }
        SymbolKind::Operator(op) => match op {
            OpKind::Add => {
                assert_eq!(child_ids.len(), 2, "Addition operator requires 2 children");
                builder.add(child_ids[0], child_ids[1])
            }
            OpKind::Sub => {
                assert_eq!(child_ids.len(), 2, "Subtraction operator requires 2 children");
                builder.sub(child_ids[0], child_ids[1])
            }
            OpKind::Mul => {
                assert_eq!(child_ids.len(), 2, "Multiplication operator requires 2 children");
                builder.mul(child_ids[0], child_ids[1])
            }
            OpKind::Div => {
                assert_eq!(child_ids.len(), 2, "Division operator requires 2 children");
                builder.div(child_ids[0], child_ids[1])
            }
            OpKind::Pow => {
                assert_eq!(child_ids.len(), 2, "Power operator requires 2 children");
                builder.pow(child_ids[0], child_ids[1])
            }
            OpKind::Neg => {
                assert_eq!(child_ids.len(), 1, "Negation operator requires 1 child");
                builder.neg(child_ids[0])
            }
        },
        SymbolKind::Function(_) => {
            builder.operator(ast_node.kind, &child_ids, crate::dag::metadata::NodeFlags::EMPTY)
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

        // Convert to AST Projection
        let ast = dag_to_ast(builder.arena(), root);
        assert_eq!(ast.len(), 5);

        // Verify root of projection matches expected
        let root_ast = ast.root().unwrap();
        assert_eq!(root_ast.kind, SymbolKind::Operator(OpKind::Mul));

        // Reconvert to DAG in a fresh builder
        let mut new_builder = DagBuilder::new();
        // Seed the registry with the same names to keep SymbolIds in line
        new_builder.variable("a");
        new_builder.variable("b");

        let new_root = ast_to_dag(&ast, &mut new_builder);

        // Verify identical structure reconverted
        let new_node = new_builder.arena().get(new_root).unwrap();
        assert_eq!(new_node.kind, SymbolKind::Operator(OpKind::Mul));
        assert_eq!(new_builder.arena().len(), 5);
    }
}
