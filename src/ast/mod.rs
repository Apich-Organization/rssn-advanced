//! Local AST projection for stack-local computation.
//!
//! # Architecture
//!
//! The AST projection maps a subgraph of the global DAG into a flat,
//! stack-allocated tree using relative pointers. This gives algorithms
//! a familiar tree interface while metadata remains in the DAG arena.
//!
//! - `projection` — The `AstProjection` type and its buffer management.
//! - `pointer` — `RelPtr<i32>` / `RelPtr<i64>` relative pointer types.
//! - `convert` — DAG ↔ AST conversion routines.

pub mod convert;
pub mod pointer;
pub mod projection;

pub use projection::{AstChildList, AstNode, AstProjection, PostOrderVisitor};

/// Visitor for traversing an [`AstProjection`] tree node-by-node.
///
/// Each method corresponds to a `SymbolKind` variant. Default implementations
/// are no-ops so callers only implement the kinds they care about.
pub trait AstVisitor {
    /// Called for each variable node.
    fn visit_variable(
        &mut self,
        _idx: usize,
        _sym_id: crate::dag::symbol::SymbolId,
        _dag_id: crate::dag::node::DagNodeId,
    ) {
    }
    /// Called for each constant node.
    fn visit_constant(&mut self, _idx: usize, _val: f64, _dag_id: crate::dag::node::DagNodeId) {}
    /// Called for each operator node.
    fn visit_operator(
        &mut self,
        _idx: usize,
        _op: crate::dag::symbol::OpKind,
        _children: &AstChildList,
        _dag_id: crate::dag::node::DagNodeId,
    ) {
    }
    /// Called for each function-call node.
    fn visit_function(
        &mut self,
        _idx: usize,
        _fn_id: crate::dag::symbol::FnId,
        _children: &AstChildList,
        _dag_id: crate::dag::node::DagNodeId,
    ) {
    }
}

/// Iterates the flat `AstProjection` buffer in storage order, calling the
/// appropriate `visit_*` method on `visitor` for each node.
pub fn walk_projection<V: AstVisitor>(proj: &AstProjection, visitor: &mut V) {
    for (idx, node) in proj.nodes.iter().enumerate() {
        match node.kind {
            crate::dag::symbol::SymbolKind::Variable(sym_id) => {
                visitor.visit_variable(idx, sym_id, node.dag_id);
            }
            crate::dag::symbol::SymbolKind::Constant(val) => {
                visitor.visit_constant(idx, val, node.dag_id);
            }
            crate::dag::symbol::SymbolKind::Operator(op) => {
                visitor.visit_operator(idx, op, &node.children, node.dag_id);
            }
            crate::dag::symbol::SymbolKind::Function(fn_id) => {
                visitor.visit_function(idx, fn_id, &node.children, node.dag_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::convert::dag_to_ast;
    use crate::dag::builder::DagBuilder;

    #[test]
    fn walk_counts_node_kinds() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let c = b.constant(2.0);
        let expr = b.mul(x, c);
        let proj = dag_to_ast(b.arena(), expr);

        struct Counter {
            vars: usize,
            consts: usize,
            ops: usize,
        }
        impl AstVisitor for Counter {
            fn visit_variable(
                &mut self,
                _idx: usize,
                _sym_id: crate::dag::symbol::SymbolId,
                _dag_id: crate::dag::node::DagNodeId,
            ) {
                self.vars += 1;
            }
            fn visit_constant(
                &mut self,
                _idx: usize,
                _val: f64,
                _dag_id: crate::dag::node::DagNodeId,
            ) {
                self.consts += 1;
            }
            fn visit_operator(
                &mut self,
                _idx: usize,
                _op: crate::dag::symbol::OpKind,
                _children: &AstChildList,
                _dag_id: crate::dag::node::DagNodeId,
            ) {
                self.ops += 1;
            }
        }
        let mut counter = Counter {
            vars: 0,
            consts: 0,
            ops: 0,
        };
        walk_projection(&proj, &mut counter);
        assert_eq!(counter.vars, 1);
        assert_eq!(counter.consts, 1);
        assert_eq!(counter.ops, 1);
    }
}
