//! Integration tests for the AST projection module.

#[cfg(test)]
mod ast_tests {
    use rssn_advanced::ast::convert::{dag_to_ast, ast_to_dag};
    use rssn_advanced::dag::builder::DagBuilder;
    use rssn_advanced::dag::symbol::{OpKind, SymbolKind};

    #[test]
    fn test_complex_ast_projection_traversal() {
        let mut builder = DagBuilder::new();

        // Build: x * y + z^2
        let x = builder.variable("x");
        let y = builder.variable("y");
        let z = builder.variable("z");
        let two = builder.constant(2.0);

        let mul = builder.mul(x, y);
        let power = builder.pow(z, two);
        let root = builder.add(mul, power);

        // Project the entire DAG expression to a stack-local AST tree
        let ast = dag_to_ast(builder.arena(), root);

        // Verify size
        assert_eq!(ast.len(), 7, "AST projection should contain exactly 7 nodes");

        // Manually traverse the AST projection using relative pointers
        let root_node = ast.root().unwrap();
        assert_eq!(root_node.kind, SymbolKind::Operator(OpKind::Add));

        // Get left child (x * y)
        if let rssn_advanced::ast::projection::AstChildList::Two([left_ptr, right_ptr]) = root_node.children {
            let left_child = ast.resolve(0, left_ptr).unwrap();
            assert_eq!(left_child.kind, SymbolKind::Operator(OpKind::Mul));

            // Get left-left child ("x")
            if let rssn_advanced::ast::projection::AstChildList::Two([ll_ptr, _]) = left_child.children {
                let left_idx = left_ptr.resolve(0).unwrap();
                let ll_child = ast.resolve(left_idx, ll_ptr).unwrap();
                if let SymbolKind::Variable(sym_id) = ll_child.kind {
                    assert_eq!(builder.registry().name(sym_id), Some("x"));
                } else {
                    panic!("Expected variable x");
                }
            }

            // Get right child (z^2)
            let right_child = ast.resolve(0, right_ptr).unwrap();
            assert_eq!(right_child.kind, SymbolKind::Operator(OpKind::Pow));
        } else {
            panic!("Expected addition to have exactly 2 children");
        }
    }

    #[test]
    fn test_ast_projection_mutability_and_dedup_writeback() {
        let mut builder = DagBuilder::new();

        // Build: (a + b) + (a + b)
        let a = builder.variable("a");
        let b = builder.variable("b");
        let sum = builder.add(a, b);
        let root = builder.add(sum, sum);

        assert_eq!(builder.arena().len(), 4, "DAG size should be exactly 4 due to perfect sharing");

        // Project DAG to stack-local AST
        let mut ast = dag_to_ast(builder.arena(), root);
        assert_eq!(ast.len(), 7, "Unshared AST tree projection should have 7 nodes");

        // Let's perform a local transformation: rewrite (a + b) + (a + b) to (a + b) * 2.0
        // We replace the root node (index 0) with a Multiplication operator having sum (index 1) and a new constant 2.0 as children.
        // We can append constant 2.0 at the end of the AST array (index 7).
        let two_const = rssn_advanced::ast::projection::AstNode {
            kind: SymbolKind::Constant,
            value: Some(2.0),
            dag_id: rssn_advanced::dag::node::DagNodeId::NONE,
            children: rssn_advanced::ast::projection::AstChildList::Empty,
        };
        ast.nodes.push(two_const);
        let const_idx = ast.len() - 1;

        // Root is at index 0. The sum subtree starts at index 1.
        let left_ptr = rssn_advanced::ast::pointer::RelPtr::from_indices(0, 1);
        let right_ptr = rssn_advanced::ast::pointer::RelPtr::from_indices(0, const_idx);

        // Update the root to be a multiplication operator
        ast.nodes[0].kind = SymbolKind::Operator(OpKind::Mul);
        ast.nodes[0].children = rssn_advanced::ast::projection::AstChildList::Two([left_ptr, right_ptr]);

        // Writeback the modified stack-local AST to the DAG builder
        let new_root = ast_to_dag(&ast, &mut builder);

        // Verify the newly written DAG structure
        let new_root_node = builder.arena().get(new_root).unwrap();
        assert_eq!(new_root_node.kind, SymbolKind::Operator(OpKind::Mul));

        // Verify node count has only increased by 2 (constant 2.0 + multiplication node)
        // (a + b) was already in the builder's dedup map, so it gets perfectly shared!
        assert_eq!(builder.arena().len(), 6, "Writeback failed to deduplicate subexpressions correctly");
    }
}
