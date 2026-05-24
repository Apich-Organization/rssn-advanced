//! Integration tests for the DAG module.

#[cfg(test)]
mod dag_tests {
    use rssn_advanced::dag::builder::DagBuilder;
    use rssn_advanced::dag::symbol::SymbolKind;

    #[test]
    fn test_dag_deduplication_and_sharing() {
        let mut builder = DagBuilder::new();

        // Build: (x + y) * (x + y)
        // This should result in exactly:
        // - 1 node for variable "x"
        // - 1 node for variable "y"
        // - 1 node for operator "+" (x + y)
        // - 1 node for operator "*" ((x + y) * (x + y))
        // Total nodes: 4!

        let x = builder.variable("x");
        let y = builder.variable("y");

        let add1 = builder.add(x, y);
        let add2 = builder.add(x, y);

        assert_eq!(
            add1, add2,
            "Identical additions must be deduplicated to the exact same DagNodeId"
        );

        let expr = builder.mul(add1, add2);

        // Verify total node count in the arena is exactly 4
        assert_eq!(
            builder.arena().len(),
            4,
            "DAG did not achieve perfect structural sharing!"
        );

        // Retrieve nodes and verify structure
        let mul_node = builder.arena().get(expr).unwrap();
        assert_eq!(mul_node.children.len(), 2);
        assert_eq!(mul_node.children.as_slice()[0], add1);
        assert_eq!(mul_node.children.as_slice()[1], add1);

        let add_node = builder.arena().get(add1).unwrap();
        assert_eq!(add_node.children.len(), 2);
        assert_eq!(add_node.children.as_slice()[0], x);
        assert_eq!(add_node.children.as_slice()[1], y);

        // Verify variable names in registry
        if let SymbolKind::Variable(x_sym) = builder.arena().get(x).unwrap().kind {
            assert_eq!(builder.registry().name(x_sym), Some("x"));
        } else {
            panic!("Expected x to be a variable");
        }
    }

    #[test]
    fn test_complex_algebraic_construction() {
        let mut builder = DagBuilder::new();

        // Build: -((a * b) ^ 2.0) / c
        let a = builder.variable("a");
        let b = builder.variable("b");
        let c = builder.variable("c");
        let two = builder.constant(2.0);

        let mul = builder.mul(a, b);
        let power = builder.pow(mul, two);
        let negation = builder.neg(power);
        let division = builder.div(negation, c);

        assert_eq!(
            builder.arena().len(),
            8,
            "Expected exactly 8 nodes in the arena"
        );

        // Verify root node properties
        let root = builder.arena().get(division).unwrap();
        assert_eq!(root.children.len(), 2);
        assert_eq!(root.children.as_slice()[0], negation);
        assert_eq!(root.children.as_slice()[1], c);
    }
}
