//! Integration tests for parallel computation.

#[cfg(test)]
mod parallel_tests {
    use rssn_advanced::dag::builder::DagBuilder;
    use rssn_advanced::dag::symbol::SymbolId;
    use rssn_advanced::parallel::permission::SymbolPermissions;
    use rssn_advanced::parallel::simplify::{SimplifyConfig, ThreadLocalState};
    use rssn_advanced::parallel::solver::parallel_evaluate;
    use rssn_advanced::parallel::splitter::split_commutative_tree;
    use rssn_advanced::parser::expr::parse_expression;

    #[test]
    fn test_commutativity_permission() {
        let permissions = SymbolPermissions::new();
        let sym_x = SymbolId::new(0);
        let sym_y = SymbolId::new(1);

        assert!(!permissions.is_commutative(sym_x));
        assert!(!permissions.is_commutative(sym_y));

        permissions.set_commutative(sym_x, true);
        assert!(permissions.is_commutative(sym_x));
        assert!(!permissions.is_commutative(sym_y));

        permissions.set_commutative(sym_x, false);
        assert!(!permissions.is_commutative(sym_x));
    }

    #[test]
    fn test_tree_splitting_and_parallel_evaluation() {
        let mut builder = DagBuilder::new();

        // Build: x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
        let root = parse_expression("x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7", &mut builder).unwrap();

        let permissions = SymbolPermissions::new();
        // Enable commutativity for all 8 variables (SymbolId 0 to 7)
        for i in 0..8 {
            permissions.set_commutative(SymbolId::new(i), true);
        }

        // Split addition tree into 4 parallel chunks
        let chunks = split_commutative_tree(builder.arena(), root, &permissions, 4);
        assert_eq!(chunks.len(), 4, "Expected exactly 4 partitioned chunks");
        for chunk in &chunks {
            assert_eq!(
                chunk.len(),
                2,
                "Expected exactly 2 leaf nodes per chunk in balanced split"
            );
        }

        // Variables: x0=1.0, x1=2.0, x2=3.0, x3=4.0, x4=5.0, x5=6.0, x6=7.0, x7=8.0
        let vars = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Evaluate chunks in parallel using our Solver
        let total_sum = parallel_evaluate(builder.arena(), chunks, &vars);

        // Expected: 1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0 = 36.0
        assert!(
            (total_sum - 36.0).abs() < f64::EPSILON,
            "Parallel evaluation computed wrong sum"
        );
    }

    #[test]
    fn test_simplify_config_builder() {
        let config = SimplifyConfig::new().intermediate_rounds(5);
        assert_eq!(config.intermediate_rounds, 5);
        assert!(config.enable_global_dedup);
    }

    #[test]
    fn test_thread_local_isolation() {
        let state = ThreadLocalState::new();
        assert_eq!(state.get_count(), 0);

        state.increment();
        state.increment();
        assert_eq!(state.get_count(), 2);
    }
}
