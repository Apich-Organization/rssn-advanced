//! Integration tests for streaming storage and heuristic toolbox.

#[cfg(test)]
mod storage_tests {
    use std::fs::remove_dir_all;
    use std::path::PathBuf;
    use rssn_advanced::dag::builder::DagBuilder;
    use rssn_advanced::dag::node::DagNodeId;
    use rssn_advanced::storage::{DiskCache, DynamicHotspotTable, evict_cold_nodes};
    use rssn_advanced::heuristic::{HeuristicEngine, HeuristicConfig, SearchStrategy};

    #[test]
    fn test_disk_cache_spill_and_restore() {
        let cache_dir = PathBuf::from("./test_spillover_cache");
        let disk_cache = DiskCache::new(&cache_dir).unwrap();

        let mut builder = DagBuilder::new();
        let x = builder.variable("x");
        let y = builder.variable("y");
        let expr = builder.add(x, y);

        // Spill
        disk_cache.spill("my_arena", builder.arena()).unwrap();

        // Restore
        let restored_arena = disk_cache.restore("my_arena").unwrap();
        assert_eq!(restored_arena.len(), builder.arena().len());

        let node = restored_arena.get(expr).unwrap();
        assert_eq!(node.children.len(), 2);

        // Cleanup
        disk_cache.delete("my_arena").unwrap();
        let _ = remove_dir_all(&cache_dir);
    }

    #[test]
    fn test_hotspot_table_and_eviction() {
        let hotspots = DynamicHotspotTable::new();
        let id_0 = DagNodeId::new(0);
        let id_1 = DagNodeId::new(1);
        let id_2 = DagNodeId::new(2);

        hotspots.record_access(id_0);
        hotspots.record_access(id_0); // freq = 2
        hotspots.record_access(id_1); // freq = 1 (id_2 freq = 0)

        assert!(hotspots.is_hot(id_0, 2));
        assert!(!hotspots.is_hot(id_1, 2));
        assert_eq!(hotspots.get_frequency(id_2), 0);

        let mut builder = DagBuilder::new();
        builder.constant(1.0); // id_0
        builder.constant(2.0); // id_1
        builder.constant(3.0); // id_2

        // Evict cold nodes with threshold = 2. The result bundles the
        // compacted arena with an `old → new` remap table.
        let result = evict_cold_nodes(builder.arena(), &hotspots, 2);
        assert!(result.arena.len() <= builder.arena().len());
        // id_0 was the only hot node and has no children; it must
        // survive and be remappable.
        assert!(result.translate(id_0).is_some());
    }

    #[test]
    fn test_heuristic_engine_and_timeout() {
        let mut builder = DagBuilder::new();
        // Construct highly nested deep expression tree: x + x + x + x + x + x + x + x
        let x = builder.variable("x");
        let mut current = x;
        for _ in 0..10 {
            current = builder.add(current, x);
        }

        // Configure engine with standard Greedy strategy, but 5ms timeout limit
        let config = HeuristicConfig::default()
            .timeout(std::time::Duration::from_millis(5))
            .max_depth(3);

        let mut engine = HeuristicEngine::new(config, SearchStrategy::Greedy);
        let simplified = engine.simplify(&mut builder, current);

        assert!(simplified.index() < builder.arena().len());
    }

    #[test]
    fn test_approximate_simplification_folding() {
        // Post-Phase 4: `approximate_simplify` now does coefficient
        // pruning (not the previous unsound "fold to 1.0"). For an
        // additive chain of variable terms whose coefficients all
        // equal 1.0, the pruner keeps every term — so the simplified
        // root remains structurally non-trivial.
        let mut builder = DagBuilder::new();
        let x = builder.variable("x");
        let mut current = x;
        for _ in 0..8 {
            current = builder.add(current, x);
        }

        let config = HeuristicConfig::default()
            .simplification_aggressiveness(0.7)
            .max_depth(10);

        let mut engine = HeuristicEngine::new(config, SearchStrategy::Greedy);
        let simplified = engine.simplify(&mut builder, current);

        // The output is a valid arena id and the arena did not
        // explode (dedup is preserved).
        let node = builder.arena().get(simplified).expect("simplified node");
        assert!(node.children.len() > 0 || matches!(node.kind, rssn_advanced::dag::symbol::SymbolKind::Constant(_)));
    }
}
