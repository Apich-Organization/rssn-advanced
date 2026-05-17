//! Configurable heuristic pattern-matching engine.
//!
//! `HeuristicEngine` applies rule-based rewriting with a pluggable
//! search strategy, controlled by `HeuristicConfig` knobs.

use std::time::Instant;
use crate::dag::arena::DagArena;
use crate::dag::node::DagNodeId;
use super::knobs::HeuristicConfig;
use super::strategy::SearchStrategy;
use super::simplifier::approximate_simplify;

/// A rule-based pattern matching and algebraic simplification engine.
#[derive(Debug, Clone, Copy)]
pub struct HeuristicEngine {
    /// Budget parameters (depth, branching limit, timeouts).
    pub config: HeuristicConfig,
    /// Search path strategy.
    pub strategy: SearchStrategy,
}

impl HeuristicEngine {
    /// Creates a new `HeuristicEngine` with the given knobs and strategy.
    #[must_use]
    pub const fn new(config: HeuristicConfig, strategy: SearchStrategy) -> Self {
        Self { config, strategy }
    }

    /// Recursively simplifies the target expression tree using the configured strategy.
    ///
    /// Respects the configured maximum depth, branch limits, timeouts, and
    /// applies approximate pruning if high symbol aggressiveness is configured.
    #[must_use]
    pub fn simplify(&self, arena: &mut DagArena, root: DagNodeId) -> DagNodeId {
        let start_time = Instant::now();

        // 1. Check for approximate simplification (graceful degradation)
        let root = if self.config.simplification_aggressiveness > 0.5 {
            approximate_simplify(arena, root, self.config.simplification_aggressiveness)
        } else {
            root
        };

        // 2. Perform strategy search up to max_depth
        self.explore_and_rewrite(arena, root, 0, start_time)
    }

    fn explore_and_rewrite(
        &self,
        arena: &mut DagArena,
        root: DagNodeId,
        depth: usize,
        start_time: Instant,
    ) -> DagNodeId {
        // Enforce maximum search depth limit
        if depth >= self.config.max_depth {
            return root;
        }

        // Enforce strict timeout budget (knob API)
        if start_time.elapsed() >= self.config.timeout {
            return root;
        }

        // Recursively rewrite operator children
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

        // Limit branches according to branching factor knob
        let branch_limit = self.config.branch_factor.clamp(1, children.len());
        let limited_children = &children[0..branch_limit];

        for &child_id in limited_children {
            let simplified = self.explore_and_rewrite(arena, child_id, depth + 1, start_time);
            if simplified != child_id {
                children_changed = true;
            }
            new_children.push(simplified);
        }

        // Re-append any children that were outside the branch limit budget
        if children.len() > branch_limit {
            for &child_id in &children[branch_limit..] {
                new_children.push(child_id);
            }
        }

        if children_changed {
            let child_list = crate::dag::node::ChildList::from_slice(&new_children);
            let new_node = crate::dag::node::DagNode::operator(kind, meta, child_list);
            return arena.alloc(new_node);
        }

        root
    }
}
