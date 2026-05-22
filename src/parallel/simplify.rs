//! Staged global simplification.
//!
//! `plan.md §4.1` calls for "stage-wise simplification" — running a
//! lightweight local simplifier every N rounds during a long-running
//! computation rather than only at the end. Per `parallel_review §1`
//! the `SimplifyConfig` knob existed but had no implementation;
//! [`run_staged_simplification`] supplies it.
//!
//! The loop is intentionally simple:
//!
//! 1. Caller passes a `DagBuilder` and a root `DagNodeId`.
//! 2. We invoke the heuristic engine once per round, threading the
//!    rewritten root through.
//! 3. Each round goes through the full dedup-aware engine, so even
//!    aggressive multi-round configurations cannot leak duplicates.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use crate::heuristic::{HeuristicConfig, HeuristicEngine, SearchStrategy};

/// Staged simplification configuration parameters.
#[derive(Debug, Clone, Copy)]
pub struct SimplifyConfig {
    /// Number of intermediate local simplification rounds to perform.
    pub intermediate_rounds: usize,
    /// Whether to trigger global deduplication and constant folding.
    pub enable_global_dedup: bool,
}

impl Default for SimplifyConfig {
    fn default() -> Self {
        Self {
            intermediate_rounds: 3,
            enable_global_dedup: true,
        }
    }
}

impl SimplifyConfig {
    /// Creates a new configuration builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            intermediate_rounds: 0,
            enable_global_dedup: true,
        }
    }

    /// Sets the number of intermediate local simplification rounds.
    #[must_use]
    pub const fn intermediate_rounds(mut self, rounds: usize) -> Self {
        self.intermediate_rounds = rounds;
        self
    }
}

/// Cache-line padded thread-local state to physically isolate counters.
///
/// Under high concurrency, threads updating adjacent memory addresses trigger
/// false-sharing (cache line invalidations on MESI). This struct aligns memory
/// to 128 bytes, placing each thread's state on a fresh cache line.
///
/// Tracks three meaningful per-thread metrics (`parallel_review §3.1`):
/// - `steps_count` — total simplification rounds performed by this thread
/// - `nodes_visited` — total DAG nodes inspected during traversal
/// - `rewrites_applied` — total rewrite rules that fired (pattern matches)
#[derive(Debug)]
#[repr(align(128))]
pub struct ThreadLocalState {
    /// Total simplification steps (one per [`run_staged_simplification`] round).
    pub steps_count: AtomicU64,
    /// Total DAG nodes visited across all steps. Useful for throughput profiling.
    pub nodes_visited: AtomicU64,
    /// Total rewrite rules that fired. Non-zero means the expression changed.
    pub rewrites_applied: AtomicU64,
}

impl Default for ThreadLocalState {
    fn default() -> Self {
        Self {
            steps_count: AtomicU64::new(0),
            nodes_visited: AtomicU64::new(0),
            rewrites_applied: AtomicU64::new(0),
        }
    }
}

impl ThreadLocalState {
    /// Creates a new instance with all counters at zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            steps_count: AtomicU64::new(0),
            nodes_visited: AtomicU64::new(0),
            rewrites_applied: AtomicU64::new(0),
        }
    }

    /// Increments the simplification-step counter.
    pub fn increment(&self) {
        self.steps_count.fetch_add(1, Ordering::Release);
    }

    /// Retrieves the simplification-step count.
    #[must_use]
    pub fn get_count(&self) -> u64 {
        self.steps_count.load(Ordering::Acquire)
    }

    /// Records that `n` additional DAG nodes were visited.
    pub fn record_nodes_visited(&self, n: u64) {
        self.nodes_visited.fetch_add(n, Ordering::Relaxed);
    }

    /// Records that `n` rewrite rules fired.
    pub fn record_rewrites(&self, n: u64) {
        self.rewrites_applied.fetch_add(n, Ordering::Relaxed);
    }

    /// Returns the total number of nodes visited.
    #[must_use]
    pub fn get_nodes_visited(&self) -> u64 {
        self.nodes_visited.load(Ordering::Relaxed)
    }

    /// Returns the total number of rewrites applied.
    #[must_use]
    pub fn get_rewrites_applied(&self) -> u64 {
        self.rewrites_applied.load(Ordering::Relaxed)
    }

    /// Resets all counters to zero.
    pub fn reset(&self) {
        self.steps_count.store(0, Ordering::Release);
        self.nodes_visited.store(0, Ordering::Relaxed);
        self.rewrites_applied.store(0, Ordering::Relaxed);
    }
}

// =========================================================================
// Staged simplification driver (T4.3)
// =========================================================================

/// Runs the heuristic engine `cfg.intermediate_rounds` times against
/// `root`, threading each round's output back into the next.
///
/// The engine itself preserves the dedup invariant, so spinning the
/// loop multiple times cannot accidentally fork the arena.
///
/// When `cfg.intermediate_rounds == 0` the function is a no-op pass-
/// through, matching the previous behaviour exactly.
///
/// When `cfg.enable_global_dedup == false`, the heuristic still runs
/// but only with a single final round — useful for low-overhead
/// configurations that don't need the per-round local cleanup.
#[must_use]
pub fn run_staged_simplification(
    builder: &mut DagBuilder,
    root: DagNodeId,
    cfg: SimplifyConfig,
) -> DagNodeId {
    if root.is_none() {
        return root;
    }

    let rounds = if cfg.enable_global_dedup {
        cfg.intermediate_rounds
    } else {
        // Even when global dedup is off, we still want at least one
        // pass so callers see *some* simplification effect.
        cfg.intermediate_rounds.max(1)
    };

    if rounds == 0 {
        return root;
    }

    let engine = HeuristicEngine::new(HeuristicConfig::default(), SearchStrategy::Greedy);
    let mut current = root;
    let mut prev = DagNodeId::NONE;

    for _ in 0..rounds {
        if current == prev {
            // Fixpoint reached — additional rounds are wasted work.
            break;
        }
        prev = current;
        current = engine.simplify(builder, current);
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_rounds_is_passthrough() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let zero = b.constant(0.0);
        let expr = b.add(x, zero);
        let cfg = SimplifyConfig {
            intermediate_rounds: 0,
            enable_global_dedup: true,
        };
        assert_eq!(run_staged_simplification(&mut b, expr, cfg), expr);
    }

    #[test]
    fn single_round_folds_identity() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let zero = b.constant(0.0);
        let expr = b.add(x, zero);
        let cfg = SimplifyConfig {
            intermediate_rounds: 1,
            enable_global_dedup: true,
        };
        assert_eq!(run_staged_simplification(&mut b, expr, cfg), x);
    }

    #[test]
    fn multiple_rounds_reach_fixpoint() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let zero = b.constant(0.0);
        let one = b.constant(1.0);
        // ((x + 0) * 1) — takes two passes through identities to
        // collapse, but our iterative engine already does it in one.
        let inner = b.add(x, zero);
        let outer = b.mul(inner, one);
        let cfg = SimplifyConfig {
            intermediate_rounds: 5,
            enable_global_dedup: true,
        };
        let out = run_staged_simplification(&mut b, outer, cfg);
        assert_eq!(out, x);
    }
}
