//! Built-in search strategies.
//!
//! Exposes a pluggable strategy parameter that controls the rewrite rule search traversal.

/// Pluggable heuristic search strategies for algebraic rewriting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchStrategy {
    /// Always pick the first matching simplified rewrite rule.
    #[default]
    Greedy,
    /// Explore top-K candidates at each expansion step to escape sub-optimal trees.
    BeamSearch(usize),
    /// Perform randomized restarts using high branching factors.
    RandomRestart(usize),
}
