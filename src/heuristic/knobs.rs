//! Tunable heuristic parameters.
//!
//! `HeuristicConfig` exposes knobs that control the trade-off between
//! simplification quality and computational cost.

use std::time::Duration;

use crate::egraph::EGraphConfig;

/// Tunable parameters controlling heuristic search budgets.
#[derive(Debug, Clone, Copy)]
pub struct HeuristicConfig {
    /// Maximum search depth during rewriting.
    pub max_depth: usize,
    /// Maximum number of candidates evaluated per step (beam width).
    pub branch_factor: usize,
    /// Wall-clock timeout limit.
    pub timeout: Duration,
    /// Aggressiveness multiplier for approximate pruning (0.0 = exact, 1.0 = highly lossy).
    pub simplification_aggressiveness: f64,
    /// When `true`, run an E-graph equality-saturation pass after the
    /// iterative pattern-matching pass and extract the cheapest result.
    /// Disabled by default to preserve backward compatibility.
    pub use_egraph: bool,
    /// Budget parameters for the E-graph post-pass. Only meaningful when
    /// [`use_egraph`](Self::use_egraph) is `true`.
    pub egraph_config: EGraphConfig,
}

impl Default for HeuristicConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            branch_factor: 4,
            timeout: Duration::from_millis(500),
            simplification_aggressiveness: 0.1,
            use_egraph: false,
            egraph_config: EGraphConfig::default(),
        }
    }
}

impl HeuristicConfig {
    /// Creates a new configuration with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enables the E-graph equality-saturation post-pass with the given budget.
    #[must_use]
    pub const fn with_egraph(mut self, cfg: EGraphConfig) -> Self {
        self.use_egraph = true;
        self.egraph_config = cfg;
        self
    }

    /// Sets the maximum search depth limit.
    #[must_use]
    pub const fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Sets the maximum branching factor (beam width).
    #[must_use]
    pub const fn branch_factor(mut self, factor: usize) -> Self {
        self.branch_factor = factor;
        self
    }

    /// Sets the search timeout.
    #[must_use]
    pub const fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the approximate simplification aggressiveness.
    #[must_use]
    pub const fn simplification_aggressiveness(mut self, aggressiveness: f64) -> Self {
        self.simplification_aggressiveness = aggressiveness;
        self
    }
}
