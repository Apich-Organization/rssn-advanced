//! Heuristic search toolbox for NP-hard pattern matching.
//!
//! Provides a "programmable heuristic engine" — a configurable framework
//! for approximate simplification when exact methods face symbol explosion.
//! Users tune knobs (depth, branching, timeout) to trade accuracy for speed.
//!
//! - `engine` — `HeuristicEngine`: the configurable pattern-matching core.
//! - `strategy` — Built-in strategies: greedy, beam search, random restart.
//! - `knobs` — Tunable parameters (`HeuristicConfig`).
//! - `simplifier` — Approximate simplification for graceful degradation.

pub mod engine;
pub mod knobs;
pub mod patterns;
pub mod rule_registry;
pub mod simplifier;
pub mod strategy;

pub use engine::HeuristicEngine;
pub use knobs::HeuristicConfig;
pub use rule_registry::RuleRegistry;
pub use simplifier::approximate_simplify;
pub use strategy::SearchStrategy;
