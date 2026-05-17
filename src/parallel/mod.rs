//! Parallel computation engine.
//!
//! # Architecture
//!
//! Exploits the commutativity of algebraic operations to split expressions
//! into independent chunks for parallel simplification. Each chunk gets
//! its own AST projection — no shared mutable state during computation.
//!
//! - `splitter` — Commutativity-based expression partitioning.
//! - `solver` — Async parallel solver with work distribution.
//! - `simplify` — Staged global simplification (end-of-computation default).
//! - `permission` — Per-symbol commutativity permission flags.

pub mod permission;
pub mod simplify;
pub mod solver;
pub mod splitter;
