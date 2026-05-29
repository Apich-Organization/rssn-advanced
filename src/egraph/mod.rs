//! Lightweight E-graph for equality saturation over hash-consed DAGs.
//!
//! ## What is an E-graph?
//!
//! An E-graph (equality graph) is a data structure for compactly representing
//! many equivalent expressions simultaneously. It augments a DAG with
//! **equivalence classes** (e-classes) so that semantically equivalent
//! sub-expressions can co-exist and be compared by cost.
//!
//! ## Why not `egg`?
//!
//! The [`egg`] crate is a general-purpose E-graph library. It is too heavy
//! for our use case: it owns its own AST representation, does not compose
//! with hash-consed DAGs, and imposes heavy dependencies. Instead, we build
//! a lightweight E-graph that:
//!
//! * Sits **on top** of our existing [`DagBuilder`] — new rewrites go
//!   through the dedup map, preserving structural sharing.
//! * Uses a compact **path-compressed union-find** directly over
//!   [`DagNodeId`] values (u32 indices into the arena).
//! * Runs **equality saturation** with a configurable round/merge budget to
//!   bound compile-time cost.
//! * Extracts the **minimum-cost** representative from each e-class using a
//!   bottom-up cost function tailored to our JIT's operation costs.
//!
//! ## Integration with the heuristic engine
//!
//! Call [`EGraph::new`] → [`EGraph::saturate`] → [`EGraph::extract`] as a
//! post-pass after [`crate::heuristic::HeuristicEngine::simplify`], or as a
//! standalone optimizer for expressions that need algebraic equivalence
//! reasoning (commutativity, constant folding, x^2=x*x, etc.).
//!
//! ```rust
//! use rssn_advanced::dag::builder::DagBuilder;
//! use rssn_advanced::egraph::{EGraph, EGraphConfig};
//!
//! let mut builder = DagBuilder::new();
//! let x   = builder.variable("x");
//! let one = builder.constant(1.0);
//! let expr = builder.mul(x, one);   // x * 1
//!
//! let mut eg = EGraph::new(&mut builder, EGraphConfig::default());
//! eg.saturate(expr);
//! let best = eg.extract(expr);       // returns x (cheaper than x*1)
//! assert_eq!(best, x);
//! ```
//!
//! [`DagBuilder`]: crate::dag::builder::DagBuilder
//! [`DagNodeId`]: crate::dag::node::DagNodeId

pub mod cost;
#[allow(clippy::module_inception)]
pub mod egraph;
pub mod union_find;

pub use cost::{CostWeights, node_cost};
pub use egraph::{EGraph, EGraphConfig, RewriteRule};
pub use union_find::UnionFind;
