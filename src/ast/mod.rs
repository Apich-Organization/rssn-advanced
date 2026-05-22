//! Local AST projection for stack-local computation.
//!
//! # Architecture
//!
//! The AST projection maps a subgraph of the global DAG into a flat,
//! stack-allocated tree using relative pointers. This gives algorithms
//! a familiar tree interface while metadata remains in the DAG arena.
//!
//! - `projection` — The `AstProjection` type and its buffer management.
//! - `pointer` — `RelPtr<i32>` / `RelPtr<i64>` relative pointer types.
//! - `convert` — DAG ↔ AST conversion routines.

pub mod convert;
pub mod pointer;
pub mod projection;

pub use projection::AstVisitor;
