//! Global DAG storage for symbolic expressions.
//!
//! # Architecture
//!
//! The DAG is the **single source of truth** for all symbolic data. Every
//! unique sub-expression is stored exactly once via hash-consing (structural
//! deduplication). This module provides:
//!
//! - `symbol` — Symbol identifiers, kinds, and the symbol registry.
//! - `metadata` — Per-node metadata: hash, coefficients, arity, flags.
//! - `node` — The `DagNode` type and its children representation.
//! - `arena` — A typed arena allocator (`DagArena`) for bulk node storage.
//! - `dedup` — Hash-consing layer that ensures structural sharing.
//! - `builder` — High-level fluent API for constructing expressions.

pub mod arena;
pub mod builder;
pub mod dedup;
pub mod metadata;
pub mod node;
pub mod symbol;
