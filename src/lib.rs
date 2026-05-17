//! This is rssn-advanced: The next generation symbolic core of rssn.
//! rssn-advanced is part of the rssn project and please notice that the main rssn crate is still the main focus of development.
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/Apich-Organization/rssn/refs/heads/dev/doc/logo.png"
)]
#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/Apich-Organization/rssn/refs/heads/dev/doc/favicon.ico"
)]

// -------------------------------------------------------------------------
// Rust Lint Configuration: rssn-advanced
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// LEVEL 1: CRITICAL ERRORS (Deny)
// -------------------------------------------------------------------------
#![deny(
    // Rust Compiler Errors
    dead_code,
    unreachable_code,
    improper_ctypes_definitions,
    future_incompatible,
    nonstandard_style,
    rust_2018_idioms,
    clippy::perf,
    clippy::correctness,
    clippy::suspicious,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::missing_safety_doc,
    clippy::same_item_push,
    clippy::implicit_clone,
    clippy::all,
    clippy::pedantic,
    warnings,
    missing_docs,
    clippy::nursery,
    clippy::single_call_fn,
)]
// -------------------------------------------------------------------------
// LEVEL 2: STYLE WARNINGS (Warn)
// -------------------------------------------------------------------------
#![warn(
    unsafe_code,
    clippy::dbg_macro,
    clippy::todo,
    clippy::unnecessary_safety_comment
)]
// -------------------------------------------------------------------------
// LEVEL 3: ALLOW/IGNORABLE (Allow)
// -------------------------------------------------------------------------
#![allow(
    clippy::restriction,
    unused_doc_comments,
    clippy::empty_line_after_outer_attr,
    clippy::empty_line_after_doc_comments
)]

// =========================================================================
// Module Declarations
// =========================================================================

/// Global DAG (Directed Acyclic Graph) storage for symbolic expressions.
///
/// Provides hash-consed, structurally-shared storage for all symbol nodes.
/// The DAG serves as the canonical representation — all expression data
/// ultimately lives here, deduplicated via structural hashing.
pub mod dag;

/// Local AST (Abstract Syntax Tree) projection for computation.
///
/// Projects a subgraph of the global DAG into a stack-local tree using
/// relative pointers (`i32` / `i64`). This provides an algorithm-friendly
/// tree view without duplicating the underlying metadata.
pub mod ast;

/// Symbolic expression parser.
///
/// Parses mathematical expressions (e.g. `"x^2 + 2*x + 1"`) into the
/// global DAG using `nom`-based combinators with precedence climbing.
pub mod parser;

/// JIT compilation pipeline for symbolic derivation rules.
///
/// Compiles algebraic rewrite rules (add, mul, div, custom) into native
/// machine code via Cranelift. Gated behind the `cranelift-jit` feature.
#[cfg(feature = "cranelift-jit")]
pub mod jit;

/// Parallel computation engine.
///
/// Exploits commutativity to split expressions into independent chunks
/// for async parallel simplification, with staged global merging.
pub mod parallel;

/// Streaming storage and dynamic caching.
///
/// Provides disk-backed spillover for large DAGs and a dynamic hotspot
/// table that tracks intermediate result frequency for auto-eviction.
pub mod storage;

/// Heuristic search toolbox for NP-hard pattern matching.
///
/// A configurable "knob-based" engine that allows controlled approximate
/// simplification when exact methods hit symbol explosion.
pub mod heuristic;

/// SIMD-optimized preset function library.
///
/// Hardware-accelerated batch operations (arithmetic, hashing) with
/// runtime feature detection and scalar fallback.
pub mod simd;

/// C/C++ Foreign Function Interface.
///
/// Exposes a flat, `extern "C"` API surface via `cbindgen`-compatible
/// types and opaque handles. Includes an async bridge for multi-language
/// integration.
pub mod ffi;
