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
#![warn(clippy::dbg_macro, clippy::todo, clippy::unnecessary_safety_comment)]
// -------------------------------------------------------------------------
// LEVEL 3: ALLOW/IGNORABLE (Allow)
// -------------------------------------------------------------------------
#![allow(
    unsafe_code,
    clippy::restriction,
    unused_doc_comments,
    clippy::empty_line_after_outer_attr,
    clippy::empty_line_after_doc_comments
)]

// =========================================================================
// Module Declarations
// =========================================================================

/// Inline-assembly preset suite (AVX2 / AES-NI / scalar fallback).
///
/// Each preset is a 4-lane `f64` kernel emitted via `core::arch::asm!`
/// — no reliance on auto-vectorization. Used by both `simd` (slice
/// wrappers) and indirectly by `jit` (peephole patterns). Lives at the
/// crate root so neither subsystem has to feature-gate the other.
pub mod asm_presets;

/// Cold-path error infrastructure.
///
/// Hosts the `rssn_error!` macro and the module-level error enums.
/// Replaces the previous ad-hoc `unwrap()` / `expect()` / `assert_eq!`
/// pattern with `#[cold] #[inline(never)]` constructors so that error
/// handling stays off the hot path.
pub mod error;

/// Zero-copy borrowed containers and `bincode-next` `BorrowDecode` glue.
///
/// `BorrowedSlice` / `BorrowedArena` decode by `take_bytes` directly off
/// the input buffer, and `MmapBuffer` provides file-backed storage with
/// 8-byte aligned bytes for safe reinterpretation as `&[T: Pod]`.
pub mod zerocopy;

/// Fiber-based task runtime built on `dtact`.
///
/// Replaces `std::thread::spawn` with lightweight fibers throughout the
/// crate. `parallel_for_each` is the workhorse used by `parallel::solver`
/// and `ffi::async_bridge`.
pub mod runtime;

/// Allocator-light shared utilities (worklist traversals, helpers).
pub mod util;

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
/// machine code via Cranelift. Gated behind the `jit` feature (alias
/// `cranelift-jit` kept for backward compatibility).
#[cfg(feature = "jit")]
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

/// Lightweight E-graph for equality saturation.
///
/// Implements equality saturation over the hash-consed DAG without importing
/// the heavy `egg` crate. Uses a path-compressed union-find directly over
/// [`dag::node::DagNodeId`] values, and extracts the minimum-cost
/// representative after each saturation run.
pub mod egraph;

/// SIMD-optimized preset function library.
///
/// Hardware-accelerated batch operations (arithmetic, hashing) with
/// runtime feature detection and scalar fallback.
pub mod simd;

/// Unified custom-operator extension system.
///
/// A single [`custom::descriptor::CustomOpDescriptor`] bundles every
/// pipeline-facing property of a user-defined operator (JIT eval function,
/// batch-vectorisability flag, heuristic simplification rules, and e-graph
/// rewrite rules).  Register descriptors into a
/// [`custom::descriptor::CustomOpRegistry`], then call the three integration
/// methods to wire the operator into all pipeline stages simultaneously:
///
/// ```rust,ignore
/// let reg = Arc::new(registry);
/// reg.apply_to_jit(&mut compiler);
/// let rule_reg = reg.build_rule_registry();
/// reg.apply_to_egraph(&mut egraph);
/// ```
///
/// C/C++ callers use the `rssn_custom_op_*` family in [`ffi::c_api`].
pub mod custom;

/// C/C++ Foreign Function Interface.
///
/// Exposes a flat, `extern "C"` API surface via `cbindgen`-compatible
/// types and opaque handles. Includes an async bridge for multi-language
/// integration.
pub mod ffi;

#[cfg_attr(miri, ignore)]
mod readme {
    #![doc = include_str!("../README.md")]
}
