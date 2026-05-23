//! JIT compilation pipeline for symbolic derivation rules.
//!
//! # Architecture
//!
//! Compiles algebraic rewrite rules into native machine code using the
//! Cranelift code generator. This module is gated behind the `jit` feature.
//!
//! - `compiler` — `JitCompiler` wrapping `cranelift_jit::JITModule`.
//! - `primitives` — Built-in rules: add/sub, mul, div with guards.
//! - `cache` — Compiled function cache keyed by rule hash.
//! - `codegen` — Cranelift IR generation helpers and prefetch emission.
//! - `analysis` — Pre-codegen bottom-up analysis pass (is_nonzero, pow_expansion).
//! - `passes` — Named optimization passes: emit_int_pow, emit_sqrt.
//! - `batch` — Vectorized (2× unrolled scalar) batch evaluation.

pub mod analysis;
pub mod cache;
pub mod codegen;
pub mod compiler;
pub mod passes;
pub mod primitives;

pub use analysis::{NodeAnalysis, PowExpansion};
pub use compiler::{CompiledBatchFn, CompiledExprFn, JitCompiler, OptConfig};
