//! JIT compilation pipeline for symbolic derivation rules.
//!
//! # Architecture
//!
//! Compiles algebraic rewrite rules into native machine code using the
//! Cranelift code generator. This module is gated behind the `jit` feature.
//!
//! - `compiler` — `JitCompiler` wrapping `cranelift_jit::JITModule`.
//! - `primitives` — Built-in rules: add/sub, mul, div with guards.
//! - `custom` — User-defined derivation rules compiled to native code.
//! - `cache` — Compiled function cache keyed by rule hash.
//! - `codegen` — Cranelift IR generation helpers and prefetch emission.

pub mod cache;
pub mod codegen;
pub mod compiler;
pub mod custom;
pub mod primitives;
