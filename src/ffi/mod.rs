//! C/C++ Foreign Function Interface.
//!
//! Exposes a flat, `extern "C"` API surface using opaque handles and
//! `RssnStatus` error codes.  All types are `cbindgen`-compatible for
//! automatic C header generation.
//!
//! ## Module layout
//!
//! | Module | Contents |
//! |--------|----------|
//! | `types` | `RssnStatus` error enum and primitive C-compatible types |
//! | `c_api` | Synchronous DAG building, simplification, JIT compile/execute |
//! | `jit_context` | Persistent `RssnJitContext` handle (amortises Cranelift init) |
//! | `async_bridge` | Fiber-backed async variants: simplify, compile, eval |
//!
//! ## API stability tiers
//!
//! **Canonical (preferred):** `*_v2` functions return `RssnStatus` and write
//! their output through an `out_*` pointer.  New code should use these.
//!
//! **Legacy (deprecated):** the non-suffixed variants (`rssn_dag_add`, etc.)
//! return `u32::MAX` or `0.0` on error — ambiguous when `0` is a valid result.
//! They are retained for ABI compatibility and will not be removed before a
//! major version bump.

pub mod async_bridge;
pub mod c_api;
pub mod jit_context;
pub mod types;

pub use async_bridge::{
    // Compile async (simplify + JIT compile)
    RssnAsyncCompileHandle,
    // Eval async (simplify + JIT compile + execute)
    RssnAsyncEvalHandle,
    // Simplify async
    RssnAsyncHandle,
    rssn_async_compile_join,
    rssn_async_eval_join,
    rssn_async_join,
    rssn_dag_compile_async,
    rssn_dag_eval_async,
    rssn_dag_simplify_async,
};
pub use c_api::{
    // ── Custom batch operators ───────────────────────────────────────────
    RssnBatchOpCallback,
    // ── Type definitions ──────────────────────────────────────────────────
    RssnEGraphConfig,
    RssnEGraphRuleCallback,
    RssnKind,
    RssnNodeDesc,
    RssnRuleCallback,
    RssnRuleRegistry,
    RssnSimplifyConfig,

    rssn_batch_op_register,
    rssn_batch_op_unregister,

    // ── Binary operators (legacy, deprecated) ─────────────────────────────
    rssn_dag_add,
    // ── Binary operators (canonical _v2) ──────────────────────────────────
    rssn_dag_add_v2,
    // ── Batch build (reduced FFI overhead) ───────────────────────────────
    rssn_dag_batch_build,
    rssn_dag_call_fn, // legacy

    rssn_dag_call_fn_v2,
    rssn_dag_compile,       // legacy (still returns RssnStatus)
    rssn_dag_compile_batch, // compiles 2-row ILP-vectorised version
    // ── JIT compile / execute (canonical) ───────────────────────────────
    rssn_dag_compile_v2,
    rssn_dag_constant,

    rssn_dag_constant_v2,

    rssn_dag_div,
    rssn_dag_div_v2,
    // ── E-graph equality saturation ──────────────────────────────────────
    rssn_dag_egraph_saturate_extract,

    rssn_dag_execute, // legacy (returns f64, 0.0 on error)

    rssn_dag_execute_batch, // dispatches the vectorised batch fn
    // ── Bulk / batch evaluation (amortises FFI overhead) ─────────────────
    rssn_dag_execute_bulk, // scalar JIT fn called in a tight Rust loop
    rssn_dag_execute_v2,
    rssn_dag_free,

    rssn_dag_get_packed,

    // ── Function nodes ────────────────────────────────────────────────────
    rssn_dag_intern_function,
    rssn_dag_mod,

    rssn_dag_mod_v2,

    rssn_dag_mul,
    rssn_dag_mul_v2,
    // ── Unary operators (legacy, deprecated) ─────────────────────────────
    rssn_dag_neg,

    // ── Unary operators (canonical _v2) ───────────────────────────────────
    rssn_dag_neg_v2,

    // ── DAG lifecycle ─────────────────────────────────────────────────────
    rssn_dag_new,
    // ── Expression parsing ────────────────────────────────────────────────
    rssn_dag_parse,

    rssn_dag_pow,
    rssn_dag_pow_v2,
    rssn_dag_simplify, // legacy

    // ── Simplification ───────────────────────────────────────────────────
    rssn_dag_simplify_v2,
    rssn_dag_simplify_with_config,
    rssn_dag_simplify_with_egraph,
    rssn_dag_simplify_with_rules,
    rssn_dag_sub,
    rssn_dag_sub_v2,
    // ── Leaf nodes (legacy, deprecated — u32::MAX sentinel) ───────────────
    rssn_dag_variable,
    // ── Leaf nodes (canonical _v2 — preferred) ────────────────────────────
    rssn_dag_variable_v2,
    // ── JIT custom function registration ─────────────────────────────────
    rssn_jit_register_fn_1,
    rssn_jit_register_fn_2,
    rssn_jit_register_fn_3,

    rssn_rule_register,
    rssn_rule_registry_free,
    // ── C-side rewrite rule registry ─────────────────────────────────────
    rssn_rule_registry_new,
};
pub use jit_context::{rssn_dag_compile_with_ctx, rssn_jit_context_free, rssn_jit_context_new};
pub use types::RssnStatus;

// Re-export the JIT-gated compile-with-opts and its config type.
#[cfg(feature = "cranelift-jit")]
pub use c_api::{RssnOptConfig, rssn_dag_compile_with_opts};
