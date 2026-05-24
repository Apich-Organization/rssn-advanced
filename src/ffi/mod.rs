//! C/C++ Foreign Function Interface.
//!
//! Exposes a flat, `extern "C"` API surface using opaque handles and
//! error codes. All types are `cbindgen`-compatible for automatic
//! C header generation.
//!
//! - `c_api`        — `#[no_mangle] pub extern "C"` entry points for all
//!                    operators, parsing, function calls, and rule registration.
//! - `types`        — C-compatible opaque handles and error codes.
//! - `async_bridge` — Callback-based async CFFI bridge for multi-language use.
//! - `jit_context`  — Persistent JIT compilation context handle.

pub mod async_bridge;
pub mod c_api;
pub mod jit_context;
pub mod types;

pub use async_bridge::{RssnAsyncHandle, rssn_async_join, rssn_dag_simplify_async};
pub use c_api::{
    RssnEGraphConfig,
    RssnEGraphRuleCallback,
    RssnKind,
    RssnNodeDesc,
    RssnRuleCallback,
    RssnRuleRegistry,
    // Configuration types
    RssnSimplifyConfig,
    // Binary arithmetic operators (legacy u32-sentinel variants)
    rssn_dag_add,
    // Binary arithmetic operators (status-returning v2 variants)
    rssn_dag_add_v2,
    // Batch build + packed snapshot (reduced FFI overhead)
    rssn_dag_batch_build,
    rssn_dag_call_fn,
    rssn_dag_call_fn_v2,
    // JIT compile/execute
    rssn_dag_compile,
    rssn_dag_compile_v2,
    rssn_dag_constant,
    rssn_dag_constant_v2,
    rssn_dag_div,
    rssn_dag_div_v2,
    // E-graph equality saturation
    rssn_dag_egraph_saturate_extract,
    rssn_dag_execute,
    rssn_dag_execute_v2,
    rssn_dag_free,
    rssn_dag_get_packed,
    // Function interning and call nodes
    rssn_dag_intern_function,
    rssn_dag_mod,
    rssn_dag_mod_v2,
    rssn_dag_mul,
    rssn_dag_mul_v2,
    // Unary operators
    rssn_dag_neg,
    // Unary operators v2
    rssn_dag_neg_v2,
    // Core DAG lifecycle
    rssn_dag_new,
    // Expression parsing
    rssn_dag_parse,
    rssn_dag_pow,
    rssn_dag_pow_v2,
    // Simplification
    rssn_dag_simplify,
    rssn_dag_simplify_v2,
    rssn_dag_simplify_with_config,
    rssn_dag_simplify_with_egraph,
    rssn_dag_simplify_with_rules,
    rssn_dag_sub,
    rssn_dag_sub_v2,
    // Variable and constant leaf nodes
    rssn_dag_variable,
    rssn_dag_variable_v2,
    // JIT function registration from C
    rssn_jit_register_fn_1,
    rssn_jit_register_fn_2,
    rssn_jit_register_fn_3,
    rssn_rule_register,
    rssn_rule_registry_free,
    // C-side rule registry
    rssn_rule_registry_new,
};
pub use jit_context::{rssn_dag_compile_with_ctx, rssn_jit_context_free, rssn_jit_context_new};
pub use types::RssnStatus;

// Re-export the JIT-gated compile-with-opts and its config type.
#[cfg(feature = "cranelift-jit")]
pub use c_api::{RssnOptConfig, rssn_dag_compile_with_opts};
