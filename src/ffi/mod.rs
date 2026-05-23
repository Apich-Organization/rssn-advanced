//! C/C++ Foreign Function Interface.
//!
//! Exposes a flat, `extern "C"` API surface using opaque handles and
//! error codes. All types are `cbindgen`-compatible for automatic
//! C header generation.
//!
//! - `c_api`       — `#[no_mangle] pub extern "C"` entry points.
//! - `types`       — C-compatible opaque handles and error codes.
//! - `async_bridge` — Callback-based async CFFI bridge for multi-language use.
//! - `jit_context` — Persistent JIT compilation context handle.

pub mod async_bridge;
pub mod c_api;
pub mod jit_context;
pub mod types;

pub use async_bridge::{
    rssn_async_join, rssn_dag_simplify_async_v2, RssnAsyncHandle,
};
pub use c_api::{
    rssn_dag_add, rssn_dag_add_v2, rssn_dag_compile, rssn_dag_compile_v2, rssn_dag_constant,
    rssn_dag_constant_v2, rssn_dag_execute, rssn_dag_execute_v2, rssn_dag_free, rssn_dag_new,
    rssn_dag_simplify, rssn_dag_simplify_v2, rssn_dag_simplify_with_config, rssn_dag_variable,
    rssn_dag_variable_v2, RssnSimplifyConfig,
};
pub use jit_context::{rssn_dag_compile_with_ctx, rssn_jit_context_free, rssn_jit_context_new};
pub use types::RssnStatus;
