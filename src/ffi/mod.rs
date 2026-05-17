//! C/C++ Foreign Function Interface.
//!
//! Exposes a flat, `extern "C"` API surface using opaque handles and
//! error codes. All types are `cbindgen`-compatible for automatic
//! C header generation.
//!
//! - `c_api` — `#[no_mangle] pub extern "C"` entry points.
//! - `types` — C-compatible opaque handles and error codes.
//! - `async_bridge` — Callback-based async CFFI bridge for multi-language use.

pub mod async_bridge;
pub mod c_api;
pub mod types;

pub use async_bridge::{rssn_dag_simplify_async, RssnSimplifyCallback};
pub use c_api::{
    rssn_dag_add, rssn_dag_compile, rssn_dag_constant, rssn_dag_execute, rssn_dag_free,
    rssn_dag_new, rssn_dag_simplify, rssn_dag_variable,
};
pub use types::RssnStatus;
