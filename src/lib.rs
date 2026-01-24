//! This is rssn-advanced --- a crate containing complex and delicate mathematical algorithms exspcialy heuristic one based on rssn.
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
    clippy::empty_line_after_doc_comments
)]

/// System and physical constants.
pub mod constant;
/// FFI APIs for the 'rssn-advanced' Library.
#[cfg(feature = "ffi_api")]
pub mod ffi_apis;