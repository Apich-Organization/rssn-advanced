//! FFI API for the rssn-advanced library.
//!
//! This module provides a C-compatible foreign function interface (FFI) for interacting
//! with the core data structures and functions of the `rssn-advanced` library.
#![allow(unsafe_code)]
#![allow(clippy::indexing_slicing)]
#![allow(
    clippy::no_mangle_with_rust_abi
)]
// In ffi_apis, we use raw pointers to pass data to and from Rust and C.
// clippy::not_unsafe_ptr_arg_deref is triggered by this.
#![allow(
    clippy::not_unsafe_ptr_arg_deref
)]
// This is enforced by clippy::nursery. It has too high false positive rate.
#![allow(clippy::option_if_let_else)]

#[macro_use]
/// FFI macros.

pub mod macros;

/// Common FFI utilities.
pub mod common;
/// FFI APIs for the constants module.
pub mod constant_ffi;
