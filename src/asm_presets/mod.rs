//! Inline-assembly preset suite for the JIT.
//!
//! `plan.md §4.3` and the review summary (`jit_review §1`,
//! `simd_review §1`) both demand "explicit `inline_asm!` presets, not
//! auto-vectorization". This module is the implementation: each preset
//! lives in its own file and exposes a single `apply` function which —
//! on `x86_64` with the appropriate CPU feature detected — drops into
//! `core::arch::asm!` to emit the exact target instruction
//! (`vfmadd231pd`, `vpxor`, `vcmppd`, …). On any other path it falls
//! back to a scalar implementation that the compiler may auto-vectorize
//! or scalarize as it sees fit, with the strict guarantee that the
//! observable f64 outputs are identical.
//!
//! Every preset operates on length-4 `f64` slices (one AVX2 256-bit
//! register's worth) and one 2-lane `u64` block for the hash kernel.
//! Bulk callers should chunk their data into 4-element windows; the
//! tail is the caller's responsibility.

pub mod add_f64x4_avx2;
pub mod cmp_eq_f64x4;
pub mod coef_merge_f64x4;
pub mod fma_f64x4_avx2;
pub mod hash_u64x2_aesni;
pub mod mul_f64x4_avx2;
