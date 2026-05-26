//! SIMD-optimized preset function library.
//!
//! The slice-iterating wrappers in this module dispatch to the 4-lane
//! kernels under [`crate::asm_presets`]. Every kernel emits explicit
//! `core::arch::asm!` instructions for its target ISA — there is no
//! reliance on the compiler's auto-vectorizer (`simd_review §1`).
//!
//! - `arithmetic` — Batch add, mul, scalar-add, pow, cmp-eq, FMA,
//!   coefficient-merge.
//! - `hash` — AES-NI based batch hashing for dedup.
//! - `detect` — Runtime CPU feature detection (SSE4.2, AVX2, NEON).

pub mod arithmetic;
pub mod detect;
pub mod hash;
pub mod kernel;

pub use arithmetic::{
    BatchError, batch_add, batch_add_scalar, batch_cmp_eq, batch_coef_merge, batch_eval,
    batch_eval_parallel, batch_fma, batch_mul, batch_pow,
};
pub use detect::{has_avx2, has_neon, has_sse42};
pub use hash::batch_hash;
pub use kernel::{ScalarKernel, SimdKernel, global_kernel};
