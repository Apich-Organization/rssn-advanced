//! Inline-assembly preset suite — multi-architecture SIMD kernels.
//!
//! `plan.md §4.3` and the review summary (`jit_review §1`,
//! `simd_review §1`) both demand "explicit `inline_asm!` presets, not
//! auto-vectorization". Each preset exposes a single `apply` function
//! that dispatches to the best available SIMD backend for the current
//! target, then falls back to a scalar implementation guaranteed to
//! produce identical `f64` outputs.
//!
//! ## Architecture dispatch
//!
//! | Preset | x86_64 | AArch64 | riscv64 |
//! |---|---|---|---|
//! | add / sub / mul / div | AVX2 (`vaddpd` / `vsubpd` / `vmulpd` / `vdivpd`) | NEON (`fadd` / `fsub` / `fmul` / `fdiv`) | RVV 1.0 (`vfadd.vv` / `vfsub.vv` / `vfmul.vv` / `vfdiv.vv`) |
//! | sqrt / neg / abs | AVX2 (`vsqrtpd` / `vxorpd` / `vandpd`) | NEON (`fsqrt` / `fneg` / `fabs`) | RVV 1.0 (`vfsqrt.v` / `vfsgnjn.vv` / `vfsgnjx.vv`) |
//! | fma | AVX2 (`vfmadd231pd`) | NEON (`fmla`) | RVV 1.0 (`vfmacc.vv`) |
//! | coef_merge | AVX2 (`vmulpd` ×3) | NEON (`fmul` ×3) | RVV 1.0 (`vfmul.vv` ×3) |
//! | cmp_eq | AVX2 (`vcmpeqpd` + `vmovmskpd`) | NEON (`fcmeq` + `umov`) | scalar |
//! | hash | AES-NI (`aesenc`) | AES crypto ext (`aese` + `aesmc`) | scalar |
//!
//! **AArch64**: NEON is mandatory on ARMv8-A — no runtime detection.
//! The optional AES crypto extension is checked at runtime via
//! `std::arch::is_aarch64_feature_detected!("aes")`.
//!
//! **riscv64**: The RVV path activates only when the crate is compiled
//! with `-C target-feature=+v` (`#[cfg(target_feature = "v")]`).
//! Runtime detection (`is_riscv_feature_detected!`) is nightly-only and
//! not used here.
//!
//! Every preset operates on length-4 `f64` slices (one 256-bit register
//! worth on AVX2; two 128-bit registers on NEON; one `vl=4` group on RVV)
//! and one 2-lane `u64` block for the hash kernel. Bulk callers chunk
//! their data into 4-element windows; the tail is the caller's
//! responsibility.

pub mod abs_f64x2_neon;
pub mod abs_f64x4;
pub mod add_f64x2_neon;
pub mod add_f64x4_avx2;
pub mod cmp_eq_f64x4;
pub mod coef_merge_f64x4;
pub mod div_f64x2_neon;
pub mod div_f64x4_avx2;
pub mod fma_f64x4_avx2;
pub mod hash_u64x2_aesni;
pub mod mul_f64x2_neon;
pub mod mul_f64x4_avx2;
pub mod neg_f64x2_neon;
pub mod neg_f64x4;
pub mod sqrt_f64x2_neon;
pub mod sqrt_f64x4;
pub mod sub_f64x2_neon;
pub mod sub_f64x4_avx2;
