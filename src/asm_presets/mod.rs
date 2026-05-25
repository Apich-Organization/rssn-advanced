#![allow(unreachable_code)]

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
//! ### 4-lane `f64` kernels (`f64x4` / `_avx2` variants)
//!
//! | Preset | x86_64 | AArch64 | riscv64 |
//! |---|---|---|---|
//! | add / sub / mul / div | AVX2 (`vaddpd` / `vsubpd` / `vmulpd` / `vdivpd`) | NEON (`fadd` / `fsub` / `fmul` / `fdiv`) | RVV 1.0 (`vfadd.vv` / `vfsub.vv` / `vfmul.vv` / `vfdiv.vv`) |
//! | sqrt / neg / abs | AVX2 (`vsqrtpd` / `vxorpd` / `vandpd`) | NEON (`fsqrt` / `fneg` / `fabs`) | RVV 1.0 (`vfsqrt.v` / `vfsgnjn.vv` / `vfsgnjx.vv`) |
//! | fma | AVX2 + FMA (`vfmadd231pd`) | NEON (`fmla`) | RVV 1.0 (`vfmacc.vv`) |
//! | coef_merge | AVX2 (`vmulpd` ×3) | NEON (`fmul` ×3) | RVV 1.0 (`vfmul.vv` ×3) |
//! | cmp_eq | AVX2 (`vcmpeqpd` + `vmovmskpd`) | NEON (`fcmeq` + `umov`) | RVV 1.0 (`vmfeq.vv` + `vmerge.vim`) |
//!
//! ### 2-lane `f64` kernels (`f64x2` / `_neon` variants)
//!
//! | Preset | x86_64 | AArch64 | riscv64 |
//! |---|---|---|---|
//! | add / sub / mul / div | SSE2 (`addpd` / `subpd` / `mulpd` / `divpd`) | NEON (`fadd` / `fsub` / `fmul` / `fdiv`) | RVV 1.0 (`vfadd.vv` / `vfsub.vv` / `vfmul.vv` / `vfdiv.vv`, vl=2) |
//! | sqrt / neg / abs | SSE2 (`sqrtpd` / `xorpd` / `andpd`) | NEON (`fsqrt` / `fneg` / `fabs`) | RVV 1.0 (`vfsqrt.v` / `vfsgnjn.vv` / `vfsgnjx.vv`, vl=2) |
//!
//! ### Hash kernel
//!
//! | Preset | x86_64 | AArch64 | riscv64 |
//! |---|---|---|---|
//! | hash_u64x2 | AES-NI (`aesenc`) | AES crypto ext (`aese` + `aesmc`) | Zkn (`aes64esm` ×2 + XOR); scalar xor-mix fallback |
//!
//! ## Detection strategy
//!
//! **x86_64**: AVX2 and AES-NI are checked at runtime via
//! `std::is_x86_feature_detected!`. SSE2 is part of the x86_64 ABI
//! baseline and is used unconditionally (no runtime check needed).
//!
//! **AArch64**: NEON is mandatory on ARMv8-A — no runtime detection.
//! The optional AES crypto extension is checked at runtime via
//! `std::arch::is_aarch64_feature_detected!("aes")`, except on Apple
//! Silicon (M1+) where it is compile-time guaranteed.
//!
//! **riscv64**: The RVV path activates only when compiled with
//! `-C target-feature=+v` (`#[cfg(target_feature = "v")]`).
//! The Zkn path activates only with `-C target-feature=+zkn`
//! (`#[cfg(target_feature = "zkn")]`).
//! Runtime detection (`is_riscv_feature_detected!`) is nightly-only and
//! not used here.
//!
//! ## Lane widths
//!
//! `f64x4` kernels operate on length-4 `f64` slices (one 256-bit register
//! worth on AVX2; two 128-bit registers on NEON; one `vl=4` group on RVV).
//! `f64x2` kernels operate on fixed `[f64; 2]` arrays (one 128-bit register
//! on SSE2/NEON; `vl=2` on RVV). The hash kernel operates on a 2-lane `u64`
//! block. Bulk callers chunk their data into appropriate windows; tail
//! handling is the caller's responsibility.

pub mod abs_f64x2;
pub mod abs_f64x4;
pub mod add_f64x2;
pub mod add_f64x4;
pub mod cmp_eq_f64x4;
pub mod coef_merge_f64x4;
pub mod div_f64x2;
pub mod div_f64x4;
pub mod fma_f64x4;
pub mod hash_u64x2_aesni;
pub mod mul_f64x2;
pub mod mul_f64x4;
pub mod neg_f64x2;
pub mod neg_f64x4;
pub mod sqrt_f64x2;
pub mod sqrt_f64x4;
pub mod sub_f64x2;
pub mod sub_f64x4;
