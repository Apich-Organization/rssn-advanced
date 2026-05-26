//! `sqrt_f64x2` — 2-lane packed `f64` square root.
//!
//! * `x86_64`: SSE2 `sqrtpd xmm` (128-bit, 2 lanes; SSE2 is `x86_64` ABI baseline).
//! * `AArch64`: NEON `fsqrt v.2d` (NEON is mandatory on ARMv8-A).
//! * riscv64 + RVV: `vfsqrt.v` with `vsetvli vl=2, e64`.
//! * fallback: `f64::sqrt()` per lane.
//!
//! `sqrt(-x) → NaN` per IEEE-754; no special-casing is performed.

#![allow(unsafe_code)]

/// Computes element-wise square root of 2 `f64` values, writing results to `out`.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn apply(inp: &[f64; 2], out: &mut [f64; 2]) {
    #[cfg(target_arch = "x86_64")]
    {
        // SSE2 is part of the x86_64 ABI baseline — no runtime detection needed.
        unsafe {
            use core::arch::asm;
            asm!(
                "movupd xmm0, xmmword ptr [{inp}]",
                "sqrtpd xmm0, xmm0",
                "movupd xmmword ptr [{out}], xmm0",
                inp = in(reg) inp.as_ptr(),
                out = in(reg) out.as_mut_ptr(),
                out("xmm0") _,
                options(nostack, preserves_flags),
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on ARMv8-A — no runtime detection needed.
        unsafe {
            use core::arch::asm;
            asm!(
                "ld1 {{v0.2d}}, [{inp}]",
                "fsqrt v0.2d, v0.2d",
                "st1 {{v0.2d}}, [{out}]",
                inp = in(reg) inp.as_ptr(),
                out = in(reg) out.as_mut_ptr(),
                out("v0") _,
                options(nostack, preserves_flags),
            );
        }
        return;
    }

    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        // RVV 1.0 with vl=2, e64.
        unsafe {
            use core::arch::asm;
            asm!(
                "li t0, 2",
                "vsetvli t0, t0, e64, m1, ta, ma",
                "vle64.v v0, ({inp})",
                "vfsqrt.v v0, v0",
                "vse64.v v0, ({out})",
                inp = in(reg) inp.as_ptr(),
                out = in(reg) out.as_mut_ptr(),
                out("t0") _,
                out("v0") _,
                options(nostack),
            );
        }
        return;
    }

    // Scalar fallback.
    out[0] = inp[0].sqrt();
    out[1] = inp[1].sqrt();
}

/// AArch64 NEON intrinsic helper — retained for backward compatibility.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn apply_neon(inp: &[f64; 2], out: &mut [f64; 2]) {
    use std::arch::aarch64::*;
    let a = vld1q_f64(inp.as_ptr());
    vst1q_f64(out.as_mut_ptr(), vsqrtq_f64(a));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqrt_known_values() {
        let a = [4.0_f64, 9.0];
        let mut out = [0.0_f64; 2];
        apply(&a, &mut out);
        assert!((out[0] - 2.0).abs() < 1e-12);
        assert!((out[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn sqrt_negative_is_nan() {
        let a = [-1.0_f64, 0.0];
        let mut out = [0.0_f64; 2];
        apply(&a, &mut out);
        assert!(out[0].is_nan());
        assert!((out[1] - 0.0).abs() < 1e-12);
    }
}
