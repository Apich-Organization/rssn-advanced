//! `div_f64x2` — 2-lane packed `f64` division.
//!
//! * `x86_64`: SSE2 `divpd xmm` (128-bit, 2 lanes; SSE2 is `x86_64` ABI baseline).
//! * `AArch64`: NEON `fdiv v.2d` (NEON is mandatory on ARMv8-A).
//! * riscv64 + RVV: `vfdiv.vv` with `vsetvli vl=2, e64`.
//! * fallback: scalar loop.
//!
//! Division-by-zero follows IEEE-754: `x / ±0.0 → ±Inf`, `0.0 / 0.0 → NaN`.

#![allow(unsafe_code)]

/// Divides `lhs` by `rhs` element-wise, writing the result to `out`.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn apply(lhs: &[f64; 2], rhs: &[f64; 2], out: &mut [f64; 2]) {
    #[cfg(target_arch = "x86_64")]
    {
        // SSE2 is part of the x86_64 ABI baseline — no runtime detection needed.
        unsafe {
            use core::arch::asm;
            asm!(
                "movupd xmm0, xmmword ptr [{lhs}]",
                "movupd xmm1, xmmword ptr [{rhs}]",
                "divpd  xmm0, xmm1",
                "movupd xmmword ptr [{out}], xmm0",
                lhs = in(reg) lhs.as_ptr(),
                rhs = in(reg) rhs.as_ptr(),
                out = in(reg) out.as_mut_ptr(),
                out("xmm0") _,
                out("xmm1") _,
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
                "ld1 {{v0.2d}}, [{lhs}]",
                "ld1 {{v1.2d}}, [{rhs}]",
                "fdiv v0.2d, v0.2d, v1.2d",
                "st1 {{v0.2d}}, [{out}]",
                lhs = in(reg) lhs.as_ptr(),
                rhs = in(reg) rhs.as_ptr(),
                out = in(reg) out.as_mut_ptr(),
                out("v0") _,
                out("v1") _,
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
                "vle64.v v0, ({lhs})",
                "vle64.v v1, ({rhs})",
                "vfdiv.vv v0, v0, v1",
                "vse64.v v0, ({out})",
                lhs = in(reg) lhs.as_ptr(),
                rhs = in(reg) rhs.as_ptr(),
                out = in(reg) out.as_mut_ptr(),
                out("t0") _,
                out("v0") _,
                out("v1") _,
                options(nostack),
            );
        }
        return;
    }

    // Scalar fallback.
    out[0] = lhs[0] / rhs[0];
    out[1] = lhs[1] / rhs[1];
}

/// AArch64 NEON intrinsic helper — retained for backward compatibility.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn apply_neon(lhs: &[f64; 2], rhs: &[f64; 2], out: &mut [f64; 2]) {
    use std::arch::aarch64::*;
    let a = vld1q_f64(lhs.as_ptr());
    let b = vld1q_f64(rhs.as_ptr());
    vst1q_f64(out.as_mut_ptr(), vdivq_f64(a, b));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn div_matches_scalar() {
        let a = [10.0_f64, 20.0];
        let b = [2.0_f64, 4.0];
        let mut out = [0.0_f64; 2];
        apply(&a, &b, &mut out);
        assert_eq!(out, [5.0, 5.0]);
    }

    #[test]
    fn div_by_zero_is_inf() {
        let a = [1.0_f64, -1.0];
        let b = [0.0_f64; 2];
        let mut out = [0.0_f64; 2];
        apply(&a, &b, &mut out);
        assert!(out[0].is_infinite() && out[0] > 0.0);
        assert!(out[1].is_infinite() && out[1] < 0.0);
    }
}
