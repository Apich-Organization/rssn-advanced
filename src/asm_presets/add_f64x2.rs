//! `add_f64x2` — 2-lane packed `f64` addition.
//!
//! * `x86_64`: SSE2 `addpd xmm` (128-bit, 2 lanes; SSE2 is `x86_64` ABI baseline).
//! * `AArch64`: NEON `fadd v.2d` (NEON is mandatory on ARMv8-A).
//! * riscv64 + RVV: `vfadd.vv` with `vsetvli vl=2, e64`.
//! * fallback: scalar loop.

#![allow(unsafe_code)]

/// Adds two 2-lane `f64` arrays element-wise, writing the result to `out`.
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
                "addpd  xmm0, xmm1",
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
                "fadd v0.2d, v0.2d, v1.2d",
                "st1 {{v0.2d}}, [{out}]",
                lhs = inout(reg) lhs.as_ptr() => _,
                rhs = inout(reg) rhs.as_ptr() => _,
                out = inout(reg) out.as_mut_ptr() => _,
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
                "vfadd.vv v0, v0, v1",
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
    out[0] = lhs[0] + rhs[0];
    out[1] = lhs[1] + rhs[1];
}

/// AArch64 NEON intrinsic helper — retained for backward compatibility.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn apply_neon(lhs: &[f64; 2], rhs: &[f64; 2], out: &mut [f64; 2]) {
    use std::arch::aarch64::*;
    let a = unsafe { vld1q_f64(lhs.as_ptr()) };
    let b = unsafe { vld1q_f64(rhs.as_ptr()) };
    unsafe { vst1q_f64(out.as_mut_ptr(), vaddq_f64(a, b)) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_matches_scalar() {
        let a = [1.0_f64, 2.0];
        let b = [10.0_f64, 20.0];
        let mut out = [0.0_f64; 2];
        apply(&a, &b, &mut out);
        assert_eq!(out, [11.0, 22.0]);
    }

    #[test]
    fn add_negative() {
        let a = [-3.0_f64, 4.0];
        let b = [3.0_f64, -4.0];
        let mut out = [0.0_f64; 2];
        apply(&a, &b, &mut out);
        assert_eq!(out, [0.0, 0.0]);
    }
}
