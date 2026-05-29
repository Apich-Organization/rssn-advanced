//! `neg_f64x2` — 2-lane packed `f64` negation.
//!
//! * `x86_64`: SSE2 `xorpd xmm` with sign-bit mask (128-bit, 2 lanes;
//!   SSE2 is `x86_64` ABI baseline).
//! * `AArch64`: NEON `fneg v.2d` (NEON is mandatory on ARMv8-A).
//! * riscv64 + RVV: `vfsgnjn.vv` (sign-inject negated self) with `vsetvli vl=2, e64`.
//! * fallback: unary minus per lane.
//!
//! IEEE-754 negation: sign bit is flipped, magnitude unchanged. `neg(NaN) → NaN`.

#![allow(unsafe_code)]

/// Sign-bit mask for 2-lane f64: sets bit 63 of each 64-bit lane.
#[cfg(target_arch = "x86_64")]
static SIGN_MASK_F64X2: [u64; 2] = [0x8000_0000_0000_0000; 2];

/// Negates 2 `f64` values element-wise, writing results to `out`.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn apply(inp: &[f64; 2], out: &mut [f64; 2]) {
    #[cfg(target_arch = "x86_64")]
    {
        // SSE2 is part of the x86_64 ABI baseline — no runtime detection needed.
        // XOR with the sign-bit mask flips bit 63 of every lane in one instruction.
        unsafe {
            use core::arch::asm;
            asm!(
                "movupd xmm0, xmmword ptr [{inp}]",
                "movupd xmm1, xmmword ptr [{mask}]",
                "xorpd  xmm0, xmm1",
                "movupd xmmword ptr [{out}], xmm0",
                inp  = in(reg) inp.as_ptr(),
                mask = in(reg) SIGN_MASK_F64X2.as_ptr(),
                out  = in(reg) out.as_mut_ptr(),
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
                "ld1 {{v0.2d}}, [{inp}]",
                "fneg v0.2d, v0.2d",
                "st1 {{v0.2d}}, [{out}]",
                inp = inout(reg) inp.as_ptr() => _,
                out = inout(reg) out.as_mut_ptr() => _,
                out("v0") _,
                options(nostack, preserves_flags),
            );
        }
        return;
    }

    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        // RVV 1.0 with vl=2, e64.
        // `vfsgnjn.vv vd, vs2, vs1`: result magnitude = |vs2|, sign = NOT sign(vs1).
        // Self-applied (vs1 = vs2 = v0): -v0.
        unsafe {
            use core::arch::asm;
            asm!(
                "li t0, 2",
                "vsetvli t0, t0, e64, m1, ta, ma",
                "vle64.v v0, ({inp})",
                "vfsgnjn.vv v0, v0, v0",
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
    out[0] = -inp[0];
    out[1] = -inp[1];
}

/// AArch64 NEON intrinsic helper — retained for backward compatibility.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn apply_neon(inp: &[f64; 2], out: &mut [f64; 2]) {
    use std::arch::aarch64::*;
    let a = unsafe { vld1q_f64(inp.as_ptr()) };
    unsafe { vst1q_f64(out.as_mut_ptr(), vnegq_f64(a)) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neg_flips_sign() {
        let a = [1.0_f64, -2.0];
        let mut out = [0.0_f64; 2];
        apply(&a, &mut out);
        assert_eq!(out[0], -1.0);
        assert_eq!(out[1], 2.0);
    }

    #[test]
    fn neg_nan_is_nan() {
        let a = [f64::NAN, f64::INFINITY];
        let mut out = [0.0_f64; 2];
        apply(&a, &mut out);
        assert!(out[0].is_nan());
        assert!(out[1].is_infinite() && out[1] < 0.0);
    }

    #[test]
    fn neg_zero() {
        // IEEE-754: neg(0.0) = -0.0; neg(-0.0) = 0.0; both compare == 0.0.
        let a = [0.0_f64, -0.0_f64];
        let mut out = [0.0_f64; 2];
        apply(&a, &mut out);
        assert_eq!(out[0] + out[1], 0.0);
    }
}
