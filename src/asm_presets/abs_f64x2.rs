//! `abs_f64x2` — 2-lane packed `f64` absolute value.
//!
//! * `x86_64`: SSE2 `andpd xmm` with sign-clear mask (128-bit, 2 lanes;
//!   SSE2 is `x86_64` ABI baseline).
//! * `AArch64`: NEON `fabs v.2d` (NEON is mandatory on ARMv8-A).
//! * riscv64 + RVV: `vfsgnjx.vv` (sign XOR self = 0) with `vsetvli vl=2, e64`.
//! * fallback: `f64::abs()` per lane.
//!
//! IEEE-754: `abs(NaN) → NaN` with sign bit cleared.

#![allow(unsafe_code)]

/// Bitmask that clears the IEEE-754 sign bit in each 64-bit lane.
#[cfg(target_arch = "x86_64")]
static ABS_MASK_F64X2: [u64; 2] = [0x7FFF_FFFF_FFFF_FFFF; 2];

/// Computes the absolute value of 2 `f64` values element-wise, writing results to `out`.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn apply(inp: &[f64; 2], out: &mut [f64; 2]) {
    #[cfg(target_arch = "x86_64")]
    {
        // SSE2 is part of the x86_64 ABI baseline — no runtime detection needed.
        // AND with the abs mask clears bit 63 of every lane in one instruction.
        unsafe {
            use core::arch::asm;
            asm!(
                "movupd xmm0, xmmword ptr [{inp}]",
                "movupd xmm1, xmmword ptr [{mask}]",
                "andpd  xmm0, xmm1",
                "movupd xmmword ptr [{out}], xmm0",
                inp  = in(reg) inp.as_ptr(),
                mask = in(reg) ABS_MASK_F64X2.as_ptr(),
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
                "fabs v0.2d, v0.2d",
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
        // `vfsgnjx.vv vd, vs2, vs1`: result sign = XOR(sign(vs2), sign(vs1)).
        // Self-applied (vs1 = vs2 = v0): sign = XOR(s, s) = 0 → |v0|.
        unsafe {
            use core::arch::asm;
            asm!(
                "li t0, 2",
                "vsetvli t0, t0, e64, m1, ta, ma",
                "vle64.v v0, ({inp})",
                "vfsgnjx.vv v0, v0, v0",
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
    out[0] = inp[0].abs();
    out[1] = inp[1].abs();
}

/// AArch64 NEON intrinsic helper — retained for backward compatibility.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn apply_neon(inp: &[f64; 2], out: &mut [f64; 2]) {
    use std::arch::aarch64::*;
    let a = unsafe { vld1q_f64(inp.as_ptr()) };
    unsafe { vst1q_f64(out.as_mut_ptr(), vabsq_f64(a)) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abs_known_values() {
        let a = [-3.0_f64, 4.0];
        let mut out = [0.0_f64; 2];
        apply(&a, &mut out);
        assert_eq!(out[0], 3.0);
        assert_eq!(out[1], 4.0);
    }

    #[test]
    fn abs_nan_and_inf() {
        let a = [f64::NAN, f64::NEG_INFINITY];
        let mut out = [0.0_f64; 2];
        apply(&a, &mut out);
        assert!(out[0].is_nan());
        assert!(out[1].is_infinite() && out[1] > 0.0);
    }

    #[test]
    fn abs_negative_zero() {
        let a = [-0.0_f64, 0.0_f64];
        let mut out = [0.0_f64; 2];
        apply(&a, &mut out);
        // abs(-0.0) = 0.0 and abs(0.0) = 0.0; both compare == 0.
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
    }
}
