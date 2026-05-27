//! `neg_f64x4` — packed `f64x4` negation (unary).
//!
//! * `x86_64` + AVX2: XOR with sign-bit mask (`vxorpd ymm`) — flips bit 63
//!   of every lane with no intermediate register needed.
//! * `AArch64`: two `fneg v.2d` NEON ops (NEON is mandatory on ARMv8-A).
//! * riscv64 + RVV: `vfsgnjn.vv v0, v0, v0` (sign-inject negated self).
//! * fallback: unary minus per lane.
//!
//! IEEE-754 negation: the sign bit is flipped, magnitude is unchanged.
//! `neg(NaN) → NaN` (different sign bit but still NaN).

#![allow(unsafe_code)]

/// Sign-bit mask for f64: sets bit 63 (the sign bit) of each 64-bit lane.
// repr(C) alignment is at least 8 bytes from the u64 type; AVX2 vandpd/vxorpd
// with a ymm register operand has no alignment requirement.
#[cfg(target_arch = "x86_64")]
static SIGN_MASK_F64X4: [u64; 4] = [0x8000_0000_0000_0000; 4];

/// Negates 4 `f64` values element-wise, writing results to `out`.
/// Both slices must have length 4.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn apply(inp: &[f64], out: &mut [f64]) {
    debug_assert!(
        inp.len() == 4 && out.len() == 4,
        "neg_f64x4::apply requires exactly 4-element slices \
         (got inp={}, out={})",
        inp.len(),
        out.len()
    );
    if inp.len() != 4 || out.len() != 4 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: lengths checked above; AVX2 detected.
            // Load sign mask into ymm1 via vmovupd (handles any alignment),
            // then XOR with input — one instruction to flip all sign bits.
            unsafe {
                use core::arch::asm;
                asm!(
                    "vmovupd ymm0, ymmword ptr [{inp}]",
                    "vmovupd ymm1, ymmword ptr [{mask}]",
                    "vxorpd  ymm0, ymm0, ymm1",
                    "vmovupd ymmword ptr [{out}], ymm0",
                    inp  = in(reg) inp.as_ptr(),
                    mask = in(reg) SIGN_MASK_F64X4.as_ptr(),
                    out  = in(reg) out.as_mut_ptr(),
                    out("ymm0") _,
                    out("ymm1") _,
                    options(nostack, preserves_flags),
                );
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: lengths checked above; NEON is mandatory on AArch64.
        unsafe {
            use core::arch::asm;
            let mut inp_ptr = inp.as_ptr();
            let mut out_ptr = out.as_mut_ptr();
            asm!(
                "ld1 {{v0.2d}}, [{inp}], #16",
                "ld1 {{v1.2d}}, [{inp}]",
                "fneg v0.2d, v0.2d",
                "fneg v1.2d, v1.2d",
                "st1 {{v0.2d}}, [{out}], #16",
                "st1 {{v1.2d}}, [{out}]",
                inp = inout(reg) inp_ptr,
                out = inout(reg) out_ptr,
                out("v0") _, out("v1") _,
                options(nostack, preserves_flags),
            );
        }
        return;
    }

    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        // SAFETY: lengths checked above; RVV activated via target_feature = "v".
        // vfsgnjn.vv vd, vs2, vs1: result = |vs2| with sign = NOT sign(vs1).
        // Self-applied: |v0| with sign = NOT sign(v0) = -v0.
        unsafe {
            use core::arch::asm;
            asm!(
                "li t0, 4",
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
    for i in 0..4 {
        out[i] = -inp[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neg_flips_sign() {
        let a = [1.0_f64, -2.0, 0.0, -0.0];
        let mut out = [0.0_f64; 4];
        apply(&a, &mut out);
        assert_eq!(out[0], -1.0);
        assert_eq!(out[1], 2.0);
        // IEEE-754: neg(0.0) = -0.0, neg(-0.0) = 0.0 — both == 0.0 by ==.
        assert_eq!(out[2] + out[3], 0.0);
    }

    #[test]
    fn neg_nan_is_nan() {
        let a = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 3.14];
        let mut out = [0.0_f64; 4];
        apply(&a, &mut out);
        assert!(out[0].is_nan());
        assert!(out[1].is_infinite() && out[1] < 0.0);
        assert!(out[2].is_infinite() && out[2] > 0.0);
        assert!((out[3] + 3.14).abs() < 1e-12);
    }
}
