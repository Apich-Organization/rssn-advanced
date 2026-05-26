//! `abs_f64x4` — packed `f64x4` absolute value (unary).
//!
//! * `x86_64` + AVX2: AND with `0x7FFF_FFFF_FFFF_FFFF` mask (`vandpd ymm`) —
//!   clears bit 63 (the sign bit) in all 4 lanes.
//! * `AArch64`: two `fabs v.2d` NEON ops (NEON is mandatory on ARMv8-A).
//! * riscv64 + RVV: `vfsgnjx.vv v0, v0, v0` (sign-inject self XOR self = 0).
//! * fallback: `f64::abs()` per lane.
//!
//! IEEE-754: `abs(NaN) → NaN` with sign bit cleared (quiet NaN preserved).

#![allow(unsafe_code)]

/// Bitmask that clears the IEEE-754 sign bit in each 64-bit lane.
#[cfg(target_arch = "x86_64")]
static ABS_MASK_F64X4: [u64; 4] = [0x7FFF_FFFF_FFFF_FFFF; 4];

/// Computes the absolute value of 4 `f64` values element-wise, writing
/// results to `out`. Both slices must have length 4.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn apply(inp: &[f64], out: &mut [f64]) {
    debug_assert!(
        inp.len() == 4 && out.len() == 4,
        "abs_f64x4::apply requires exactly 4-element slices \
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
            // Load abs mask into ymm1 via vmovupd (handles any alignment),
            // then AND with input — clears sign bits in one instruction.
            unsafe {
                use core::arch::asm;
                asm!(
                    "vmovupd ymm0, ymmword ptr [{inp}]",
                    "vmovupd ymm1, ymmword ptr [{mask}]",
                    "vandpd  ymm0, ymm0, ymm1",
                    "vmovupd ymmword ptr [{out}], ymm0",
                    inp  = in(reg) inp.as_ptr(),
                    mask = in(reg) ABS_MASK_F64X4.as_ptr(),
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
            asm!(
                "ld1 {{v0.2d}}, [{inp}]",
                "fabs v0.2d, v0.2d",
                "st1 {{v0.2d}}, [{out}]",
                "ld1 {{v0.2d}}, [{inp}, #16]",
                "fabs v0.2d, v0.2d",
                "st1 {{v0.2d}}, [{out}, #16]",
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
        // SAFETY: lengths checked above; RVV activated via target_feature = "v".
        // vfsgnjx.vv vd, vs2, vs1: result = |vs2| with sign = XOR(sign(vs2), sign(vs1)).
        // Self-applied: |v0| with sign = XOR(sign(v0), sign(v0)) = 0 = |v0|.
        unsafe {
            use core::arch::asm;
            asm!(
                "li t0, 4",
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
    for i in 0..4 {
        out[i] = inp[i].abs();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abs_known_values() {
        let a = [-3.0_f64, 4.0, -0.0, 0.0];
        let mut out = [0.0_f64; 4];
        apply(&a, &mut out);
        assert_eq!(out[0], 3.0);
        assert_eq!(out[1], 4.0);
        // abs(-0.0) = 0.0 and abs(0.0) = 0.0; both compare == 0.
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn abs_nan_and_inf() {
        let a = [f64::NAN, f64::NEG_INFINITY, -5.5, 5.5];
        let mut out = [0.0_f64; 4];
        apply(&a, &mut out);
        assert!(out[0].is_nan());
        assert!(out[1].is_infinite() && out[1] > 0.0);
        assert!((out[2] - 5.5).abs() < 1e-12);
        assert!((out[3] - 5.5).abs() < 1e-12);
    }
}
