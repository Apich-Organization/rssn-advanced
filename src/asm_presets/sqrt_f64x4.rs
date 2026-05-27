//! `sqrt_f64x4` — packed `f64x4` square root (unary).
//!
//! * `x86_64` + AVX2: `vsqrtpd ymm` (256-bit, 4 lanes at once).
//! * `AArch64`: two `fsqrt v.2d` NEON ops (NEON is mandatory on ARMv8-A).
//! * riscv64 + RVV: `vfsqrt.v` with `vsetvli` for 4×f64.
//! * fallback: `f64::sqrt()` per lane.
//!
//! `sqrt(-x) → NaN` per IEEE-754; no special-casing is performed here.

#![allow(unsafe_code)]

/// Computes element-wise square root of 4 `f64` values, writing results to
/// `out`. Both slices must have length 4.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn apply(inp: &[f64], out: &mut [f64]) {
    debug_assert!(
        inp.len() == 4 && out.len() == 4,
        "sqrt_f64x4::apply requires exactly 4-element slices \
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
            // vsqrtpd is part of AVX (not AVX2-specific), but guarding on
            // avx2 implies avx, so this is safe.
            unsafe {
                use core::arch::asm;
                asm!(
                    "vmovupd ymm0, ymmword ptr [{inp}]",
                    "vsqrtpd ymm0, ymm0",
                    "vmovupd ymmword ptr [{out}], ymm0",
                    inp = in(reg) inp.as_ptr(),
                    out = in(reg) out.as_mut_ptr(),
                    out("ymm0") _,
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
                "fabs v0.2d, v0.2d",
                "fabs v1.2d, v1.2d",
                "st1 {{v0.2d}}, [{out}], #16",
                "st1 {{v1.2d}}, [{out}]",
                inp = inout(reg) inp_ptr,
                out = inout(reg) out_ptr,
                out("v0") _,
                out("v1") _,
                options(nostack, preserves_flags),
            );
        }
        return;
    }

    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        // SAFETY: lengths checked above; RVV activated via target_feature = "v".
        unsafe {
            use core::arch::asm;
            asm!(
                "li t0, 4",
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
    for i in 0..4 {
        out[i] = inp[i].sqrt();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqrt_known_values() {
        let a = [0.0_f64, 1.0, 4.0, 9.0];
        let mut out = [0.0_f64; 4];
        apply(&a, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-12);
        assert!((out[1] - 1.0).abs() < 1e-12);
        assert!((out[2] - 2.0).abs() < 1e-12);
        assert!((out[3] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn sqrt_negative_is_nan() {
        let a = [-1.0_f64, -4.0, 0.0, 16.0];
        let mut out = [0.0_f64; 4];
        apply(&a, &mut out);
        assert!(out[0].is_nan());
        assert!(out[1].is_nan());
        assert!((out[2] - 0.0).abs() < 1e-12);
        assert!((out[3] - 4.0).abs() < 1e-12);
    }
}
