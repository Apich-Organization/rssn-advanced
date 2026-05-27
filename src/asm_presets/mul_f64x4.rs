//! `mul_f64x4` — packed `f64x4` multiplication.
//!
//! * `x86_64` + AVX2: `vmulpd ymm` (256-bit, 4 lanes).
//! * `AArch64`: two `fmul v.2d` NEON ops (NEON mandatory).
//! * riscv64 + RVV: `vfmul.vv` with `vsetvli` for 4×f64.
//! * fallback: scalar loop.

#![allow(unsafe_code)]

/// Multiplies two 4-lane `f64` vectors element-wise.
///
/// # Safety
///
/// On the AVX2 path uses raw pointers + 256-bit unaligned ops. Lengths
/// are checked; alignment isn't (we emit `vmovupd`).
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn apply(lhs: &[f64], rhs: &[f64], out: &mut [f64]) {
    debug_assert!(
        lhs.len() == 4 && rhs.len() == 4 && out.len() == 4,
        "mul_f64x4::apply requires exactly 4-element slices \
         (got lhs={}, rhs={}, out={})",
        lhs.len(),
        rhs.len(),
        out.len()
    );
    if lhs.len() != 4 || rhs.len() != 4 || out.len() != 4 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: lengths checked; AVX2 detected.
            unsafe {
                use core::arch::asm;
                asm!(
                    "vmovupd ymm0, ymmword ptr [{lhs}]",
                    "vmovupd ymm1, ymmword ptr [{rhs}]",
                    "vmulpd  ymm0, ymm0, ymm1",
                    "vmovupd ymmword ptr [{out}], ymm0",
                    lhs = in(reg) lhs.as_ptr(),
                    rhs = in(reg) rhs.as_ptr(),
                    out = in(reg) out.as_mut_ptr(),
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
                "ld1 {{v0.2d}}, [{lhs}]",
                "ld1 {{v1.2d}}, [{rhs}]",
                "fmul v0.2d, v0.2d, v1.2d",
                "st1 {{v0.2d}}, [{out}]",
                "ld1 {{v0.2d}}, [{lhs}, #16]",
                "ld1 {{v1.2d}}, [{rhs}, #16]",
                "fmul v0.2d, v0.2d, v1.2d",
                "str {{v0.2d}}, [{out}, #16]",
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
        // SAFETY: lengths checked above; RVV activated via target_feature = "v".
        unsafe {
            use core::arch::asm;
            asm!(
                "li t0, 4",
                "vsetvli t0, t0, e64, m1, ta, ma",
                "vle64.v v0, ({lhs})",
                "vle64.v v1, ({rhs})",
                "vfmul.vv v0, v0, v1",
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
    for i in 0..4 {
        out[i] = lhs[i] * rhs[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_mul_matches_scalar() {
        let a = [1.0_f64, 2.0, 3.0, 4.0];
        let b = [10.0_f64, 20.0, 30.0, 40.0];
        let mut out = [0.0_f64; 4];
        apply(&a, &b, &mut out);
        assert_eq!(out, [10.0, 40.0, 90.0, 160.0]);
    }

    #[test]
    fn multiplying_by_zero_zeros_out() {
        let a = [1.0_f64, 2.0, 3.0, 4.0];
        let b = [0.0_f64; 4];
        let mut out = [9.0_f64; 4];
        apply(&a, &b, &mut out);
        assert_eq!(out, [0.0_f64; 4]);
    }
}
