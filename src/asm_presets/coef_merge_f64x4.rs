//! `coef_merge_f64x4` — symbolic coefficient merge kernel.
//!
//! Implements `out[i] = (c1[i] * c2[i]) * (x[i] * y[i])` over 4-lane
//! `f64` inputs in a single pass — the kernel the JIT calls when fusing
//! nested products `(c1*x)*(c2*y)` (`jit_review §1`, `plan.md §3.1`).
//!
//! * x86_64 + AVX2: three `vmulpd ymm` ops (256-bit each).
//! * AArch64: three `fmul v.2d` NEON ops × 2 128-bit rounds (NEON mandatory).
//! * riscv64 + RVV: three `vfmul.vv` ops with `vsetvli` for 4×f64.
//! * fallback: scalar loop.

#![allow(unsafe_code)]

/// Computes `out[i] = (c1[i] * c2[i]) * (x[i] * y[i])`.
///
/// # Safety
///
/// AVX2 path uses raw pointers and 256-bit unaligned ops. Lengths
/// are checked; alignment isn't.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn apply(c1: &[f64], c2: &[f64], x: &[f64], y: &[f64], out: &mut [f64]) {
    debug_assert!(
        c1.len() == 4 && c2.len() == 4 && x.len() == 4 && y.len() == 4 && out.len() == 4,
        "coef_merge_f64x4::apply requires exactly 4-element slices \
         (got c1={}, c2={}, x={}, y={}, out={})",
        c1.len(),
        c2.len(),
        x.len(),
        y.len(),
        out.len()
    );
    if c1.len() != 4 || c2.len() != 4 || x.len() != 4 || y.len() != 4 || out.len() != 4 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: lengths checked; AVX2 detected.
            unsafe {
                use core::arch::asm;
                asm!(
                    "vmovupd ymm0, ymmword ptr [{c1}]",
                    "vmovupd ymm1, ymmword ptr [{c2}]",
                    "vmovupd ymm2, ymmword ptr [{x}]",
                    "vmovupd ymm3, ymmword ptr [{y}]",
                    "vmulpd  ymm0, ymm0, ymm1",   // ymm0 = c1*c2
                    "vmulpd  ymm2, ymm2, ymm3",   // ymm2 = x*y
                    "vmulpd  ymm0, ymm0, ymm2",   // ymm0 = (c1*c2) * (x*y)
                    "vmovupd ymmword ptr [{out}], ymm0",
                    c1 = in(reg) c1.as_ptr(),
                    c2 = in(reg) c2.as_ptr(),
                    x = in(reg) x.as_ptr(),
                    y = in(reg) y.as_ptr(),
                    out = in(reg) out.as_mut_ptr(),
                    out("ymm0") _,
                    out("ymm1") _,
                    out("ymm2") _,
                    out("ymm3") _,
                    options(nostack, preserves_flags),
                );
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: lengths checked above; NEON is mandatory on AArch64.
        // Operands xs/ys alias the x/y parameters to avoid shadowing issues.
        unsafe {
            use core::arch::asm;
            asm!(
                "ld1 {{v0.2d}}, [{c1}]",
                "ld1 {{v1.2d}}, [{c2}]",
                "ld1 {{v2.2d}}, [{xs}]",
                "ld1 {{v3.2d}}, [{ys}]",
                "fmul v0.2d, v0.2d, v1.2d",
                "fmul v2.2d, v2.2d, v3.2d",
                "fmul v0.2d, v0.2d, v2.2d",
                "st1 {{v0.2d}}, [{out}]",
                "ld1 {{v0.2d}}, [{c1}, #16]",
                "ld1 {{v1.2d}}, [{c2}, #16]",
                "ld1 {{v2.2d}}, [{xs}, #16]",
                "ld1 {{v3.2d}}, [{ys}, #16]",
                "fmul v0.2d, v0.2d, v1.2d",
                "fmul v2.2d, v2.2d, v3.2d",
                "fmul v0.2d, v0.2d, v2.2d",
                "st1 {{v0.2d}}, [{out}, #16]",
                c1 = in(reg) c1.as_ptr(),
                c2 = in(reg) c2.as_ptr(),
                xs = in(reg) x.as_ptr(),
                ys = in(reg) y.as_ptr(),
                out = in(reg) out.as_mut_ptr(),
                out("v0") _,
                out("v1") _,
                out("v2") _,
                out("v3") _,
                options(nostack),
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
                "vle64.v v0, ({c1})",
                "vle64.v v1, ({c2})",
                "vle64.v v2, ({xs})",
                "vle64.v v3, ({ys})",
                "vfmul.vv v0, v0, v1",
                "vfmul.vv v2, v2, v3",
                "vfmul.vv v0, v0, v2",
                "vse64.v v0, ({out})",
                c1 = in(reg) c1.as_ptr(),
                c2 = in(reg) c2.as_ptr(),
                xs = in(reg) x.as_ptr(),
                ys = in(reg) y.as_ptr(),
                out = in(reg) out.as_mut_ptr(),
                out("t0") _,
                out("v0") _,
                out("v1") _,
                out("v2") _,
                out("v3") _,
                options(nostack),
            );
        }
        return;
    }

    // Scalar fallback.
    for i in 0..4 {
        out[i] = (c1[i] * c2[i]) * (x[i] * y[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_matches_naive() {
        let c1 = [2.0_f64, 3.0, 4.0, 5.0];
        let c2 = [1.5_f64, 2.5, 3.5, 4.5];
        let x = [10.0_f64, 20.0, 30.0, 40.0];
        let y = [0.5_f64, 1.0, 1.5, 2.0];
        let mut out = [0.0_f64; 4];
        apply(&c1, &c2, &x, &y, &mut out);
        for i in 0..4 {
            let expected = (c1[i] * c2[i]) * (x[i] * y[i]);
            assert!((out[i] - expected).abs() < 1e-12);
        }
    }
}
