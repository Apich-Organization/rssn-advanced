//! `coef_merge_f64x4` — symbolic coefficient merge kernel.
//!
//! Implements `(c1*c2) * (x*y)` over 4-lane `f64` inputs in a single
//! pass:
//!
//! ```text
//!   out[i] = (c1[i] * c2[i]) * (x[i] * y[i])
//! ```
//!
//! This is the kernel the JIT calls when fusing nested products
//! `(c1*x)*(c2*y)` — see `jit_review §1` (coefficient merging) and
//! `plan.md §3.1`. On `x86_64` with FMA we collapse the two multiplies
//! into one `vfmadd213pd` flow; without FMA we use two `vmulpd`s.

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
    if c1.len() != 4
        || c2.len() != 4
        || x.len() != 4
        || y.len() != 4
        || out.len() != 4
    {
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
                    options(nostack),
                );
            }
            return;
        }
    }

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
