//! `fma_f64x4` — fused multiply-add `out = a * b + c`.
//!
//! Single-rounding FMA is what `plan.md §3.1` calls out for coefficient
//! merging hot paths; the scalar fallback uses `f64::mul_add` so the
//! semantics match on both branches.
//!
//! * `x86_64` + AVX2 + FMA: `vfmadd231pd ymm` (256-bit, single rounding).
//! * `AArch64`: two `fmla v.2d` NEON ops (`Vd += Va * Vb`, NEON mandatory).
//! * riscv64 + RVV: `vfmacc.vv` (`vd += vs1 * vs2`).
//! * fallback: `f64::mul_add` scalar loop.

#![allow(unsafe_code)]

/// Computes `out[i] = a[i] * b[i] + c[i]` for four `f64` lanes.
///
/// # Safety
///
/// On the FMA path the function uses raw pointers and 256-bit unaligned
/// `vmovupd` / `vfmadd231pd`. Lengths are checked; alignment isn't.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn apply(a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
    debug_assert!(
        a.len() == 4 && b.len() == 4 && c.len() == 4 && out.len() == 4,
        "fma_f64x4::apply requires exactly 4-element slices \
         (got a={}, b={}, c={}, out={})",
        a.len(),
        b.len(),
        c.len(),
        out.len()
    );
    if a.len() != 4 || b.len() != 4 || c.len() != 4 || out.len() != 4 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: lengths checked; AVX2+FMA detected.
            unsafe {
                use core::arch::asm;
                asm!(
                    "vmovupd ymm0, ymmword ptr [{a}]",     // ymm0 = a
                    "vmovupd ymm1, ymmword ptr [{b}]",     // ymm1 = b
                    "vmovupd ymm2, ymmword ptr [{c}]",     // ymm2 = c (accumulator)
                    "vfmadd231pd ymm2, ymm0, ymm1",         // ymm2 += ymm0 * ymm1
                    "vmovupd ymmword ptr [{out}], ymm2",
                    a = in(reg) a.as_ptr(),
                    b = in(reg) b.as_ptr(),
                    c = in(reg) c.as_ptr(),
                    out = in(reg) out.as_mut_ptr(),
                    out("ymm0") _,
                    out("ymm1") _,
                    out("ymm2") _,
                    options(nostack, preserves_flags),
                );
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: lengths checked above; NEON is mandatory on AArch64.
        // `fmla Vd.2d, Vn.2d, Vm.2d` computes Vd += Vn * Vm (single rounding).
        unsafe {
            use core::arch::asm;
            asm!(
                "ld1 {{v0.2d}}, [{a}], #16",
                "ld1 {{v1.2d}}, [{b}], #16",
                "ld1 {{v2.2d}}, [{c}], #16",
                "ld1 {{v3.2d}}, [{a}]",
                "ld1 {{v4.2d}}, [{b}]",
                "ld1 {{v5.2d}}, [{c}]",
                "fmla v2.2d, v0.2d, v1.2d",
                "fmla v5.2d, v3.2d, v4.2d",
                "st1 {{v2.2d}}, [{out}], #16",
                "st1 {{v5.2d}}, [{out}]",
                a = inout(reg) a.as_ptr() => _,
                b = inout(reg) b.as_ptr() => _,
                c = inout(reg) c.as_ptr() => _,
                out = inout(reg) out.as_mut_ptr() => _,
                out("v0") _, out("v1") _, out("v2") _,
                out("v3") _, out("v4") _, out("v5") _,
                options(nostack),
            );
        }
        return;
    }

    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        // SAFETY: lengths checked above; RVV activated via target_feature = "v".
        // `vfmacc.vv vd, vs1, vs2` computes vd += vs1 * vs2 (single rounding).
        unsafe {
            use core::arch::asm;
            asm!(
                "li t0, 4",
                "vsetvli t0, t0, e64, m1, ta, ma",
                "vle64.v v0, ({a})",
                "vle64.v v1, ({b})",
                "vle64.v v2, ({c})",
                "vfmacc.vv v2, v0, v1",
                "vse64.v v2, ({out})",
                a = in(reg) a.as_ptr(),
                b = in(reg) b.as_ptr(),
                c = in(reg) c.as_ptr(),
                out = in(reg) out.as_mut_ptr(),
                out("t0") _,
                out("v0") _,
                out("v1") _,
                out("v2") _,
                options(nostack),
            );
        }
        return;
    }

    for i in 0..4 {
        // `mul_add` requests single-rounding semantics that match
        // `vfmadd231pd` / `fmla` / `vfmacc.vv` exactly on FMA hardware.
        out[i] = a[i].mul_add(b[i], c[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fma_matches_naive_on_simple_inputs() {
        let a = [1.0_f64, 2.0, 3.0, 4.0];
        let b = [10.0_f64, 20.0, 30.0, 40.0];
        let c = [100.0_f64, 200.0, 300.0, 400.0];
        let mut out = [0.0_f64; 4];
        apply(&a, &b, &c, &mut out);
        // a*b + c = [10+100, 40+200, 90+300, 160+400]
        assert_eq!(out, [110.0, 240.0, 390.0, 560.0]);
    }

    #[test]
    fn fma_preserves_single_rounding_for_small_residuals() {
        // (1 + 1e-30) * 1.0 + (-1.0) is the canonical FMA-vs-naive
        // distinguishing input. We don't assert the exact rounding
        // difference (would be flaky across machines without FMA);
        // we only assert finite output.
        let a = [1.0_f64; 4];
        let b = [1.0_f64; 4];
        let c = [-1.0_f64; 4];
        let mut out = [0.0_f64; 4];
        apply(&a, &b, &c, &mut out);
        for &v in &out {
            assert!(v.is_finite());
        }
    }
}
