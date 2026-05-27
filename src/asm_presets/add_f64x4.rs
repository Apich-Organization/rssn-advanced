//! `add_f64x4` — packed `f64x4` addition.
//!
//! * `x86_64` + AVX2: single `vaddpd ymm` (256-bit, 4 lanes at once).
//! * `AArch64`: two `fadd v.2d` NEON ops (128-bit each; NEON is mandatory).
//! * riscv64 + RVV: `vfadd.vv` with `vsetvli` for 4×f64.
//! * fallback: scalar loop.

#![allow(unsafe_code)]

/// Adds two 4-lane `f64` vectors element-wise, writing the result to
/// `out`. All three slices must have length 4; otherwise the call
/// returns without doing anything to avoid panicking in the hot path.
///
/// # Safety
///
/// On the AVX2 path the function dereferences raw pointers and uses
/// 256-bit unaligned loads/stores. The provided slices are checked for
/// length but not alignment; misaligned data is fine because the
/// emitted `vmovupd` instruction handles it.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn apply(lhs: &[f64], rhs: &[f64], out: &mut [f64]) {
    debug_assert!(
        lhs.len() == 4 && rhs.len() == 4 && out.len() == 4,
        "add_f64x4::apply requires exactly 4-element slices \
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
            // SAFETY: lengths checked above; AVX2 detected.
            unsafe {
                use core::arch::asm;
                asm!(
                    "vmovupd ymm0, ymmword ptr [{lhs}]",
                    "vmovupd ymm1, ymmword ptr [{rhs}]",
                    "vaddpd  ymm0, ymm0, ymm1",
                    "vmovupd ymmword ptr [{out}], ymm0",
                    lhs = in(reg) lhs.as_ptr(),
                    rhs = in(reg) rhs.as_ptr(),
                    out = in(reg) out.as_mut_ptr(),
                    out("ymm0") _,
                    out("ymm1") _,
                    // vaddpd/vmovupd do not modify integer EFLAGS; declaring
                    // preserves_flags lets the compiler skip EFLAGS save/restore
                    // around this asm block, reducing call overhead (simd_review §2.2).
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
                "fadd v0.2d, v0.2d, v1.2d",
                "st1 {{v0.2d}}, [{out}]",
                "ld1 {{v0.2d}}, [{lhs}, #16]",
                "ld1 {{v1.2d}}, [{rhs}, #16]",
                "fadd v0.2d, v0.2d, v1.2d",
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
    for i in 0..4 {
        out[i] = lhs[i] + rhs[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_add_matches_scalar() {
        let a = [1.0_f64, 2.0, 3.0, 4.0];
        let b = [10.0_f64, 20.0, 30.0, 40.0];
        let mut out = [0.0_f64; 4];
        apply(&a, &b, &mut out);
        assert_eq!(out, [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn mismatched_lengths_are_no_op_in_release() {
        // In debug builds `debug_assert!` panics on length mismatch; in
        // release builds the kernel silently no-ops. The SIMD arithmetic
        // wrappers in `simd::arithmetic` always pass correctly-sized
        // 4-element chunks, so mismatched lengths indicate a caller bug.
        let a = [1.0_f64, 2.0];
        let b = [10.0_f64, 20.0, 30.0, 40.0];
        let mut out = [0.0_f64; 4];
        apply(&a, &b, &mut out);
        assert_eq!(out, [0.0; 4], "no-op when lengths disagree (release only)");
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "requires exactly 4-element slices")]
    fn mismatched_lengths_panic_in_debug() {
        let a = [1.0_f64, 2.0];
        let b = [10.0_f64, 20.0, 30.0, 40.0];
        let mut out = [0.0_f64; 4];
        apply(&a, &b, &mut out);
    }
}
