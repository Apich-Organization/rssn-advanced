//! `add_f64x4_avx2` — packed `f64x4` addition via `vaddpd`.
//!
//! Emits a single 256-bit AVX2 add when the host supports AVX2;
//! otherwise drops into a scalar fallback.

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
                    options(nostack),
                );
            }
            return;
        }
    }

    // Scalar fallback (also used on non-x86_64).
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
    fn mismatched_lengths_are_no_op() {
        let a = [1.0_f64, 2.0];
        let b = [10.0_f64, 20.0, 30.0, 40.0];
        let mut out = [0.0_f64; 4];
        apply(&a, &b, &mut out);
        assert_eq!(out, [0.0; 4], "no-op when lengths disagree");
    }
}
