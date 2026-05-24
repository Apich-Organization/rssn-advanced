//! `cmp_eq_f64x4` — packed `f64x4` equality test.
//!
//! Returns a 4-lane boolean mask: `0xFF` per lane if equal, `0x00`
//! otherwise. `NaN == NaN` is false (IEEE 754), matching Rust `==`.
//!
//! * x86_64 + AVX2: `vcmpeqpd ymm` + `vmovmskpd` to collapse sign bits.
//! * AArch64: two `fcmeq v.2d` + `umov` to extract lane masks (NEON mandatory).
//! * riscv64: scalar (RVV predicate-mask expansion is overly complex for 4 lanes).
//! * fallback: scalar `==` loop.

#![allow(unsafe_code)]

/// Compares `lhs` and `rhs` element-wise, writing `0xFF`/`0x00` into
/// `mask` per lane.
///
/// # Safety
///
/// AVX2 path uses raw pointers and 256-bit unaligned ops, plus a
/// `vmovmskpd` to extract the per-lane sign-bit mask. Lengths are
/// checked; alignment isn't.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn apply(lhs: &[f64], rhs: &[f64], mask: &mut [u8]) {
    debug_assert!(
        lhs.len() == 4 && rhs.len() == 4 && mask.len() == 4,
        "cmp_eq_f64x4::apply requires exactly 4-element slices \
         (got lhs={}, rhs={}, mask={})",
        lhs.len(),
        rhs.len(),
        mask.len()
    );
    if lhs.len() != 4 || rhs.len() != 4 || mask.len() != 4 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: lengths checked; AVX2 detected.
            unsafe {
                use core::arch::asm;
                // After `vcmpeqpd`, each ymm0 lane is all-ones (0xFFF…)
                // for equal, all-zeros for unequal. We then collapse it
                // to a 4-bit mask via `vmovmskpd`, and explode that back
                // into 4 bytes by hand for the caller.
                let mut packed: u32 = 0;
                asm!(
                    "vmovupd ymm0, ymmword ptr [{lhs}]",
                    "vmovupd ymm1, ymmword ptr [{rhs}]",
                    "vcmpeqpd ymm0, ymm0, ymm1",
                    "vmovmskpd {packed:e}, ymm0",
                    lhs = in(reg) lhs.as_ptr(),
                    rhs = in(reg) rhs.as_ptr(),
                    packed = out(reg) packed,
                    out("ymm0") _,
                    out("ymm1") _,
                    options(nostack, readonly),
                );
                for (i, slot) in mask.iter_mut().enumerate() {
                    *slot = if packed & (1 << i) != 0 { 0xFF } else { 0x00 };
                }
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: lengths checked above; NEON is mandatory on AArch64.
        // `fcmeq` sets each 64-bit lane to all-ones (!=0) if equal, all-zeros (0) otherwise.
        let m0: u64;
        let m1: u64;
        let m2: u64;
        let m3: u64;
        unsafe {
            use core::arch::asm;
            asm!(
                "ld1 {{v0.2d}}, [{lhs}]",
                "ld1 {{v1.2d}}, [{rhs}]",
                "fcmeq v0.2d, v0.2d, v1.2d",
                "umov {m0}, v0.d[0]",
                "umov {m1}, v0.d[1]",
                "ld1 {{v0.2d}}, [{lhs}, #16]",
                "ld1 {{v1.2d}}, [{rhs}, #16]",
                "fcmeq v0.2d, v0.2d, v1.2d",
                "umov {m2}, v0.d[0]",
                "umov {m3}, v0.d[1]",
                lhs = in(reg) lhs.as_ptr(),
                rhs = in(reg) rhs.as_ptr(),
                m0 = out(reg) m0,
                m1 = out(reg) m1,
                m2 = out(reg) m2,
                m3 = out(reg) m3,
                out("v0") _,
                out("v1") _,
                options(nostack, readonly),
            );
        }
        let lane_bits = [m0, m1, m2, m3];
        for (slot, &bits) in mask.iter_mut().zip(lane_bits.iter()) {
            *slot = if bits != 0 { 0xFF } else { 0x00 };
        }
        return;
    }

    // Bitwise equality on f64 is intentional: callers want IEEE-754
    // `==` semantics where NaN != NaN.
    #[allow(clippy::float_cmp)]
    for ((l, r), slot) in lhs.iter().zip(rhs.iter()).zip(mask.iter_mut()) {
        *slot = if *l == *r { 0xFF } else { 0x00 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_lanes_set_to_ff() {
        let a = [1.0_f64, 2.0, 3.0, 4.0];
        let b = [1.0_f64, 0.0, 3.0, 4.0];
        let mut m = [0_u8; 4];
        apply(&a, &b, &mut m);
        assert_eq!(m, [0xFF, 0x00, 0xFF, 0xFF]);
    }

    #[test]
    fn nan_compares_not_equal() {
        let nan = f64::NAN;
        let a = [nan, 0.0, nan, 1.0];
        let b = [nan, 0.0, 1.0, nan];
        let mut m = [0_u8; 4];
        apply(&a, &b, &mut m);
        assert_eq!(m[0], 0x00, "NaN == NaN must be false (IEEE 754)");
        assert_eq!(m[1], 0xFF);
        assert_eq!(m[2], 0x00);
        assert_eq!(m[3], 0x00);
    }
}
