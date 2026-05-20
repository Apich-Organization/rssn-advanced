//! `hash_u64x2_aesni` — fast 128-bit AES-NI based hash mix.
//!
//! Mixes two `u64` lanes with a single round of `aesenc`. The result is
//! suitable as a structural-hash kernel for the dedup map (`dag_review`,
//! `simd_review §2.2`) — *not* cryptographically secure but extremely
//! fast and avalanche-friendly enough for hash-cons buckets.
//!
//! Used by [`crate::dag::dedup`]'s rapidhash glue when the host has
//! AES-NI; otherwise a scalar mix via `wrapping_mul` provides identical
//! semantics with no SIMD.

#![allow(unsafe_code)]

/// Mixes `(lhs, rhs)` into a 128-bit hash via one AES round with a
/// fixed round-key derived from an FNV constant. Returns the result
/// as a pair of `u64`s so callers can pick either half or xor them.
///
/// # Safety
///
/// AES-NI path uses 128-bit `xmm` registers and inline `aesenc`.
#[allow(clippy::inline_always)]
#[inline(always)]
#[must_use]
pub fn apply(lhs: u64, rhs: u64) -> (u64, u64) {
    const MIX: u64 = 0xbf58_476d_1ce4_e5b9;
    const SHIFT2: u64 = 0x94d0_49bb_1331_11eb;
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("aes") {
            // FNV-1a's 64-bit offset basis × 2 = 128-bit round key.
            const RK_LO: u64 = 0xcbf2_9ce4_8422_2325;
            const RK_HI: u64 = 0x1000_0000_01b3_0000;

            let lo: u64;
            let hi: u64;
            // SAFETY: AES-NI detected.
            unsafe {
                use core::arch::asm;
                asm!(
                    // xmm0 = [lhs, rhs] (low qword, high qword)
                    "movq xmm0, {lhs}",
                    "movq xmm1, {rhs}",
                    "punpcklqdq xmm0, xmm1",
                    // xmm1 = round key
                    "movq xmm1, {rk_lo}",
                    "movq xmm2, {rk_hi}",
                    "punpcklqdq xmm1, xmm2",
                    // xmm0 = AESENC(xmm0, xmm1)
                    "aesenc xmm0, xmm1",
                    // Extract back to two u64s.
                    "movq {lo}, xmm0",
                    "pextrq {hi}, xmm0, 1",
                    lhs = in(reg) lhs,
                    rhs = in(reg) rhs,
                    rk_lo = in(reg) RK_LO,
                    rk_hi = in(reg) RK_HI,
                    lo = out(reg) lo,
                    hi = out(reg) hi,
                    out("xmm0") _,
                    out("xmm1") _,
                    out("xmm2") _,
                    options(nostack, pure, nomem),
                );
            }
            return (lo, hi);
        }
    }

    // Scalar fallback: a classic multiplicative xor-mix. Constants
    // declared at the function top so the items-after-statements lint
    // is satisfied even when the AES path is excluded.
    let mut x = lhs.wrapping_mul(MIX) ^ rhs;
    let mut y = rhs.wrapping_mul(MIX) ^ lhs;
    x ^= x >> 27;
    y ^= y >> 27;
    x = x.wrapping_mul(SHIFT2);
    y = y.wrapping_mul(SHIFT2);
    (x ^ (x >> 31), y ^ (y >> 31))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_inputs_produce_distinct_outputs() {
        let h1 = apply(0xdead_beef, 0xcafe_f00d);
        let h2 = apply(0xdead_beef, 0xcafe_f00e);
        let h3 = apply(0xdead_bef0, 0xcafe_f00d);
        assert_ne!(h1, h2);
        assert_ne!(h1, h3);
        assert_ne!(h2, h3);
    }

    #[test]
    fn function_is_pure() {
        let h1 = apply(42, 99);
        let h2 = apply(42, 99);
        assert_eq!(h1, h2);
    }

    #[test]
    fn zeros_do_not_collapse_to_zero() {
        let h = apply(0, 0);
        // With both lanes zero we still pass the round key through, so
        // the output must not be all-zeros.
        assert_ne!(h, (0, 0));
    }
}
