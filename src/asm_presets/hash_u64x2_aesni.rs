//! `hash_u64x2_aesni` — fast 128-bit hardware-AES hash mix.
//!
//! Mixes two `u64` lanes via one AES encryption round. The result is
//! suitable as a structural-hash kernel for the dedup map (`dag_review`,
//! `simd_review §2.2`) — *not* cryptographically secure but extremely
//! fast and avalanche-friendly enough for hash-cons buckets.
//!
//! * `x86_64` + AES-NI: `aesenc xmm` (single round, checked at runtime).
//! * `AArch64` + crypto ext: `aese v.16b` + `aesmc v.16b`. On Apple Silicon
//!   (M1+, `target_vendor = "apple"`) AES is mandatory so the runtime check
//!   is elided at compile time; other `AArch64` hosts probe via
//!   `is_aarch64_feature_detected!`.
//! * riscv64 + Zkn: `aes64esm` scalar AES-round instructions. The RISC-V Zkn
//!   extension provides `aes64esm rd, rs1, rs2` which performs `SubBytes` +
//!   `ShiftRows` + `MixColumns` on the 128-bit state `{rs2, rs1}`, outputting
//!   one 64-bit half. Two calls (with rs1/rs2 swapped) give both halves.
//!   `AddRoundKey` is performed with a subsequent XOR.
//! * fallback: scalar multiplicative xor-mix.

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

    #[cfg(target_arch = "aarch64")]
    {
        // Apple Silicon (M1+) mandates AES as part of its ARMv8.4-A+ baseline —
        // skip the runtime sysctlbyname probe. All other AArch64 targets query
        // the OS at runtime via `is_aarch64_feature_detected!`.
        #[cfg(target_vendor = "apple")]
        let has_aes = true;
        #[cfg(not(target_vendor = "apple"))]
        let has_aes = std::arch::is_aarch64_feature_detected!("aes");

        const RK_LO: u64 = 0xcbf2_9ce4_8422_2325;
        const RK_HI: u64 = 0x1000_0000_01b3_0000;

        unsafe {
            use core::arch::aarch64::{uint64x2_t, vcombine_u64, vcreate_u64};
            use core::arch::asm;

            let state: uint64x2_t = vcombine_u64(vcreate_u64(lhs), vcreate_u64(rhs));
            let rkey: uint64x2_t = vcombine_u64(vcreate_u64(RK_LO), vcreate_u64(RK_HI));
            let out: uint64x2_t;

            if has_aes {
                asm!(
                    "aese {v_state}.16b, {v_rkey}.16b",
                    "aesmc {v_state}.16b, {v_state}.16b",
                    v_state = inout(vreg) state => out,
                    v_rkey = in(vreg) rkey,
                    options(nostack, pure, nomem, preserves_flags),
                );
            } else {
                asm!(
                    "eor {v_state}.16b, {v_state}.16b, {v_rkey}.16b",
                    "rev64 {v_tmp}.16b, {v_state}.16b",
                    "ext {v_state}.16b, {v_state}.16b, {v_state}.16b, #8",
                    "eor {v_state}.16b, {v_state}.16b, {v_tmp}.16b",
                    v_state = inout(vreg) state => out,
                    v_rkey = in(vreg) rkey,
                    v_tmp = out(vreg) _,
                    options(nostack, pure, nomem, preserves_flags),
                );
            }

            // Unpack lanes back into standard scalar return values safely
            let res = core::mem::transmute::<uint64x2_t, [u64; 2]>(out);
            return (res[0], res[1]);
        }
    }

    #[cfg(all(target_arch = "riscv64", target_feature = "zkn"))]
    {
        // RISC-V Zkn scalar AES: one full round (SubBytes + ShiftRows +
        // MixColumns) via `aes64esm`, plus explicit AddRoundKey XOR.
        //
        // The 128-bit state is {lhs (low 64 bits), rhs (high 64 bits)}.
        // `aes64esm rd, rs1, rs2` computes the low (or high) half of
        // SubBytes(ShiftRows({rs2, rs1})) + MixColumns, depending on
        // which half rs1/rs2 map to. Swapping the argument order produces
        // the complementary half — this is the standard two-instruction
        // RV64 AES-round pattern from the RISC-V Scalar Crypto spec §3.4.
        const RK_LO: u64 = 0xcbf2_9ce4_8422_2325; // same FNV-derived key as other paths
        const RK_HI: u64 = 0x1000_0000_01b3_0000;
        let lo: u64;
        let hi: u64;
        unsafe {
            use core::arch::asm;
            asm!(
                // Low half: aes64esm(rd, low, high)
                "aes64esm {lo}, {lhs}, {rhs}",
                // High half: aes64esm(rd, high, low) — args swapped per spec
                "aes64esm {hi}, {rhs}, {lhs}",
                // AddRoundKey
                "xor {lo}, {lo}, {rk_lo}",
                "xor {hi}, {hi}, {rk_hi}",
                lhs   = in(reg) lhs,
                rhs   = in(reg) rhs,
                rk_lo = in(reg) RK_LO,
                rk_hi = in(reg) RK_HI,
                lo    = out(reg) lo,
                hi    = out(reg) hi,
                options(nostack, pure, nomem),
            );
        }
        return (lo, hi);
    }

    // Scalar fallback (non-crypto AArch64, riscv64 without Zkn, and other
    // targets): a classic multiplicative xor-mix. Constants declared at the
    // function top so the items-after-statements lint is satisfied even when
    // all hardware paths are excluded.
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
