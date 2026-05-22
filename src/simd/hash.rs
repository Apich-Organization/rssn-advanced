//! Vectorized hashing for batch node deduplication.
//!
//! Uses the AES-NI based 2-lane mix from [`crate::asm_presets::hash_u64x2_aesni`]
//! to compute high-entropy hashes across an arbitrary-length slice. The
//! kernel processes two `u64` lanes simultaneously; the batch wrapper feeds
//! consecutive key pairs `(keys[i], keys[i+1])` through the kernel together,
//! halving the number of AES-NI invocations for even-length inputs.
//!
//! When AES-NI isn't available the scalar fallback path in
//! `hash_u64x2_aesni::apply` kicks in transparently — same observable
//! throughput improvement from pairing, slower clock-to-clock.

use crate::asm_presets::hash_u64x2_aesni;
use crate::simd::arithmetic::BatchError;

/// Computes one `u64` hash per input key using the AES-NI 2-lane kernel.
///
/// Consecutive keys are processed in pairs: `(keys[i], keys[i+1])` yields
/// `(lo, hi)` where `lo` is used as `hashes[i]` and `hi` as `hashes[i+1]`.
/// This halves the kernel call count compared to processing one key at a
/// time. A trailing odd element is handled with a rotation companion as
/// the second lane.
///
/// # Errors
///
/// Returns [`BatchError::LengthMismatch`] when `keys.len() != hashes.len()`.
pub fn batch_hash(keys: &[u64], hashes: &mut [u64]) -> Result<(), BatchError> {
    if keys.len() != hashes.len() {
        return Err(BatchError::LengthMismatch);
    }

    let n = keys.len();
    let mut i = 0;

    // Process pairs: one AES-NI call yields two independent hashes.
    while i + 2 <= n {
        let (lo, hi) = hash_u64x2_aesni::apply(keys[i], keys[i + 1]);
        hashes[i]     = lo;
        hashes[i + 1] = hi;
        i += 2;
    }

    // Tail: 0 or 1 remaining key — use rotation as second lane seed.
    if i < n {
        let companion = keys[i].rotate_left(31);
        let (lo, hi) = hash_u64x2_aesni::apply(keys[i], companion);
        hashes[i] = lo ^ hi;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_keys_produce_distinct_hashes() {
        let keys: Vec<u64> = (0..16u64).map(|i| 0xdead_0000_0000_0000 | i).collect();
        let mut hashes = vec![0_u64; 16];
        batch_hash(&keys, &mut hashes).expect("ok");
        for i in 0..16 {
            for j in (i + 1)..16 {
                assert_ne!(hashes[i], hashes[j], "collision at {i},{j}");
            }
        }
    }

    #[test]
    fn hash_is_deterministic() {
        let keys = [1_u64, 2, 3, 4, 5];
        let mut h1 = [0_u64; 5];
        let mut h2 = [0_u64; 5];
        batch_hash(&keys, &mut h1).expect("ok");
        batch_hash(&keys, &mut h2).expect("ok");
        assert_eq!(h1, h2);
    }

    #[test]
    fn length_mismatch_errors() {
        let keys = [1_u64, 2];
        let mut hashes = [0_u64; 3];
        assert!(batch_hash(&keys, &mut hashes).is_err());
    }
}
