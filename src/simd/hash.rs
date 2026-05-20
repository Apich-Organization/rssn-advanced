//! Vectorized hashing for batch node deduplication.
//!
//! Uses the AES-NI based 2-lane mix from [`crate::asm_presets::hash_u64x2_aesni`]
//! to compute high-entropy hashes across an arbitrary-length slice. The
//! kernel processes two `u64` lanes per call; the wrapper here folds
//! the results back to a single `u64` per input by xor-mixing the
//! returned `(lo, hi)` pair.
//!
//! When AES-NI isn't available the scalar fallback path in
//! `hash_u64x2_aesni::apply` kicks in transparently — same observable
//! output, slower clock-to-clock.

use crate::asm_presets::hash_u64x2_aesni;
use crate::simd::arithmetic::BatchError;

/// Computes one `u64` hash per input key by folding the AES-NI mix's
/// `(lo, hi)` pair with xor.
///
/// # Errors
///
/// Returns [`BatchError::LengthMismatch`] when `keys.len() != hashes.len()`.
pub fn batch_hash(keys: &[u64], hashes: &mut [u64]) -> Result<(), BatchError> {
    if keys.len() != hashes.len() {
        return Err(BatchError::LengthMismatch);
    }

    // Process two-lane pairs first; the kernel takes one lane per call.
    // We still call it per element (single-element mix is well-defined:
    // the second lane gets a `0` seed). For paired performance a future
    // change can buffer two consecutive keys and use the pair.
    for (k, h) in keys.iter().zip(hashes.iter_mut()) {
        // Mix the key with its bitwise-rotated companion to keep the
        // AES round busy on both halves of the xmm register.
        let companion = k.rotate_left(31);
        let (lo, hi) = hash_u64x2_aesni::apply(*k, companion);
        *h = lo ^ hi;
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
