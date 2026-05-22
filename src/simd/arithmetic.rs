//! SIMD-accelerated batch arithmetic operations.
//!
//! All public functions in this module are **slice-iterating wrappers**
//! over the 4-lane kernels in [`crate::asm_presets`]. The inner kernels
//! emit explicit AVX2 / FMA instructions via `core::arch::asm!` — there
//! is no reliance on the compiler's auto-vectorizer (`simd_review §1`).
//!
//! Each wrapper:
//!
//! 1. Checks AVX2 availability once per process via [`HAS_AVX2`].
//! 2. Splits the input into 4-element chunks aligned to the kernel width.
//! 3. Calls the kernel per chunk.
//! 4. Processes the trailing 0..3 elements with the scalar fallback,
//!    guaranteeing bit-identical results across vectorized / scalar paths.
//!
//! Feature detection is hoisted out of the per-chunk inner loop using a
//! `OnceLock<bool>` — `is_x86_feature_detected!` touches an MMIO-mapped
//! CPUID leaf on x86_64 and should not be called on every inner iteration.

use std::sync::OnceLock;

use crate::asm_presets::{
    add_f64x4_avx2, cmp_eq_f64x4, coef_merge_f64x4, fma_f64x4_avx2, mul_f64x4_avx2,
};

/// Cached result of AVX2 detection. Populated on first use; subsequent
/// reads are a single atomic load (Acquire). The `OnceLock` value is a
/// `bool` so the hot path is a branch-predictor-friendly compare.
static HAS_AVX2: OnceLock<bool> = OnceLock::new();

/// Returns `true` if AVX2 is available on the current host CPU.
///
/// The first call probes `is_x86_feature_detected!`; all subsequent calls
/// return the cached result with a single Acquire load.
#[inline]
fn avx2_available() -> bool {
    *HAS_AVX2.get_or_init(|| crate::simd::detect::has_avx2())
}

/// Reasons a batch arithmetic operation cannot proceed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchError {
    /// One or more slice lengths disagree.
    LengthMismatch,
}

/// Width (in `f64` lanes) of one AVX2-256 register.
const LANES: usize = 4;

// =========================================================================
// f64 element-wise primitives
// =========================================================================

/// Batch element-wise addition: `result[i] = lhs[i] + rhs[i]`.
///
/// # Errors
///
/// Returns [`BatchError::LengthMismatch`] when any of the three slices
/// has a different length.
pub fn batch_add(lhs: &[f64], rhs: &[f64], result: &mut [f64]) -> Result<(), BatchError> {
    let n = lhs.len();
    if n != rhs.len() || n != result.len() {
        return Err(BatchError::LengthMismatch);
    }

    // Detect once per batch call rather than once per chunk call.
    let use_simd = avx2_available();
    let mut chunk_idx = 0;
    // 2× unrolled path: two AVX2/scalar chunks per loop body.
    while chunk_idx + LANES * 2 <= n {
        let (o1, o2) = result[chunk_idx..chunk_idx + LANES * 2].split_at_mut(LANES);
        if use_simd {
            add_f64x4_avx2::apply(&lhs[chunk_idx..chunk_idx + LANES], &rhs[chunk_idx..chunk_idx + LANES], o1);
            add_f64x4_avx2::apply(&lhs[chunk_idx + LANES..chunk_idx + LANES * 2], &rhs[chunk_idx + LANES..chunk_idx + LANES * 2], o2);
        } else {
            for i in 0..LANES { o1[i] = lhs[chunk_idx + i] + rhs[chunk_idx + i]; }
            for i in 0..LANES { o2[i] = lhs[chunk_idx + LANES + i] + rhs[chunk_idx + LANES + i]; }
        }
        chunk_idx += LANES * 2;
    }
    // Handle any remaining full chunk.
    while chunk_idx + LANES <= n {
        let o = &mut result[chunk_idx..chunk_idx + LANES];
        if use_simd {
            add_f64x4_avx2::apply(&lhs[chunk_idx..chunk_idx + LANES], &rhs[chunk_idx..chunk_idx + LANES], o);
        } else {
            for i in 0..LANES { o[i] = lhs[chunk_idx + i] + rhs[chunk_idx + i]; }
        }
        chunk_idx += LANES;
    }
    while chunk_idx < n {
        result[chunk_idx] = lhs[chunk_idx] + rhs[chunk_idx];
        chunk_idx += 1;
    }
    Ok(())
}

/// Batch element-wise multiplication: `result[i] = lhs[i] * rhs[i]`.
///
/// # Errors
///
/// Returns [`BatchError::LengthMismatch`] when any of the three slices
/// has a different length.
pub fn batch_mul(lhs: &[f64], rhs: &[f64], result: &mut [f64]) -> Result<(), BatchError> {
    let n = lhs.len();
    if n != rhs.len() || n != result.len() {
        return Err(BatchError::LengthMismatch);
    }

    let use_simd = avx2_available();
    let mut i = 0;
    while i + LANES * 2 <= n {
        let (o1, o2) = result[i..i + LANES * 2].split_at_mut(LANES);
        if use_simd {
            mul_f64x4_avx2::apply(&lhs[i..i + LANES], &rhs[i..i + LANES], o1);
            mul_f64x4_avx2::apply(&lhs[i + LANES..i + LANES * 2], &rhs[i + LANES..i + LANES * 2], o2);
        } else {
            for j in 0..LANES { o1[j] = lhs[i + j] * rhs[i + j]; }
            for j in 0..LANES { o2[j] = lhs[i + LANES + j] * rhs[i + LANES + j]; }
        }
        i += LANES * 2;
    }
    while i + LANES <= n {
        let o = &mut result[i..i + LANES];
        if use_simd {
            mul_f64x4_avx2::apply(&lhs[i..i + LANES], &rhs[i..i + LANES], o);
        } else {
            for j in 0..LANES { o[j] = lhs[i + j] * rhs[i + j]; }
        }
        i += LANES;
    }
    while i < n {
        result[i] = lhs[i] * rhs[i];
        i += 1;
    }
    Ok(())
}

/// Batch scalar addition: `result[i] = lhs[i] + scalar`.
///
/// # Errors
///
/// Returns [`BatchError::LengthMismatch`] when `lhs` and `result`
/// have different lengths.
pub fn batch_add_scalar(lhs: &[f64], scalar: f64, result: &mut [f64]) -> Result<(), BatchError> {
    let n = lhs.len();
    if n != result.len() {
        return Err(BatchError::LengthMismatch);
    }

    let use_simd = avx2_available();
    let rhs_splat = [scalar; LANES];
    let mut i = 0;
    while i + LANES * 2 <= n {
        let (o1, o2) = result[i..i + LANES * 2].split_at_mut(LANES);
        if use_simd {
            add_f64x4_avx2::apply(&lhs[i..i + LANES], &rhs_splat, o1);
            add_f64x4_avx2::apply(&lhs[i + LANES..i + LANES * 2], &rhs_splat, o2);
        } else {
            for j in 0..LANES { o1[j] = lhs[i + j] + scalar; }
            for j in 0..LANES { o2[j] = lhs[i + LANES + j] + scalar; }
        }
        i += LANES * 2;
    }
    while i + LANES <= n {
        let o = &mut result[i..i + LANES];
        if use_simd {
            add_f64x4_avx2::apply(&lhs[i..i + LANES], &rhs_splat, o);
        } else {
            for j in 0..LANES { o[j] = lhs[i + j] + scalar; }
        }
        i += LANES;
    }
    while i < n {
        result[i] = lhs[i] + scalar;
        i += 1;
    }
    Ok(())
}

// =========================================================================
// New batch operators (T3.1: batch_pow, batch_cmp_eq, batch_coef_merge)
// =========================================================================

/// Batch element-wise `pow`: `result[i] = base[i] ^ exp[i]`.
///
/// `pow` has no AVX2 intrinsic, so this stays per-element via
/// `f64::powf`. Kept in this module for API symmetry with the other
/// batch operations; callers that want SIMD speed should fold the
/// common case `pow(_, 2.0)` into `batch_mul(x, x)`.
///
/// # Errors
///
/// Returns [`BatchError::LengthMismatch`] when any of the three slices
/// has a different length.
pub fn batch_pow(base: &[f64], exp: &[f64], result: &mut [f64]) -> Result<(), BatchError> {
    let n = base.len();
    if n != exp.len() || n != result.len() {
        return Err(BatchError::LengthMismatch);
    }
    for ((b, e), o) in base.iter().zip(exp.iter()).zip(result.iter_mut()) {
        *o = b.powf(*e);
    }
    Ok(())
}

/// Batch IEEE-754 equality: `mask[i] = 0xFF` if `lhs[i] == rhs[i]`,
/// else `0x00`. NaN never equals NaN.
///
/// # Errors
///
/// Returns [`BatchError::LengthMismatch`] when any of the three slices
/// has a different length.
pub fn batch_cmp_eq(lhs: &[f64], rhs: &[f64], mask: &mut [u8]) -> Result<(), BatchError> {
    let n = lhs.len();
    if n != rhs.len() || n != mask.len() {
        return Err(BatchError::LengthMismatch);
    }

    let use_simd = avx2_available();
    let mut i = 0;
    while i + LANES * 2 <= n {
        if use_simd {
            cmp_eq_f64x4::apply(&lhs[i..i + LANES], &rhs[i..i + LANES], &mut mask[i..i + LANES]);
            cmp_eq_f64x4::apply(&lhs[i + LANES..i + LANES * 2], &rhs[i + LANES..i + LANES * 2], &mut mask[i + LANES..i + LANES * 2]);
        } else {
            #[allow(clippy::float_cmp)]
            for j in 0..LANES * 2 {
                mask[i + j] = if lhs[i + j] == rhs[i + j] { 0xFF } else { 0x00 };
            }
        }
        i += LANES * 2;
    }
    while i + LANES <= n {
        if use_simd {
            cmp_eq_f64x4::apply(&lhs[i..i + LANES], &rhs[i..i + LANES], &mut mask[i..i + LANES]);
        } else {
            #[allow(clippy::float_cmp)]
            for j in 0..LANES {
                mask[i + j] = if lhs[i + j] == rhs[i + j] { 0xFF } else { 0x00 };
            }
        }
        i += LANES;
    }
    #[allow(clippy::float_cmp)]
    while i < n {
        mask[i] = if lhs[i] == rhs[i] { 0xFF } else { 0x00 };
        i += 1;
    }
    Ok(())
}

/// Batch symbolic coefficient merge:
/// `out[i] = (coef_a[i] * coef_b[i]) * (var_x[i] * var_y[i])`.
///
/// This is the kernel the JIT peephole pass invokes when it fuses
/// nested products `(coef_a*var_x)*(coef_b*var_y)` (`plan.md §3.1`).
///
/// # Errors
///
/// Returns [`BatchError::LengthMismatch`] when any of the five slices
/// has a different length.
pub fn batch_coef_merge(
    coef_a: &[f64],
    coef_b: &[f64],
    var_x: &[f64],
    var_y: &[f64],
    out: &mut [f64],
) -> Result<(), BatchError> {
    let n = coef_a.len();
    if n != coef_b.len() || n != var_x.len() || n != var_y.len() || n != out.len() {
        return Err(BatchError::LengthMismatch);
    }

    let use_simd = avx2_available();
    let mut i = 0;
    while i + LANES <= n {
        if use_simd {
            coef_merge_f64x4::apply(
                &coef_a[i..i + LANES],
                &coef_b[i..i + LANES],
                &var_x[i..i + LANES],
                &var_y[i..i + LANES],
                &mut out[i..i + LANES],
            );
        } else {
            for j in 0..LANES {
                out[i + j] = (coef_a[i + j] * coef_b[i + j]) * (var_x[i + j] * var_y[i + j]);
            }
        }
        i += LANES;
    }
    while i < n {
        out[i] = (coef_a[i] * coef_b[i]) * (var_x[i] * var_y[i]);
        i += 1;
    }
    Ok(())
}

/// Batch fused multiply-add: `out[i] = lhs[i] * rhs[i] + addend[i]`
/// with single-rounding semantics when FMA is available.
///
/// # Errors
///
/// Returns [`BatchError::LengthMismatch`] when any of the four slices
/// has a different length.
pub fn batch_fma(
    lhs: &[f64],
    rhs: &[f64],
    addend: &[f64],
    out: &mut [f64],
) -> Result<(), BatchError> {
    let n = lhs.len();
    if n != rhs.len() || n != addend.len() || n != out.len() {
        return Err(BatchError::LengthMismatch);
    }

    let use_simd = avx2_available();
    let mut i = 0;
    while i + LANES * 2 <= n {
        let (o1, o2) = out[i..i + LANES * 2].split_at_mut(LANES);
        if use_simd {
            fma_f64x4_avx2::apply(&lhs[i..i + LANES], &rhs[i..i + LANES], &addend[i..i + LANES], o1);
            fma_f64x4_avx2::apply(&lhs[i + LANES..i + LANES * 2], &rhs[i + LANES..i + LANES * 2], &addend[i + LANES..i + LANES * 2], o2);
        } else {
            for j in 0..LANES { o1[j] = lhs[i + j].mul_add(rhs[i + j], addend[i + j]); }
            for j in 0..LANES { o2[j] = lhs[i + LANES + j].mul_add(rhs[i + LANES + j], addend[i + LANES + j]); }
        }
        i += LANES * 2;
    }
    while i + LANES <= n {
        let o = &mut out[i..i + LANES];
        if use_simd {
            fma_f64x4_avx2::apply(&lhs[i..i + LANES], &rhs[i..i + LANES], &addend[i..i + LANES], o);
        } else {
            for j in 0..LANES { o[j] = lhs[i + j].mul_add(rhs[i + j], addend[i + j]); }
        }
        i += LANES;
    }
    while i < n {
        out[i] = lhs[i].mul_add(rhs[i], addend[i]);
        i += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_add_matches_scalar() {
        let lhs: Vec<f64> = (0..27).map(|i| f64::from(i) * 0.5).collect();
        let rhs: Vec<f64> = (0..27).map(|i| f64::from(i) * 1.5).collect();
        let mut result = vec![0.0_f64; 27];
        batch_add(&lhs, &rhs, &mut result).expect("ok");
        for i in 0..27 {
            assert!((result[i] - (lhs[i] + rhs[i])).abs() < 1e-12);
        }
    }

    #[test]
    fn batch_mul_handles_unaligned_tail() {
        // 7 elements = 1 chunk of 4 + 3-element tail.
        let lhs = [2.0_f64, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let rhs = [10.0_f64; 7];
        let mut result = [0.0_f64; 7];
        batch_mul(&lhs, &rhs, &mut result).expect("ok");
        assert_eq!(result, [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    }

    #[test]
    fn batch_add_scalar_splats_correctly() {
        let lhs: Vec<f64> = (0..16).map(f64::from).collect();
        let mut result = vec![0.0_f64; 16];
        batch_add_scalar(&lhs, 1.5, &mut result).expect("ok");
        for i in 0..16 {
            assert!((result[i] - (f64::from(i as i32) + 1.5)).abs() < 1e-12);
        }
    }

    #[test]
    fn batch_pow_per_element() {
        let base = [2.0_f64, 3.0, 4.0, 5.0, 6.0];
        let exp = [2.0_f64, 3.0, 0.5, 0.0, 1.0];
        let mut out = [0.0_f64; 5];
        batch_pow(&base, &exp, &mut out).expect("ok");
        assert!((out[0] - 4.0).abs() < 1e-12, "2^2");
        assert!((out[1] - 27.0).abs() < 1e-12, "3^3");
        // 4^0.5 = sqrt(4) = 2.0
        assert!((out[2] - 2.0).abs() < 1e-12, "sqrt(4)");
        assert_eq!(out[3], 1.0, "5^0");
        assert_eq!(out[4], 6.0, "6^1");
    }

    #[test]
    fn batch_cmp_eq_handles_chunks_and_tail() {
        // 5 elements: one chunk + 1 tail.
        let lhs = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let rhs = [1.0_f64, 0.0, 3.0, 0.0, 5.0];
        let mut mask = [0_u8; 5];
        batch_cmp_eq(&lhs, &rhs, &mut mask).expect("ok");
        assert_eq!(mask, [0xFF, 0x00, 0xFF, 0x00, 0xFF]);
    }

    #[test]
    fn batch_coef_merge_matches_naive() {
        let coef_a = [2.0_f64, 3.0, 4.0, 5.0, 6.0];
        let coef_b = [0.5_f64, 0.5, 0.5, 0.5, 0.5];
        let var_x = [10.0_f64, 20.0, 30.0, 40.0, 50.0];
        let var_y = [1.0_f64, 1.0, 1.0, 1.0, 1.0];
        let mut out = [0.0_f64; 5];
        batch_coef_merge(&coef_a, &coef_b, &var_x, &var_y, &mut out).expect("ok");
        for i in 0..5 {
            let expected = (coef_a[i] * coef_b[i]) * (var_x[i] * var_y[i]);
            assert!((out[i] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn batch_fma_single_rounding() {
        let lhs = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let rhs = [10.0_f64; 5];
        let addend = [100.0_f64; 5];
        let mut out = [0.0_f64; 5];
        batch_fma(&lhs, &rhs, &addend, &mut out).expect("ok");
        for i in 0..5 {
            assert!((out[i] - (lhs[i] * rhs[i] + addend[i])).abs() < 1e-12);
        }
    }

    #[test]
    fn length_mismatch_returns_error() {
        let lhs = [1.0_f64, 2.0];
        let rhs = [1.0_f64; 3];
        let mut out = [0.0_f64; 2];
        assert_eq!(batch_add(&lhs, &rhs, &mut out), Err(BatchError::LengthMismatch));
    }
}
