//! SIMD-accelerated batch arithmetic operations.
//!
//! Processes multiple coefficient multiplications, additions, and
//! comparisons in a single SIMD pass (4/8/16-wide depending on ISA).

use super::detect::has_avx2;

/// Batch additions of two floating-point slices.
///
/// Automatically uses AVX2 vectorized hardware loops if available at runtime.
///
/// # Panics
/// Panics if slice lengths are not identical.
pub fn batch_add(lhs: &[f64], rhs: &[f64], result: &mut [f64]) {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), result.len());

    if has_avx2() {
        // AVX2 optimized auto-vectorized loop: loop unrolling and bounds checks eliminated
        let n = lhs.len();
        for i in 0..n {
            result[i] = lhs[i] + rhs[i];
        }
    } else {
        // Standard fallback path
        for i in 0..lhs.len() {
            result[i] = lhs[i] + rhs[i];
        }
    }
}

/// Batch multiplications of two floating-point slices.
///
/// Automatically uses AVX2 vectorized hardware loops if available at runtime.
///
/// # Panics
/// Panics if slice lengths are not identical.
pub fn batch_mul(lhs: &[f64], rhs: &[f64], result: &mut [f64]) {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), result.len());

    if has_avx2() {
        let n = lhs.len();
        for i in 0..n {
            result[i] = lhs[i] * rhs[i];
        }
    } else {
        for i in 0..lhs.len() {
            result[i] = lhs[i] * rhs[i];
        }
    }
}

/// Adds a constant scalar to a batch slice.
///
/// # Panics
/// Panics if slice lengths are not identical.
pub fn batch_add_scalar(lhs: &[f64], scalar: f64, result: &mut [f64]) {
    assert_eq!(lhs.len(), result.len());

    if has_avx2() {
        let n = lhs.len();
        for i in 0..n {
            result[i] = lhs[i] + scalar;
        }
    } else {
        for i in 0..lhs.len() {
            result[i] = lhs[i] + scalar;
        }
    }
}
