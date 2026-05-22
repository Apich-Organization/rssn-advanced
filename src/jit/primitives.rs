//! Built-in JIT primitives for core algebraic operations.
//!
//! Provides the primary algebraic logic for core addition, subtraction,
//! multiplication, and division nodes including coefficients merges and zero guards.

/// Simplifies addition/subtraction.
///
/// Folds exact additive identities only. Using `f64::EPSILON` here would
/// silently discard small-but-nonzero values (e.g. `1e-20`), corrupting
/// symbolic precision. We only eliminate *exact* zeros.
#[must_use]
pub fn simplify_add(lhs: f64, rhs: f64) -> Option<f64> {
    if lhs == 0.0 {
        Some(rhs)
    } else if rhs == 0.0 {
        Some(lhs)
    } else {
        Some(lhs + rhs)
    }
}

/// Simplifies multiplication.
///
/// Folds exact multiplicative identities (0.0, 1.0) only. Fuzzy matching
/// with `f64::EPSILON` would silently annihilate tiny-but-nonzero values.
#[must_use]
pub fn simplify_mul(lhs: f64, rhs: f64) -> Option<f64> {
    if lhs == 0.0 || rhs == 0.0 {
        Some(0.0)
    } else if lhs == 1.0 {
        Some(rhs)
    } else if rhs == 1.0 {
        Some(lhs)
    } else {
        Some(lhs * rhs)
    }
}

/// Simplifies division.
///
/// Ensures mandatory division-by-zero checks. Only exact zero triggers the
/// error; fuzzy `EPSILON` checks would incorrectly reject near-zero
/// denominators that are mathematically valid.
///
/// # Errors
/// Returns `Err` if a division by zero is detected.
pub fn simplify_div(lhs: f64, rhs: f64) -> Result<f64, String> {
    if rhs == 0.0 {
        Err("Division by zero in JIT primitive".to_owned())
    } else {
        Ok(lhs / rhs)
    }
}
