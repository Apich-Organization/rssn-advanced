//! Built-in JIT primitives for core algebraic operations.
//!
//! Provides the primary algebraic logic for core addition, subtraction,
//! multiplication, and division nodes including coefficients merges and zero guards.

/// Simplifies addition/subtraction.
///
/// Implements `+0` branch optimization and basic constant folding.
#[must_use]
pub fn simplify_add(lhs: f64, rhs: f64) -> Option<f64> {
    // Branch predicted identity checks
    if lhs.abs() < f64::EPSILON {
        Some(rhs)
    } else if rhs.abs() < f64::EPSILON {
        Some(lhs)
    } else {
        Some(lhs + rhs)
    }
}

/// Simplifies multiplication.
///
/// Implements coefficient merge and identity checks.
#[must_use]
pub fn simplify_mul(lhs: f64, rhs: f64) -> Option<f64> {
    if lhs.abs() < f64::EPSILON || rhs.abs() < f64::EPSILON {
        Some(0.0)
    } else if (lhs - 1.0).abs() < f64::EPSILON {
        Some(rhs)
    } else if (rhs - 1.0).abs() < f64::EPSILON {
        Some(lhs)
    } else {
        Some(lhs * rhs)
    }
}

/// Simplifies division.
///
/// Ensures mandatory division-by-zero checks.
///
/// # Errors
/// Returns `Err` if a division by zero is detected.
pub fn simplify_div(lhs: f64, rhs: f64) -> Result<f64, String> {
    if rhs.abs() < f64::EPSILON {
        Err("Division by zero in JIT primitive".to_owned())
    } else {
        Ok(lhs / rhs)
    }
}
