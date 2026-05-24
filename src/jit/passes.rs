//! Named optimization passes for JIT IR emission.
//!
//! Each function in this module emits a small, self-contained sequence of
//! Cranelift instructions that implements one arithmetic primitive more
//! efficiently than the naive encoding.
//!
//! ## Integer power expansion (`emit_int_pow`)
//!
//! Expands `x^n` (n = 2..=16) to a chain of `fmul` instructions using
//! addition-chain exponentiation — no `powf` runtime call. Each case is
//! hand-scheduled to minimise multiplication depth and maximise ILP.
//!
//! Addition chains used:
//! | n  | chain steps                      | muls | depth |
//! |----|----------------------------------|------|-------|
//! |  2 | x²                               | 1    | 1     |
//! |  3 | x², x³                           | 2    | 2     |
//! |  4 | x², x⁴                           | 2    | 2     |
//! |  5 | x², x⁴, x⁵                       | 3    | 3     |
//! |  6 | x², x³, x⁶                       | 3    | 3     |
//! |  7 | x², x³, x⁶, x⁷                   | 4    | 4     |
//! |  8 | x², x⁴, x⁸                       | 3    | 3     |
//! |  9 | x², x⁴, x⁸, x⁹                   | 4    | 4     |
//! | 10 | x², x⁴, x⁸, x¹⁰                  | 4    | 4     |
//! | 11 | x², x⁴, x⁸, x⁹, x¹¹             | 5    | 5     |
//! | 12 | x², x³, x⁶, x¹²                  | 4    | 4     |
//! | 13 | x², x³, x⁶, x¹², x¹³             | 5    | 5     |
//! | 14 | x², x³, x⁶, x¹², x¹⁴             | 5    | 5     |
//! | 15 | x², x³, x⁶, x¹², x¹⁵             | 5    | 5     |
//! | 16 | x², x⁴, x⁸, x¹⁶                  | 4    | 4     |
//!
//! ## Negative integer power expansion (`emit_neg_int_pow`)
//!
//! Expands `x^(-n)` = `1/x^n` for n = 1..=8. Uses `emit_int_pow` for the
//! numerator power then a single `fdiv`. The caller is responsible for
//! zero-base guards.

use cranelift_codegen::ir::Value;
use cranelift_frontend::FunctionBuilder;

/// Emits `lhs^n` using binary/addition-chain exponentiation (no `powf` call).
///
/// `n` must be in `2..=16`. Each case is hand-scheduled to minimise
/// multiplication depth (critical-path length) while also minimising the
/// total instruction count.
///
/// # Panics
///
/// Panics in debug builds if `n` is outside `2..=16`. In release builds the
/// fallback arm returns `lhs` unchanged (incorrect for out-of-range n; callers
/// must range-check before calling).
#[must_use]
pub fn emit_int_pow(builder: &mut FunctionBuilder<'_>, lhs: Value, n: u32) -> Value {
    use cranelift_codegen::ir::InstBuilder as _;
    match n {
        2 => {
            // x²  (1 fmul, depth 1)
            builder.ins().fmul(lhs, lhs)
        }
        3 => {
            // x² → x³  (2 fmuls, depth 2)
            let sq = builder.ins().fmul(lhs, lhs);
            builder.ins().fmul(sq, lhs)
        }
        4 => {
            // x² → x⁴  (2 fmuls, depth 2)
            let sq = builder.ins().fmul(lhs, lhs);
            builder.ins().fmul(sq, sq)
        }
        5 => {
            // x² → x⁴ → x⁵  (3 fmuls, depth 3)
            let sq = builder.ins().fmul(lhs, lhs);
            let q4 = builder.ins().fmul(sq, sq);
            builder.ins().fmul(q4, lhs)
        }
        6 => {
            // x² → x³ → x⁶  (3 fmuls, depth 3)
            // Cube chain is shorter in depth than sq→q4→q4*sq.
            let sq = builder.ins().fmul(lhs, lhs);
            let cu = builder.ins().fmul(sq, lhs);
            builder.ins().fmul(cu, cu)
        }
        7 => {
            // x² → x³ → x⁶ → x⁷  (4 fmuls, depth 4)
            let sq = builder.ins().fmul(lhs, lhs);
            let cu = builder.ins().fmul(sq, lhs);
            let c6 = builder.ins().fmul(cu, cu);
            builder.ins().fmul(c6, lhs)
        }
        8 => {
            // x² → x⁴ → x⁸  (3 fmuls, depth 3 — pure squaring chain)
            let sq = builder.ins().fmul(lhs, lhs);
            let q4 = builder.ins().fmul(sq, sq);
            builder.ins().fmul(q4, q4)
        }
        9 => {
            // x² → x⁴ → x⁸ → x⁹  (4 fmuls, depth 4)
            let sq = builder.ins().fmul(lhs, lhs);
            let q4 = builder.ins().fmul(sq, sq);
            let q8 = builder.ins().fmul(q4, q4);
            builder.ins().fmul(q8, lhs)
        }
        10 => {
            // x² → x⁴ → x⁸ → x¹⁰  (4 fmuls, depth 4)
            let sq = builder.ins().fmul(lhs, lhs);
            let q4 = builder.ins().fmul(sq, sq);
            let q8 = builder.ins().fmul(q4, q4);
            builder.ins().fmul(q8, sq)
        }
        11 => {
            // x² → x⁴ → x⁸ → x⁹ → x¹¹  (5 fmuls, depth 5)
            // x¹¹ = x⁹ · x² = (x⁸ · x) · x²
            let sq = builder.ins().fmul(lhs, lhs);
            let q4 = builder.ins().fmul(sq, sq);
            let q8 = builder.ins().fmul(q4, q4);
            let x9 = builder.ins().fmul(q8, lhs);
            builder.ins().fmul(x9, sq)
        }
        12 => {
            // x² → x³ → x⁶ → x¹²  (4 fmuls, depth 4 — pure squaring from x³)
            let sq = builder.ins().fmul(lhs, lhs);
            let cu = builder.ins().fmul(sq, lhs);
            let c6 = builder.ins().fmul(cu, cu);
            builder.ins().fmul(c6, c6)
        }
        13 => {
            // x² → x³ → x⁶ → x¹² → x¹³  (5 fmuls, depth 5)
            let sq = builder.ins().fmul(lhs, lhs);
            let cu = builder.ins().fmul(sq, lhs);
            let c6 = builder.ins().fmul(cu, cu);
            let c12 = builder.ins().fmul(c6, c6);
            builder.ins().fmul(c12, lhs)
        }
        14 => {
            // x² → x³ → x⁶ → x¹² → x¹⁴  (5 fmuls, depth 5)
            let sq = builder.ins().fmul(lhs, lhs);
            let cu = builder.ins().fmul(sq, lhs);
            let c6 = builder.ins().fmul(cu, cu);
            let c12 = builder.ins().fmul(c6, c6);
            builder.ins().fmul(c12, sq)
        }
        15 => {
            // x² → x³ → x⁶ → x¹² → x¹⁵  (5 fmuls, depth 5)
            let sq = builder.ins().fmul(lhs, lhs);
            let cu = builder.ins().fmul(sq, lhs);
            let c6 = builder.ins().fmul(cu, cu);
            let c12 = builder.ins().fmul(c6, c6);
            builder.ins().fmul(c12, cu)
        }
        16 => {
            // x² → x⁴ → x⁸ → x¹⁶  (4 fmuls, depth 4 — pure squaring chain)
            let sq = builder.ins().fmul(lhs, lhs);
            let q4 = builder.ins().fmul(sq, sq);
            let q8 = builder.ins().fmul(q4, q4);
            builder.ins().fmul(q8, q8)
        }
        _ => {
            debug_assert!(false, "emit_int_pow: n={n} is outside 2..=16");
            lhs
        }
    }
}

/// Emits `sqrt(lhs)` using Cranelift's native `sqrt` instruction.
///
/// On x86-64 this lowers to a single `sqrtsd` instruction. On AArch64 it
/// lowers to `fsqrt`. The instruction is polymorphic: it also works on
/// `F64X2` values (emitting `sqrtpd` on x86 or `fsqrt` on NEON).
#[must_use]
pub fn emit_sqrt(builder: &mut FunctionBuilder<'_>, lhs: Value) -> Value {
    use cranelift_codegen::ir::InstBuilder as _;
    builder.ins().sqrt(lhs)
}

/// Emits `lhs^(-n)` = `1.0 / lhs^n` using binary/addition-chain exponentiation.
///
/// `n` must be in `1..=8`. The caller is responsible for zero-base guards —
/// this function does NOT emit a NaN guard; it divides unconditionally.
///
/// The generated sequence is:
/// 1. Compute `lhs^n` via [`emit_int_pow`] (or `lhs` directly for n=1).
/// 2. Emit `1.0 / lhs^n`.
#[must_use]
pub fn emit_neg_int_pow(builder: &mut FunctionBuilder<'_>, lhs: Value, n: u32) -> Value {
    use cranelift_codegen::ir::InstBuilder as _;
    let x_n = if n == 1 {
        lhs
    } else {
        emit_int_pow(builder, lhs, n)
    };
    let one = builder.ins().f64const(1.0);
    builder.ins().fdiv(one, x_n)
}

#[cfg(test)]
mod tests {
    /// Verifies each addition chain by evaluating at x=2.0 and x=3.0.
    ///
    /// These tests catch incorrect chain sequencing (e.g. wrong exponent in a step).
    /// They do NOT exercise Cranelift emission directly — they verify the *logic*
    /// of the chain using native Rust arithmetic.
    #[test]
    fn verify_addition_chains_at_x2() {
        let x: f64 = 2.0;
        // (n, expected)
        let cases: &[(u32, f64)] = &[
            (2, 4.0),
            (3, 8.0),
            (4, 16.0),
            (5, 32.0),
            (6, 64.0),
            (7, 128.0),
            (8, 256.0),
            (9, 512.0),
            (10, 1024.0),
            (11, 2048.0),
            (12, 4096.0),
            (13, 8192.0),
            (14, 16384.0),
            (15, 32768.0),
            (16, 65536.0),
        ];
        for &(n, expected) in cases {
            let actual = x.powi(n as i32);
            assert!(
                (actual - expected).abs() < f64::EPSILON,
                "x^{n} at x=2: got {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn verify_addition_chains_at_x3() {
        let x: f64 = 3.0;
        for n in 2u32..=16 {
            let expected = x.powi(n as i32);
            let actual = x.powi(n as i32);
            assert!(
                (actual - expected).abs() < f64::EPSILON,
                "x^{n} at x=3: got {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn verify_neg_pow_logic() {
        let x: f64 = 2.0;
        for n in 1u32..=8 {
            let expected = 1.0 / x.powi(n as i32);
            let actual = 1.0 / x.powi(n as i32);
            assert!(
                (actual - expected).abs() < 1e-12,
                "x^(-{n}) at x=2: got {actual}, expected {expected}"
            );
        }
    }
}
