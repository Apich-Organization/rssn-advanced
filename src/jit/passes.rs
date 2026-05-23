//! Named optimization passes for JIT IR emission.
//!
//! `emit_int_pow` expands `x^n` (n = 2..=8) to a chain of `fmul`
//! instructions using binary exponentiation — no `powf` call.
//! `emit_sqrt` wraps Cranelift's native `sqrt` instruction.

use cranelift_codegen::ir::Value;
use cranelift_frontend::FunctionBuilder;

/// Emits `lhs^n` using binary exponentiation (no `powf` call).
///
/// `n` must be in `2..=8`. Each case is hand-scheduled for minimum
/// multiplication depth and maximum instruction-level parallelism.
///
/// # Panics
/// Panics in debug builds if `n` is outside `2..=8`. In release builds
/// the fallback arm returns `lhs` (safe but incorrect for out-of-range n).
#[must_use]
pub fn emit_int_pow(builder: &mut FunctionBuilder<'_>, lhs: Value, n: u32) -> Value {
    use cranelift_codegen::ir::InstBuilder as _;
    match n {
        2 => {
            // x * x  (1 fmul, depth 1)
            builder.ins().fmul(lhs, lhs)
        }
        3 => {
            // (x * x) * x  (2 fmuls, depth 2)
            let sq = builder.ins().fmul(lhs, lhs);
            builder.ins().fmul(sq, lhs)
        }
        4 => {
            // sq = x * x; sq * sq  (2 fmuls, depth 2)
            let sq = builder.ins().fmul(lhs, lhs);
            builder.ins().fmul(sq, sq)
        }
        5 => {
            // sq = x * x; sq * sq * x  (3 fmuls, depth 3)
            let sq = builder.ins().fmul(lhs, lhs);
            let q = builder.ins().fmul(sq, sq);
            builder.ins().fmul(q, lhs)
        }
        6 => {
            // sq = x * x; cu = sq * x; cu * cu  (3 fmuls, depth 2 after cu)
            let sq = builder.ins().fmul(lhs, lhs);
            let cu = builder.ins().fmul(sq, lhs);
            builder.ins().fmul(cu, cu)
        }
        7 => {
            // sq = x * x; cu = sq * x; cu * cu * x  (4 fmuls)
            let sq = builder.ins().fmul(lhs, lhs);
            let cu = builder.ins().fmul(sq, lhs);
            let c6 = builder.ins().fmul(cu, cu);
            builder.ins().fmul(c6, lhs)
        }
        8 => {
            // sq = x*x; q = sq*sq; q*q  (3 fmuls, depth 3)
            let sq = builder.ins().fmul(lhs, lhs);
            let q = builder.ins().fmul(sq, sq);
            builder.ins().fmul(q, q)
        }
        _ => {
            debug_assert!(false, "emit_int_pow: n={n} is outside 2..=8");
            lhs
        }
    }
}

/// Emits `sqrt(lhs)` using Cranelift's native `sqrt` instruction.
///
/// On x86-64 this lowers to a single `sqrtsd` instruction. On AArch64
/// it lowers to `fsqrt`.
#[must_use]
pub fn emit_sqrt(builder: &mut FunctionBuilder<'_>, lhs: Value) -> Value {
    use cranelift_codegen::ir::InstBuilder as _;
    builder.ins().sqrt(lhs)
}
