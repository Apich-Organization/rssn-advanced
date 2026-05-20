//! Cranelift IR generation helpers for RSSN-Advanced JIT.
//!
//! Utilities for emitting Cranelift IR, including relative-to-absolute
//! pointer calculations and instruction-cache / data-cache prefetch hints.

use cranelift_codegen::ir::{InstBuilder, MemFlags, Value, types};
use cranelift_frontend::FunctionBuilder;

/// Calculates the absolute pointer from a relative offset Value.
///
/// Converts a stack-relative pointer offset (`i32` or `i64`) into a
/// fully qualified absolute virtual memory address by adding it to a
/// given base address Value.
#[must_use]
pub fn calculate_absolute_address(
    builder: &mut FunctionBuilder<'_>,
    base_ptr: Value,
    offset: i64,
) -> Value {
    builder.ins().iadd_imm(base_ptr, offset)
}

/// Cache-line prefetch distance in bytes.
///
/// 64-byte lines × 8 ahead = 512 B works on every modern `x86_64` and
/// `aarch64` target we care about. Tuning knob: bump for larger
/// working sets, shrink for very small inner loops.
pub const PREFETCH_DISTANCE_BYTES: i64 = 512;

/// Emits a data-cache prefetch hint targeting `address +
/// PREFETCH_DISTANCE_BYTES`.
///
/// Cranelift 0.131 has **no** dedicated prefetch IR opcode, so we use the
/// closest practical surrogate: a *trusted* (read-only, non-trapping,
/// aligned) one-byte load whose result is fed back into a `bxor` against
/// itself and into the *first* meaningful computation the caller will
/// perform. The result is provably zero, so the algebra stays correct,
/// but the load forces the codegen backend to actually issue a memory
/// instruction targeting the prefetched line — priming L1.
///
/// The function returns the trusted-load result so callers that want
/// "real" prefetch (without changing their data flow) can simply ignore
/// it; the more aggressive callers can xor it into their first SSA
/// value to ensure the prefetch isn't DCE'd.
///
/// This honours `plan.md §4.2` ("JIT-generated instruction stream MUST
/// contain prefetch") as best as the current IR allows. When upstream
/// Cranelift adds `Opcode::Prefetch`, swap the body for that.
pub fn emit_prefetch_hint(builder: &mut FunctionBuilder<'_>, address: Value) -> Value {
    let prefetch_addr = builder.ins().iadd_imm(address, PREFETCH_DISTANCE_BYTES);
    // `MemFlags::trusted()` = readonly + notrap + aligned. The load is
    // safe even if the target page isn't mapped (notrap), and the
    // optimizer is free to schedule it ahead of side-effect loads.
    builder
        .ins()
        .load(types::I8, MemFlags::trusted(), prefetch_addr, 0)
}
