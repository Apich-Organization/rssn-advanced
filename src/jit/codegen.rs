//! Cranelift IR generation helpers for RSSN-Advanced JIT.
//!
//! Utilities for emitting Cranelift IR, including relative-to-absolute
//! pointer calculations.

use cranelift_codegen::ir::{InstBuilder, Value};
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
