//! C-compatible types for the FFI surface.
//!
//! Exposes flat status codes and struct definitions compatible with cbindgen
//! for robust cross-language integration.

/// Return status codes for C-API function invocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum RssnStatus {
    /// Operation completed successfully.
    Success = 0,
    /// A null pointer was passed for a mandatory parameter.
    NullPointer = 1,
    /// Failed to parse expression.
    ParseError = 2,
    /// Failed to JIT compile the target expression.
    CompilationError = 3,
    /// A panic occurred during execution.
    Panic = 4,
}
