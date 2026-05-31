//! C-compatible types for the FFI surface.
//!
//! Exposes flat status codes and struct definitions compatible with cbindgen
//! for robust cross-language integration.

/// Return status codes for C-API function invocations.
///
/// Values 0–6 are the original surface; 7–13 are new variants added to give
/// C consumers richer diagnostics without needing to inspect Rust error types.
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
    /// A C string argument was not valid UTF-8.
    InvalidUtf8 = 5,
    /// A `DagNodeId` argument referred to an arena slot that doesn't
    /// exist (or is the null sentinel where one wasn't expected).
    InvalidNodeId = 6,
    /// The DAG arena reached its maximum capacity (`u32::MAX - 1` nodes).
    ArenaFull = 7,
    /// Attempted to register a symbol that already exists in the registry.
    DuplicateSymbol = 8,
    /// An underlying I/O operation failed (file open, read, write, or mmap).
    StorageIo = 9,
    /// The on-disk arena format is corrupt or incompatible.
    StorageFormat = 10,
    /// A rewrite rule conflicts with an existing rule in the registry.
    RuleConflict = 11,
    /// SIMD batch operation received slices of different lengths.
    SimdLengthMismatch = 12,
    /// A node was constructed with the wrong number of children for its operator.
    InvalidArity = 13,
    /// A node descriptor referenced an invalid kind or out-of-range child index.
    InvalidNode = 14,
    /// A caller-provided output buffer was too small to hold the result.
    BufferTooSmall = 15,
    /// An index was out of bounds.
    OutOfBounds = 16,
    /// A broadcast operation failed due to incompatible dimensions.
    BroadcastError = 17,
}
