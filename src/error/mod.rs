//! Cold-path error infrastructure.
//!
//! Provides the `rssn_error!` macro that generates, for every variant of an
//! error enum, a `cold_*` constructor marked `#[cold]` `#[inline(never)]`
//! `#[track_caller]`. The hot path branches into these constructors only on
//! genuine failure, keeping the success path's I-cache footprint minimal and
//! letting the branch predictor assume success.
//!
//! Each module-level error type below is the canonical failure surface for one
//! subsystem. They are intentionally narrow — variants describe *what* went
//! wrong, not *where* (the `#[track_caller]` attribute on the cold constructor
//! captures the call site).

use core::fmt;

// =========================================================================
// The rssn_error! macro
// =========================================================================

/// Declares an error enum together with one `#[cold]` constructor per variant.
///
/// For each variant `Foo`, the macro emits a public free function
/// `cold_<enum>_foo(...) -> Result<T, Enum>` that returns
/// `Err(Enum::Foo { .. })`. Callers in hot code use:
///
/// ```ignore
/// return cold_dag_error_arena_full();
/// ```
///
/// The function is `#[cold] #[inline(never)] #[track_caller]`, ensuring the
/// compiler lays it out off the hot path and that panic-like diagnostics
/// preserve caller location.
#[doc(hidden)]
#[macro_export]
macro_rules! rssn_error {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident $( { $( $(#[$field_meta:meta])* $field:ident : $ftype:ty ),* $(,)? } )? $( ( $( $(#[$tuple_meta:meta])* $tname:ident : $ttype:ty ),* $(,)? ) )?
            ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        $vis enum $name {
            $(
                $(#[$variant_meta])*
                $variant $( { $( $(#[$field_meta])* $field : $ftype ),* } )? $( ( $( $(#[$tuple_meta])* $ttype ),* ) )?,
            )*
        }

        pastey::paste! {
            $(
                $(#[$variant_meta])*
                #[doc(hidden)]
                #[cold]
                #[track_caller]
                #[inline(never)]
                pub const fn [<cold_ $name:snake _ $variant:snake>]<T>(
                    $($($field : $ftype),*)?
                    $($($tname : $ttype),*)?
                ) -> core::result::Result<T, $name> {
                    core::result::Result::Err($name::$variant $( { $($field),* } )? $( ( $( $tname ),* ) )?)
                }
            )*
        }
    };
}

// =========================================================================
// Module-level error enums
// =========================================================================

rssn_error! {
    /// AST / projection-layer failures.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum AstError {
        /// The AST projection ran out of inline buffer space.
        ProjectionOverflow,
        /// A relative pointer overflowed its underlying integer width.
        RelPtrOverflow,
        /// Conversion between DAG and AST encountered an unknown symbol kind.
        UnknownSymbolKind,
        /// A cycle was detected while converting; the input is not a DAG.
        CycleDetected,
    }
}

rssn_error! {
    /// DAG arena / dedup / interning failures.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum DagError {
        /// The arena hit its maximum addressable capacity (`u32::MAX - 1`).
        ArenaFull,
        /// A `DagNodeId` referred to a slot outside the current arena.
        DanglingId,
        /// Attempted to dereference the null sentinel.
        NullSentinel,
        /// A node's child count exceeded the configured arity limit.
        ArityOverflow,
    }
}

rssn_error! {
    /// JIT compilation pipeline failures.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum JitError {
        /// Cranelift rejected the emitted IR (verifier or backend failure).
        VerifierRejected,
        /// A custom function symbol has no registered implementation.
        UnknownFunction,
        /// The IR builder hit its instruction budget.
        BudgetExhausted,
        /// Codegen visited a node whose shape contradicts its `SymbolKind`.
        MalformedNode,
    }
}

rssn_error! {
    /// Parallel evaluation / chunking failures.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum ParallelError {
        /// The expression has no commutative split point.
        NotSplittable,
        /// A worker fiber panicked before reporting a result.
        WorkerLost,
        /// A required variable binding was missing.
        MissingVariable,
    }
}

rssn_error! {
    /// Parser surface failures (token / grammar level).
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum ParserError {
        /// Reached end of input while a production was still pending.
        UnexpectedEof,
        /// Saw a token that no production could accept.
        UnexpectedToken,
        /// Parenthesis depth exceeded `MAX_PAREN_DEPTH`.
        ParenDepthExceeded,
        /// A numeric literal could not be parsed as `f64`.
        InvalidNumber,
    }
}

rssn_error! {
    /// Storage / cache / serialization failures.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum StorageError {
        /// Underlying IO operation failed (open, read, write, mmap).
        Io,
        /// `bincode` decoded a structurally invalid arena header.
        CorruptHeader,
        /// The on-disk version tag did not match the runtime version.
        VersionMismatch,
        /// Mmap requested but unsupported on this platform.
        MmapUnsupported,
    }
}

rssn_error! {
    /// FFI boundary failures.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum FfiError {
        /// A `*const T` or `*mut T` argument was null where required.
        NullArgument,
        /// A C string was not valid UTF-8.
        InvalidUtf8,
        /// The handle's tag/generation does not match a live object.
        StaleHandle,
        /// The fiber runtime is not initialized.
        RuntimeUninitialized,
    }
}

// =========================================================================
// Display impls (kept hand-written; the macro stays narrow on purpose)
// =========================================================================

macro_rules! impl_display {
    ($($t:ty),* $(,)?) => {
        $(
            impl fmt::Display for $t {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    fmt::Debug::fmt(self, f)
                }
            }
        )*
    };
}

impl_display!(
    AstError,
    DagError,
    JitError,
    ParallelError,
    ParserError,
    StorageError,
    FfiError,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_constructor_returns_err_variant() {
        let r: Result<(), DagError> = cold_dag_error_arena_full();
        assert_eq!(r, Err(DagError::ArenaFull));
    }

    #[test]
    fn cold_constructor_for_every_module() {
        let _: Result<u8, AstError> = cold_ast_error_projection_overflow();
        let _: Result<u8, JitError> = cold_jit_error_verifier_rejected();
        let _: Result<u8, ParallelError> = cold_parallel_error_not_splittable();
        let _: Result<u8, ParserError> = cold_parser_error_unexpected_eof();
        let _: Result<u8, StorageError> = cold_storage_error_io();
        let _: Result<u8, FfiError> = cold_ffi_error_null_argument();
    }

    #[test]
    fn display_uses_debug() {
        assert_eq!(format!("{}", DagError::ArenaFull), "ArenaFull");
    }
}
