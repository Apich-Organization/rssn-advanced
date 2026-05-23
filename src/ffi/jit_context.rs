//! Persistent JIT compilation context exposed over the C ABI.
//!
//! `rssn_dag_compile` in `c_api` constructs a fresh `JitCompiler` on
//! every call — paying Cranelift initialisation cost (ISA detection +
//! JIT module setup) each time. This module exposes an opaque
//! `RssnJitContext` handle that wraps a long-lived `JitCompiler`, letting
//! callers amortise that cost across many compile calls.
//!
//! Typical usage from C:
//! ```c
//! RssnJitContext *ctx = rssn_jit_context_new();
//! void *fn_ptr = NULL;
//! RssnStatus s = rssn_dag_compile_with_ctx(ctx, builder, root_id, &fn_ptr);
//! if (s == RSSN_SUCCESS) { double r = ((double(*)(const double*))fn_ptr)(vars); }
//! rssn_jit_context_free(ctx);
//! ```

#![allow(unsafe_code)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::os::raw::c_void;

use crate::dag::builder::DagBuilder;
use crate::ffi::types::RssnStatus;

#[cfg(feature = "cranelift-jit")]
use std::sync::{Mutex, OnceLock};

// =========================================================================
// JIT-enabled build
// =========================================================================

/// Opaque persistent JIT context. Holds the Cranelift `JitCompiler` so it
/// can be reused across multiple `rssn_dag_compile_with_ctx` calls without
/// re-paying the per-call ISA detection / module initialisation cost.
#[cfg(feature = "cranelift-jit")]
pub struct RssnJitContext {
    compiler: crate::jit::compiler::JitCompiler,
}

/// Process-level shared JIT context.
///
/// `rssn_dag_compile` routes here instead of creating a fresh `JitCompiler`
/// on every call. Cranelift ISA detection and module setup happen exactly
/// once, on first use. `JitCompiler` is `!Sync` so we wrap in `Mutex`.
#[cfg(feature = "cranelift-jit")]
static GLOBAL_JIT_CTX: OnceLock<Mutex<RssnJitContext>> = OnceLock::new();

/// Returns a reference to the process-level `Mutex<RssnJitContext>`,
/// initialising it on first call.
#[cfg(feature = "cranelift-jit")]
pub(crate) fn global_jit_ctx() -> &'static Mutex<RssnJitContext> {
    GLOBAL_JIT_CTX.get_or_init(|| {
        Mutex::new(RssnJitContext {
            compiler: crate::jit::compiler::JitCompiler::new(),
        })
    })
}

/// Implementation block — internal access used by `c_api.rs`.
#[cfg(feature = "cranelift-jit")]
impl RssnJitContext {
    /// Returns a mutable reference to the underlying `JitCompiler`.
    #[inline]
    pub(crate) fn compiler_mut(&mut self) -> &mut crate::jit::compiler::JitCompiler {
        &mut self.compiler
    }
}

/// Creates a new persistent JIT context.
///
/// Returns a raw pointer to the context, or NULL if construction panicked.
/// The caller must free the returned context with [`rssn_jit_context_free`].
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_jit_context_new() -> *mut RssnJitContext {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Box::into_raw(Box::new(RssnJitContext {
            compiler: crate::jit::compiler::JitCompiler::new(),
        }))
    }));
    result.unwrap_or(std::ptr::null_mut())
}

/// Frees a persistent JIT context previously created by
/// [`rssn_jit_context_new`]. Passing NULL is a safe no-op.
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_jit_context_free(ctx: *mut RssnJitContext) {
    if ctx.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = unsafe { Box::from_raw(ctx) };
    }));
}

/// Compiles a DAG expression using a persistent JIT context.
///
/// Reuses the Cranelift module stored in `ctx`, skipping the per-call
/// ISA detection + module setup that `rssn_dag_compile` pays. On
/// `Success`, writes the compiled function pointer to `*out_fn`.
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_compile_with_ctx(
    ctx: *mut RssnJitContext,
    builder: *mut DagBuilder,
    root: u32,
    out_fn: *mut *mut c_void,
) -> RssnStatus {
    if ctx.is_null() || builder.is_null() || out_fn.is_null() {
        return RssnStatus::NullPointer;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let ctx_ref = unsafe { &mut *ctx };
        let builder_ref = unsafe { &mut *builder };
        let root_id = crate::dag::node::DagNodeId::new(root);
        let ast = crate::ast::convert::dag_to_ast(builder_ref.arena(), root_id);
        match ctx_ref.compiler_mut().compile(&ast) {
            Ok(compiled_fn) => {
                unsafe { *out_fn = compiled_fn as *mut c_void };
                RssnStatus::Success
            }
            Err(_) => RssnStatus::CompilationError,
        }
    }));
    result.unwrap_or(RssnStatus::Panic)
}

// =========================================================================
// Non-JIT stubs
// =========================================================================

/// Stub for non-JIT builds: always returns NULL.
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_jit_context_new() -> *mut c_void {
    std::ptr::null_mut()
}

/// Stub for non-JIT builds: no-op.
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_jit_context_free(_ctx: *mut c_void) {}

/// Stub for non-JIT builds: always returns `CompilationError`.
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_compile_with_ctx(
    _ctx: *mut c_void,
    _builder: *mut DagBuilder,
    _root: u32,
    _out_fn: *mut *mut c_void,
) -> RssnStatus {
    RssnStatus::CompilationError
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(all(test, feature = "cranelift-jit"))]
mod tests {
    use super::*;
    use crate::dag::builder::DagBuilder;
    use crate::parser::expr::parse_expression;

    #[test]
    fn jit_context_compile_and_execute() {
        let ctx = rssn_jit_context_new();
        assert!(!ctx.is_null());

        let mut b = DagBuilder::new();
        let _id = parse_expression("x + 1.0", &mut b).expect("parse");
        let builder_ptr: *mut DagBuilder = &mut b;

        // Compile with the persistent context.
        let mut fn_ptr: *mut c_void = std::ptr::null_mut();
        let status = rssn_dag_compile_with_ctx(
            ctx,
            builder_ptr,
            b.arena().len() as u32 - 1,
            &mut fn_ptr,
        );
        assert_eq!(status, RssnStatus::Success);
        assert!(!fn_ptr.is_null());

        rssn_jit_context_free(ctx);
    }

    #[test]
    fn null_context_returns_null_pointer_status() {
        let mut b = DagBuilder::new();
        let _ = b.variable("x");
        let mut fn_ptr: *mut c_void = std::ptr::null_mut();
        let status = rssn_dag_compile_with_ctx(
            std::ptr::null_mut(),
            &mut b,
            0,
            &mut fn_ptr,
        );
        assert_eq!(status, RssnStatus::NullPointer);
    }
}
