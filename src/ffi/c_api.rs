//! `extern "C"` entry points for the RSSN-Advanced API.
//!
//! Exposes a flat C API, capturing panics securely at the FFI boundary to
//! avoid undefined behavior (UB).

#![allow(unsafe_code)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::time::Duration;
use std::panic::catch_unwind;
use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use crate::heuristic::{HeuristicEngine, HeuristicConfig, SearchStrategy};
use super::types::RssnStatus;

/// Creates a new `DagBuilder` context.
///
/// Returns a raw pointer to the builder, or NULL if creation failed or panicked.
/// The returned pointer must be freed exactly once via [`rssn_dag_free`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_new() -> *mut DagBuilder {
    let result = catch_unwind(|| {
        Box::into_raw(Box::new(DagBuilder::new()))
    });
    result.unwrap_or(std::ptr::null_mut())
}

/// Releases the memory of a previously allocated `DagBuilder`.
///
/// # Safety
///
/// `builder` must be a pointer previously returned by [`rssn_dag_new`], or NULL.
/// After this call the pointer is dangling and must not be used.
/// Passing a pointer not from `rssn_dag_new`, or freeing twice, is undefined behaviour.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_free(builder: *mut DagBuilder) {
    if builder.is_null() {
        return;
    }
    let _ = catch_unwind(|| {
        let _ = unsafe { Box::from_raw(builder) };
    });
}

/// Allocates a new variable node in the DAG.
///
/// Returns the index of the variable node, or `u32::MAX` if a panic
/// occurred, the builder was null, or `name` was not valid UTF-8.
///
/// Looks up `name` zero-allocation on the hot path:
/// [`CStr::to_bytes`] → `SymbolRegistry::intern_bytes`. Only the
/// first time a given name is interned does an allocation happen
/// (`ffi_review §2`).
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
/// - `name` must be a valid, non-null, null-terminated C string valid for the duration of
///   this call.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_variable(builder: *mut DagBuilder, name: *const c_char) -> u32 {
    if builder.is_null() || name.is_null() {
        return u32::MAX;
    }
    let result = catch_unwind(|| -> u32 {
        let builder_ref = unsafe { &mut *builder };
        let c_str = unsafe { CStr::from_ptr(name) };
        builder_ref
            .variable_bytes(c_str.to_bytes())
            .map_or(u32::MAX, DagNodeId::value)
    });
    result.unwrap_or(u32::MAX)
}

/// Allocates a new constant node in the DAG.
///
/// # Safety
///
/// `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_constant(builder: *mut DagBuilder, val: f64) -> u32 {
    if builder.is_null() {
        return u32::MAX;
    }
    let result = catch_unwind(|| {
        let builder_ref = unsafe { &mut *builder };
        builder_ref.constant(val).value()
    });
    result.unwrap_or(u32::MAX)
}

/// Allocates a new addition node in the DAG: `lhs + rhs`.
///
/// # Safety
///
/// `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_add(builder: *mut DagBuilder, lhs: u32, rhs: u32) -> u32 {
    if builder.is_null() {
        return u32::MAX;
    }
    let result = catch_unwind(|| {
        let builder_ref = unsafe { &mut *builder };
        builder_ref.add(DagNodeId::new(lhs), DagNodeId::new(rhs)).value()
    });
    result.unwrap_or(u32::MAX)
}

/// Simplifies a target expression using the default heuristic engine.
///
/// Returns the new root node index of the simplified expression.
///
/// # Safety
///
/// `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_simplify(builder: *mut DagBuilder, root: u32) -> u32 {
    if builder.is_null() {
        return u32::MAX;
    }
    let result = catch_unwind(|| {
        let builder_ref = unsafe { &mut *builder };
        let root_id = DagNodeId::new(root);

        let config = HeuristicConfig::default();
        let mut engine = HeuristicEngine::new(config, SearchStrategy::Greedy);

        engine.simplify(builder_ref, root_id).value()
    });
    result.unwrap_or(u32::MAX)
}

/// JIT compiles a target expression and writes the native function pointer to `out_fn`.
///
/// `out_fn` can be called via `rssn_dag_execute` or cast directly as `double (*)(const double*)`.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
/// - `out_fn` must be a valid, non-null pointer to a `*mut c_void` that the function will write to.
/// - The compiled function pointer written to `*out_fn` remains valid until the `JITModule`
///   backing this compiler is dropped. Do not call it after that.
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_compile(
    builder: *mut DagBuilder,
    root: u32,
    out_fn: *mut *mut c_void,
) -> RssnStatus {
    if builder.is_null() || out_fn.is_null() {
        return RssnStatus::NullPointer;
    }

    let result = catch_unwind(|| {
        let builder_ref = unsafe { &mut *builder };
        let root_id = DagNodeId::new(root);
        
        // Project root node to AST
        let ast = crate::ast::convert::dag_to_ast(builder_ref.arena(), root_id);

        // JIT compile
        let mut compiler = crate::jit::compiler::JitCompiler::new();
        match compiler.compile(&ast) {
            Ok(compiled_fn) => {
                let ptr = compiled_fn as *mut c_void;
                unsafe { *out_fn = ptr };
                RssnStatus::Success
            }
            Err(_) => RssnStatus::CompilationError,
        }
    });

    result.unwrap_or(RssnStatus::Panic)
}

/// JIT compiles a target expression and writes the native function pointer to `out_fn`.
///
/// `out_fn` can be called via `rssn_dag_execute` or cast directly as `double (*)(const double*)`.
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_compile(
    _builder: *mut DagBuilder,
    _root: u32,
    _out_fn: *mut *mut c_void,
) -> RssnStatus {
    RssnStatus::CompilationError
}

/// Executes a previously compiled JIT function with the given variable input array.
///
/// # Safety
///
/// - `func` must be a valid function pointer previously written by [`rssn_dag_compile`],
///   with signature `double (*)(const double*)`.
/// - `variables` must be a valid pointer to an array of at least as many `f64` values
///   as there are distinct variables in the compiled expression, ordered by `SymbolId`.
/// - Both pointers must remain valid for the duration of this call.
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_execute(func: *const c_void, variables: *const f64) -> f64 {
    if func.is_null() || variables.is_null() {
        return 0.0;
    }
    let result = catch_unwind(|| {
        let compiled_fn: crate::jit::compiler::CompiledExprFn = unsafe { std::mem::transmute(func) };
        compiled_fn(variables)
    });
    result.unwrap_or(0.0)
}

/// Executes a previously compiled JIT function with the given variable input array.
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_execute(_func: *const c_void, _variables: *const f64) -> f64 {
    0.0
}

// =========================================================================
// T6.2 — status-returning v2 surface
// =========================================================================
//
// Each `*_v2` function takes an `out_id: *mut u32` (or equivalent) and
// returns [`RssnStatus`]. This replaces the `u32::MAX` sentinel
// convention used by the original API (`ffi_review §1`). The original
// functions remain as backward-compat wrappers; new C consumers should
// prefer the v2 forms.

/// Creates a new variable node. Status-returning variant.
///
/// On `Success`, writes the new node id to `*out_id`.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
/// - `name` must be a valid, non-null, null-terminated C string valid for this call.
/// - `out_id` must be a valid, non-null, writable `u32` pointer.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_variable_v2(
    builder: *mut DagBuilder,
    name: *const c_char,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || name.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    let result = catch_unwind(|| -> RssnStatus {
        let builder_ref = unsafe { &mut *builder };
        let c_str = unsafe { CStr::from_ptr(name) };
        builder_ref.variable_bytes(c_str.to_bytes()).map_or(
            RssnStatus::InvalidUtf8,
            |id| {
                unsafe { *out_id = id.value() };
                RssnStatus::Success
            },
        )
    });
    result.unwrap_or(RssnStatus::Panic)
}

/// Creates a new constant node. Status-returning variant.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
/// - `out_id` must be a valid, non-null, writable `u32` pointer.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_constant_v2(
    builder: *mut DagBuilder,
    val: f64,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    let result = catch_unwind(|| -> RssnStatus {
        let builder_ref = unsafe { &mut *builder };
        let id = builder_ref.constant(val);
        unsafe { *out_id = id.value() };
        RssnStatus::Success
    });
    result.unwrap_or(RssnStatus::Panic)
}

/// Creates an addition node. Status-returning variant.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
/// - `out_id` must be a valid, non-null, writable `u32` pointer.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_add_v2(
    builder: *mut DagBuilder,
    lhs: u32,
    rhs: u32,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    if lhs == u32::MAX || rhs == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    let result = catch_unwind(|| -> RssnStatus {
        let builder_ref = unsafe { &mut *builder };
        let id = builder_ref.add(DagNodeId::new(lhs), DagNodeId::new(rhs));
        unsafe { *out_id = id.value() };
        RssnStatus::Success
    });
    result.unwrap_or(RssnStatus::Panic)
}

/// Executes a previously compiled JIT function. Status-returning variant.
///
/// On `Success`, writes the result to `*out_val`.
///
/// # Safety
///
/// - `func` must be a valid function pointer previously written by [`rssn_dag_compile`].
/// - `variables` must be a valid pointer to an array of at least as many `f64` values
///   as there are variables in the compiled expression, ordered by `SymbolId`.
/// - `out_val` must be a valid, non-null, writable `f64` pointer.
/// - All pointers must remain valid for the duration of this call.
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_execute_v2(
    func: *const c_void,
    variables: *const f64,
    out_val: *mut f64,
) -> RssnStatus {
    if func.is_null() || variables.is_null() || out_val.is_null() {
        return RssnStatus::NullPointer;
    }
    let result = catch_unwind(|| {
        let compiled_fn: crate::jit::compiler::CompiledExprFn =
            unsafe { std::mem::transmute(func) };
        compiled_fn(variables)
    });
    match result {
        Ok(val) => {
            unsafe { *out_val = val };
            RssnStatus::Success
        }
        Err(_) => RssnStatus::Panic,
    }
}

/// Executes a previously compiled JIT function (stub for non-JIT builds).
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_execute_v2(
    _func: *const c_void,
    _variables: *const f64,
    _out_val: *mut f64,
) -> RssnStatus {
    RssnStatus::CompilationError
}

/// Simplifies an expression. Status-returning variant.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
/// - `out_id` must be a valid, non-null, writable `u32` pointer.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_simplify_v2(
    builder: *mut DagBuilder,
    root: u32,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    if root == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    let result = catch_unwind(|| -> RssnStatus {
        let builder_ref = unsafe { &mut *builder };
        let root_id = DagNodeId::new(root);
        let config = HeuristicConfig::default();
        let mut engine = HeuristicEngine::new(config, SearchStrategy::Greedy);
        let id = engine.simplify(builder_ref, root_id);
        unsafe { *out_id = id.value() };
        RssnStatus::Success
    });
    result.unwrap_or(RssnStatus::Panic)
}

/// JIT compiles a target expression. Status-returning variant.
///
/// On `Success`, writes the compiled function pointer to `*out_fn`.
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer to a `DagBuilder` from [`rssn_dag_new`].
/// - `out_fn` must be a valid, non-null, writable `*mut c_void` pointer.
/// - The compiled function pointer remains valid until the `JITModule` is dropped.
#[cfg(feature = "cranelift-jit")]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_compile_v2(
    builder: *mut DagBuilder,
    root: u32,
    out_fn: *mut *mut c_void,
) -> RssnStatus {
    if builder.is_null() || out_fn.is_null() {
        return RssnStatus::NullPointer;
    }
    if root == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    let result = catch_unwind(|| {
        let builder_ref = unsafe { &mut *builder };
        let root_id = DagNodeId::new(root);
        let ast = crate::ast::convert::dag_to_ast(builder_ref.arena(), root_id);
        let mut compiler = crate::jit::compiler::JitCompiler::new();
        match compiler.compile(&ast) {
            Ok(compiled_fn) => {
                unsafe { *out_fn = compiled_fn as *mut c_void };
                RssnStatus::Success
            }
            Err(_) => RssnStatus::CompilationError,
        }
    });
    result.unwrap_or(RssnStatus::Panic)
}

/// JIT compiles a target expression (stub for non-JIT builds).
///
/// Always returns [`RssnStatus::CompilationError`] when the `cranelift-jit`
/// feature is not enabled.
#[cfg(not(feature = "cranelift-jit"))]
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_compile_v2(
    _builder: *mut DagBuilder,
    _root: u32,
    _out_fn: *mut *mut *mut c_void,
) -> RssnStatus {
    RssnStatus::CompilationError
}

/// C-compatible simplification configuration.
///
/// Pass a pointer to this struct to [`rssn_dag_simplify_with_config`] to
/// override the default heuristic parameters. Pass NULL to use defaults.
#[repr(C)]
pub struct RssnSimplifyConfig {
    /// Maximum rewrite depth (default: 10).
    pub max_depth: u32,
    /// Wall-clock timeout in milliseconds (default: 500).
    pub timeout_ms: u64,
    /// Approximate-pruning aggressiveness in `[0.0, 1.0]` (default: 0.1).
    pub aggressiveness: f64,
}

/// Simplifies an expression using a caller-supplied configuration.
///
/// If `config` is NULL, behaves identically to [`rssn_dag_simplify_v2`].
///
/// # Safety
///
/// - `builder` must be a valid, non-null pointer to a `DagBuilder`.
/// - `out_id` must be a valid, non-null, writable `u32` pointer.
/// - If `config` is non-null, it must point to a valid `RssnSimplifyConfig`.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_simplify_with_config(
    builder: *mut DagBuilder,
    root: u32,
    config: *const RssnSimplifyConfig,
    out_id: *mut u32,
) -> RssnStatus {
    if builder.is_null() || out_id.is_null() {
        return RssnStatus::NullPointer;
    }
    if root == u32::MAX {
        return RssnStatus::InvalidNodeId;
    }
    let (max_depth, timeout_ms, aggressiveness) = if config.is_null() {
        let def = HeuristicConfig::default();
        (
            def.max_depth,
            def.timeout.as_millis() as u64,
            def.simplification_aggressiveness,
        )
    } else {
        let c = unsafe { &*config };
        (c.max_depth as usize, c.timeout_ms, c.aggressiveness)
    };
    let result = catch_unwind(|| -> RssnStatus {
        let builder_ref = unsafe { &mut *builder };
        let root_id = DagNodeId::new(root);
        let cfg = HeuristicConfig::default()
            .max_depth(max_depth)
            .timeout(Duration::from_millis(timeout_ms))
            .simplification_aggressiveness(aggressiveness);
        let mut engine = HeuristicEngine::new(cfg, SearchStrategy::Greedy);
        let id = engine.simplify(builder_ref, root_id);
        unsafe { *out_id = id.value() };
        RssnStatus::Success
    });
    result.unwrap_or(RssnStatus::Panic)
}
