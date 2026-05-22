//! `extern "C"` entry points for the RSSN-Advanced API.
//!
//! Exposes a flat C API, capturing panics securely at the FFI boundary to
//! avoid undefined behavior (UB).

#![allow(unsafe_code)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::panic::catch_unwind;
use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use crate::heuristic::{HeuristicEngine, HeuristicConfig, SearchStrategy};
use super::types::RssnStatus;

/// Creates a new `DagBuilder` context.
///
/// Returns a raw pointer to the builder, or NULL if creation failed or panicked.
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_new() -> *mut DagBuilder {
    let result = catch_unwind(|| {
        Box::into_raw(Box::new(DagBuilder::new()))
    });
    result.unwrap_or(std::ptr::null_mut())
}

/// Releases the memory of a previously allocated `DagBuilder`.
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
#[unsafe(no_mangle)]
pub extern "C" fn rssn_dag_simplify(builder: *mut DagBuilder, root: u32) -> u32 {
    if builder.is_null() {
        return u32::MAX;
    }
    let result = catch_unwind(|| {
        let builder_ref = unsafe { &mut *builder };
        let root_id = DagNodeId::new(root);

        let config = HeuristicConfig::default();
        let engine = HeuristicEngine::new(config, SearchStrategy::Greedy);
        
        engine.simplify(builder_ref, root_id).value()
    });
    result.unwrap_or(u32::MAX)
}

/// JIT compiles a target expression and writes the native function pointer to `out_fn`.
///
/// `out_fn` can be called via `rssn_dag_execute` or cast directly as `double (*)(const double*)`.
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
        let engine = HeuristicEngine::new(config, SearchStrategy::Greedy);
        let id = engine.simplify(builder_ref, root_id);
        unsafe { *out_id = id.value() };
        RssnStatus::Success
    });
    result.unwrap_or(RssnStatus::Panic)
}
