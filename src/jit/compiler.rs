//! Core JIT compiler wrapping Cranelift.
//!
//! `JitCompiler` compiles stack-local AST projection trees into callable,
//! optimized native machine code. The IR generator is **iterative**
//! (work-stack + SSA-value stack) so even an expression a million nodes
//! deep does not blow the OS stack — see `jit_review §2`. It folds a
//! peephole pass over the per-node IR emission so that `x + 0`, `x * 1`,
//! `x * 0`, etc. cost zero instructions (`jit_review §1` / `§2`).

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use cranelift_codegen::Context;
use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Signature, Value, types};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::ast::projection::{AstNode, AstProjection};
use crate::dag::symbol::{FnId, OpKind, SymbolKind};

/// A JIT-compiled expression function pointer.
///
/// It takes a pointer to an array of variable values (`*const f64`),
/// ordered by their `SymbolId` values, and returns the computed float result.
pub type CompiledExprFn = extern "C" fn(*const f64) -> f64;

/// User-supplied native function: one `f64` argument, one `f64` return.
///
/// The common math-library signature (`sin`, `cos`, `log`, …). Registered via
/// [`JitCompiler::register_custom_function`].
pub type CustomFn1 = extern "C" fn(f64) -> f64;

/// User-supplied native function: two `f64` arguments, one `f64` return.
///
/// Suitable for two-argument math functions (`pow`, `atan2`, …). Registered
/// via [`JitCompiler::register_custom_function_2`].
pub type CustomFn2 = extern "C" fn(f64, f64) -> f64;

/// User-supplied native function: three `f64` arguments, one `f64` return.
///
/// Suitable for three-argument operations (`fma`, `clamp`, …). Registered
/// via [`JitCompiler::register_custom_function_3`].
pub type CustomFn3 = extern "C" fn(f64, f64, f64) -> f64;

extern "C" fn jit_powf(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

extern "C" fn jit_fmod(lhs: f64, rhs: f64) -> f64 {
    lhs % rhs
}

/// Entry in the custom function registry: function pointer + argument count.
#[derive(Clone, Copy)]
struct CustomFnEntry {
    /// Raw function pointer (cast to `usize` for `Send`/`Sync` safety).
    ptr: usize,
    /// Number of `f64` arguments the function accepts (1, 2, or 3).
    arity: u8,
}

/// Shared registry of custom function pointers, keyed by `FnId.0`.
///
/// Stored as `usize` (not `*const u8`) so the type is `Send`/`Sync`
/// without unsafe markers. Uses `RwLock` so concurrent readers (the
/// symbol-lookup closure) never block each other; only `register_custom_function`
/// takes a write lock.
type CustomFnRegistry = Arc<RwLock<HashMap<u32, CustomFnEntry>>>;

/// The primary compiler context for compiling symbolic expressions to native code.
pub struct JitCompiler {
    module: JITModule,
    builder_ctx: FunctionBuilderContext,
    /// Shared with the symbol-lookup closure baked into the
    /// `JITModule`. Late `register_custom_function` calls update this
    /// map; the closure consults it whenever Cranelift needs to
    /// resolve an unknown symbol.
    custom_fns: CustomFnRegistry,
    /// Reusable work stack for `compile_ast_iterative`. Cleared at the
    /// start of each compile call; kept here to amortise allocation cost
    /// across repeated compilations.
    work_stack: Vec<Frame>,
    /// Reusable SSA-value stack for `compile_ast_iterative`.
    work_values: Vec<Value>,
}

impl std::fmt::Debug for JitCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitCompiler").finish_non_exhaustive()
    }
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl JitCompiler {
    /// Attempts to create a new `JitCompiler` for the host target.
    ///
    /// Returns `Err(JitError::InitFailed)` if the Cranelift backend or
    /// native ISA cannot be initialised — for example, on an unsupported
    /// architecture or in a cross-compilation environment without a
    /// registered native target (`error_review §4`).
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::JitError::InitFailed`] if any step of the
    /// Cranelift backend initialisation fails.
    pub fn try_new() -> Result<Self, crate::error::JitError> {
        let isa_builder = cranelift_native::builder()
            .map_err(|_| crate::error::JitError::InitFailed)?;

        let mut flag_builder = cranelift_codegen::settings::builder();
        cranelift_codegen::settings::Configurable::set(&mut flag_builder, "opt_level", "speed")
            .map_err(|_| crate::error::JitError::InitFailed)?;

        let isa = isa_builder
            .finish(cranelift_codegen::settings::Flags::new(flag_builder))
            .map_err(|_| crate::error::JitError::InitFailed)?;

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        builder.symbol("powf", jit_powf as *const u8);
        builder.symbol("fmod", jit_fmod as *const u8);

        let custom_fns: CustomFnRegistry = Arc::new(RwLock::new(HashMap::new()));
        let lookup_registry = Arc::clone(&custom_fns);
        builder.symbol_lookup_fn(Box::new(move |name: &str| -> Option<*const u8> {
            let id_str = name.strip_prefix("rssn_custom_fn_")?;
            let id: u32 = id_str.parse().ok()?;
            let guard = lookup_registry
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.get(&id).map(|entry| entry.ptr as *const u8)
        }));

        let module = JITModule::new(builder);
        Ok(Self {
            module,
            builder_ctx: FunctionBuilderContext::new(),
            custom_fns,
            work_stack: Vec::with_capacity(64),
            work_values: Vec::with_capacity(64),
        })
    }

    /// Creates a new `JitCompiler` instance initialized for the host target.
    ///
    /// # Panics
    /// Panics if the host native target cannot be built. For a fallible
    /// variant, use [`Self::try_new`].
    #[must_use]
    pub fn new() -> Self {
        Self::try_new().expect("JIT compiler initialization failed")
    }

    /// Registers a user-defined `extern "C" fn(f64) -> f64` so the JIT
    /// can resolve `SymbolKind::Function(fn_id)` references at link
    /// time. May be called any time before the next [`Self::compile`].
    ///
    /// The `fn_id` must match whatever the symbolic layer assigned to
    /// the corresponding function name (typically via `DagBuilder`).
    pub fn register_custom_function(&self, fn_id: FnId, func: CustomFn1) {
        // The cast `func as usize` is a no-op pointer cast; the value
        // is later cast back to `*const u8` inside the lookup closure.
        let mut guard = self
            .custom_fns
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.insert(fn_id.0, CustomFnEntry { ptr: func as usize, arity: 1 });
    }

    /// Registers a user-defined `extern "C" fn(f64, f64) -> f64` so the JIT
    /// can resolve `SymbolKind::Function(fn_id)` references at link time.
    ///
    /// Two-argument variant of [`Self::register_custom_function`], suitable
    /// for functions like `pow`, `atan2`, or user-defined binary operators
    /// (`jit_review §3.1`).
    pub fn register_custom_function_2(&self, fn_id: FnId, func: CustomFn2) {
        let mut guard = self
            .custom_fns
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.insert(fn_id.0, CustomFnEntry { ptr: func as usize, arity: 2 });
    }

    /// Registers a user-defined `extern "C" fn(f64, f64, f64) -> f64` so the JIT
    /// can resolve `SymbolKind::Function(fn_id)` references at link time.
    ///
    /// Three-argument variant of [`Self::register_custom_function`], suitable
    /// for `fma`, `clamp`, and similar ternary operations (`jit_review §3.1`).
    pub fn register_custom_function_3(&self, fn_id: FnId, func: CustomFn3) {
        let mut guard = self
            .custom_fns
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.insert(fn_id.0, CustomFnEntry { ptr: func as usize, arity: 3 });
    }

    /// Compiles an `AstProjection` expression into a native callable function.
    ///
    /// # Errors
    /// Returns a [`crate::error::JitError`] if compilation or linking fails.
    pub fn compile(&mut self, ast: &AstProjection) -> Result<CompiledExprFn, crate::error::JitError> {
        if ast.is_empty() {
            return crate::error::cold_jit_error_malformed_node();
        }

        let mut ctx = Context::new();

        // Define signature: fn(*const f64) -> f64
        let ptr_type = self.module.target_config().pointer_type();
        ctx.func.signature.params.push(AbiParam::new(ptr_type));
        ctx.func.signature.returns.push(AbiParam::new(types::F64));

        let mut func_builder = FunctionBuilder::new(&mut ctx.func, &mut self.builder_ctx);
        let entry_block = func_builder.create_block();
        func_builder.append_block_params_for_function_params(entry_block);
        func_builder.switch_to_block(entry_block);
        func_builder.seal_block(entry_block);

        let vars_ptr = func_builder.block_params(entry_block)[0];

        // Declare the powf helper function in the current function context.
        let mut powf_sig = Signature::new(self.module.target_config().default_call_conv);
        powf_sig.params.push(AbiParam::new(types::F64));
        powf_sig.params.push(AbiParam::new(types::F64));
        powf_sig.returns.push(AbiParam::new(types::F64));

        let powf_name = self
            .module
            .declare_function("powf", Linkage::Import, &powf_sig)
            .map_err(|_| crate::error::JitError::InitFailed)?;
        let powf_func_ref = self
            .module
            .declare_func_in_func(powf_name, func_builder.func);

        // fmod: same binary signature as powf (two f64 → one f64).
        let fmod_name = self
            .module
            .declare_function("fmod", Linkage::Import, &powf_sig)
            .map_err(|_| crate::error::JitError::InitFailed)?;
        let fmod_func_ref = self
            .module
            .declare_func_in_func(fmod_name, func_builder.func);

        // Walk the AST once and import every distinct custom function
        // it references. Refuse to compile if any referenced id was
        // not registered via `register_custom_function`.
        let mut custom_refs: HashMap<u32, cranelift_codegen::ir::FuncRef> = HashMap::new();

        // Snapshot the registry (id → arity) under a read lock, then drop
        // it before any module work — keeps the lock window minimal.
        let registered_entries: HashMap<u32, u8> = {
            let guard = self
                .custom_fns
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.iter().map(|(&id, e)| (id, e.arity)).collect()
        };

        // Capture call_conv before any mutable borrows of self.module.
        let default_call_conv = self.module.target_config().default_call_conv;

        // Build per-arity signatures (1-, 2-, 3-arg f64→f64).
        let make_fn_sig = |arity: u8| -> Signature {
            let mut sig = Signature::new(default_call_conv);
            for _ in 0..arity {
                sig.params.push(AbiParam::new(types::F64));
            }
            sig.returns.push(AbiParam::new(types::F64));
            sig
        };

        for node in &ast.nodes {
            if let SymbolKind::Function(fn_id) = node.kind {
                if custom_refs.contains_key(&fn_id.0) {
                    continue;
                }
                let arity = registered_entries.get(&fn_id.0).copied()
                    .ok_or(crate::error::JitError::UnknownFunction)?;
                let sig = make_fn_sig(arity);
                let sym = format!("rssn_custom_fn_{}", fn_id.0);
                let fid = self
                    .module
                    .declare_function(&sym, Linkage::Import, &sig)
                    .map_err(|_| crate::error::JitError::InitFailed)?;
                let fr = self.module.declare_func_in_func(fid, func_builder.func);
                custom_refs.insert(fn_id.0, fr);
            }
        }

        // Clear and reuse the scratch buffers from the previous compile call
        // to avoid per-call heap allocation for the work-stack and value-stack.
        self.work_stack.clear();
        self.work_values.clear();
        let root_val = compile_ast_iterative(
            ast,
            &mut func_builder,
            vars_ptr,
            powf_func_ref,
            fmod_func_ref,
            &custom_refs,
            &mut self.work_stack,
            &mut self.work_values,
        )?;

        func_builder.ins().return_(&[root_val]);
        func_builder.finalize();

        let fn_name = format!("expr_{}", ast.nodes[0].dag_id.0);
        let func_id = self
            .module
            .declare_function(&fn_name, Linkage::Export, &ctx.func.signature)
            .map_err(|_| crate::error::JitError::VerifierRejected)?;

        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|_| crate::error::JitError::VerifierRejected)?;

        self.module.clear_context(&mut ctx);

        self.module
            .finalize_definitions()
            .map_err(|_| crate::error::JitError::VerifierRejected)?;

        let code_ptr = self.module.get_finalized_function(func_id);

        // SAFETY: Cranelift returns the address of native code matching
        // exactly the signature we declared above (`fn(*const f64) -> f64`).
        let compiled_fn: CompiledExprFn = unsafe { std::mem::transmute(code_ptr) };
        Ok(compiled_fn)
    }

    /// Convenience wrapper: converts the DAG subgraph rooted at `root`
    /// to an `AstProjection` and compiles it in one call.
    ///
    /// This is the idiomatic entry point when the caller already holds a
    /// `DagArena` and wants a native function without manually calling
    /// [`crate::ast::convert::dag_to_ast`] first.
    ///
    /// # Errors
    /// Returns a [`crate::error::JitError`] if the AST conversion or compilation fails.
    pub fn compile_dag(
        &mut self,
        arena: &crate::dag::arena::DagArena,
        root: crate::dag::node::DagNodeId,
    ) -> Result<CompiledExprFn, crate::error::JitError> {
        let ast = crate::ast::convert::dag_to_ast(arena, root);
        self.compile(&ast)
    }
}

// =========================================================================
// Iterative codegen (T2.1)
// =========================================================================

/// Work-stack frame for the iterative post-order walk.
///
/// `cursor` advances 0..=arity. The frame is "ready to emit" exactly when
/// `cursor == arity`; at that point the top `arity` entries of the value
/// stack are the SSA values for this node's children.
struct Frame {
    idx: usize,
    arity: usize,
    cursor: usize,
}

fn compile_ast_iterative(
    ast: &AstProjection,
    builder: &mut FunctionBuilder<'_>,
    vars_ptr: Value,
    powf_func_ref: cranelift_codegen::ir::FuncRef,
    fmod_func_ref: cranelift_codegen::ir::FuncRef,
    custom_refs: &HashMap<u32, cranelift_codegen::ir::FuncRef>,
    stack: &mut Vec<Frame>,
    mut values: &mut Vec<Value>,
) -> Result<Value, crate::error::JitError> {
    // Callers clear these before passing — guaranteed by `compile()`.

    // FMA peephole: maps each Mul-result Value to its two input factors so
    // that when an enclosing Add is emitted we can fold `a*b + c` into one
    // `fma(a, b, c)` instruction (jit_review §4 / simd_review §4).
    let mut mul_factors: HashMap<Value, (Value, Value)> = HashMap::new();

    // Seed with the root.
    let root_node = ast
        .nodes
        .first()
        .ok_or(crate::error::JitError::MalformedNode)?;
    stack.push(Frame {
        idx: 0,
        arity: root_node.children.len(),
        cursor: 0,
    });

    while let Some(top) = stack.last_mut() {
        // Decide whether to push a child or emit this node, then act —
        // splitting the decision from the action keeps the borrow
        // checker happy when we go from `last_mut()` to `push/pop`.
        let action = if top.cursor < top.arity {
            let Some(node) = ast.nodes.get(top.idx) else {
                return Err(crate::error::JitError::MalformedNode);
            };
            let Some(&child_ptr) = node.children.as_slice().get(top.cursor) else {
                return Err(crate::error::JitError::MalformedNode);
            };
            let child_idx = child_ptr.resolve(top.idx)
                .ok_or(crate::error::JitError::MalformedNode)?;
            top.cursor += 1;
            Action::Descend(child_idx)
        } else {
            Action::Emit(top.idx, top.arity)
        };

        match action {
            Action::Descend(child_idx) => {
                let Some(child_node) = ast.nodes.get(child_idx) else {
                    return Err(crate::error::JitError::MalformedNode);
                };
                stack.push(Frame {
                    idx: child_idx,
                    arity: child_node.children.len(),
                    cursor: 0,
                });
            }
            Action::Emit(idx, arity) => {
                stack.pop();
                emit_one_node(
                    ast,
                    idx,
                    arity,
                    builder,
                    vars_ptr,
                    powf_func_ref,
                    fmod_func_ref,
                    custom_refs,
                    &mut values,
                    &mut mul_factors,
                )?;
            }
        }
    }

    let result = values.pop().ok_or(crate::error::JitError::MalformedNode)?;
    if !values.is_empty() {
        return Err(crate::error::JitError::VerifierRejected);
    }
    Ok(result)
}

enum Action {
    Descend(usize),
    Emit(usize, usize),
}

#[allow(clippy::too_many_arguments)]
fn emit_one_node(
    ast: &AstProjection,
    idx: usize,
    arity: usize,
    builder: &mut FunctionBuilder<'_>,
    vars_ptr: Value,
    powf_func_ref: cranelift_codegen::ir::FuncRef,
    fmod_func_ref: cranelift_codegen::ir::FuncRef,
    custom_refs: &HashMap<u32, cranelift_codegen::ir::FuncRef>,
    values: &mut Vec<Value>,
    mul_factors: &mut HashMap<Value, (Value, Value)>,
) -> Result<(), crate::error::JitError> {
    let node = &ast.nodes[idx];

    match node.kind {
        SymbolKind::Constant(_) => {
            values.push(builder.ins().f64const(node.value)); // AstNode.value: f64 (not Option)
        }
        SymbolKind::Variable(sym_id) => {
            let val = emit_variable_load(builder, vars_ptr, sym_id.0);
            values.push(val);
        }
        SymbolKind::Operator(op) => {
            let split_at = values.len().checked_sub(arity)
                .ok_or(crate::error::JitError::MalformedNode)?;
            // Take children out in order — the iterative walker pushes
            // left-to-right, so children[0..arity] are already correct.
            let child_vals: Vec<Value> = values.drain(split_at..).collect();
            let result = emit_operator(builder, op, &child_vals, powf_func_ref, fmod_func_ref, node, mul_factors)?;
            values.push(result);
        }
        SymbolKind::Function(fn_id) => {
            // T2.6: resolve `SymbolKind::Function(fn_id)` to the
            // FuncRef declared in `compile()` and emit a call with
            // 1, 2, or 3 f64 arguments depending on the registration
            // arity (`jit_review §3.1`).
            let func_ref = custom_refs.get(&fn_id.0).copied()
                .ok_or(crate::error::JitError::UnknownFunction)?;
            let split_at = values.len().checked_sub(arity)
                .ok_or(crate::error::JitError::MalformedNode)?;
            let child_vals: Vec<Value> = values.drain(split_at..).collect();
            if child_vals.is_empty() || child_vals.len() > 3 {
                return Err(crate::error::JitError::MalformedNode);
            }
            let call = builder.ins().call(func_ref, &child_vals);
            let result = builder
                .inst_results(call)
                .first()
                .copied()
                .ok_or(crate::error::JitError::VerifierRejected)?;
            values.push(result);
        }
    }
    Ok(())
}

fn emit_variable_load(
    builder: &mut FunctionBuilder<'_>,
    vars_ptr: Value,
    sym_idx: u32,
) -> Value {
    // SymbolId * 8 (one f64 per variable). u32 → i64 via i64 is safe.
    let offset = i64::from(sym_idx).wrapping_mul(8);
    let addr = builder.ins().iadd_imm(vars_ptr, offset);
    builder.ins().load(types::F64, MemFlags::new(), addr, 0)
}

/// Emits IR for a single algebraic operator, applying peephole identity
/// simplifications first (T2.5). The peephole runs at IR time — the
/// constant arguments here are whatever the codegen walker materialised
/// into `child_vals`, which may include `f64const` instructions we
/// emitted moments ago.
///
/// `mul_factors` tracks the two inputs of each recently-emitted `fmul`
/// result; this enables the FMA peephole in `OpKind::Add` that folds
/// `a*b + c` into a single `fma(a, b, c)` instruction (jit_review §4).
fn emit_operator(
    builder: &mut FunctionBuilder<'_>,
    op: OpKind,
    child_vals: &[Value],
    powf_func_ref: cranelift_codegen::ir::FuncRef,
    fmod_func_ref: cranelift_codegen::ir::FuncRef,
    ast_node: &AstNode,
    mul_factors: &mut HashMap<Value, (Value, Value)>,
) -> Result<Value, crate::error::JitError> {
    use crate::jit::primitives::{simplify_add, simplify_mul};

    // Look up the immediate constant behind each child SSA value, if any.
    // Cranelift exposes this via `func.dfg.value_def`; for `f64const` it
    // returns an `InstructionData::UnaryIeee64` with the value.
    let constants: Vec<Option<f64>> = child_vals
        .iter()
        .map(|v| constant_behind(builder, *v))
        .collect();

    let arity = child_vals.len();
    match op {
        OpKind::Add => {
            if arity != 2 {
                return Err(crate::error::JitError::MalformedNode);
            }
            // Peephole: `x + 0 → x`, `0 + x → x`, `c1 + c2 → const`.
            match (constants[0], constants[1]) {
                (Some(l), Some(r)) => {
                    let folded = simplify_add(l, r).unwrap_or(l + r);
                    Ok(builder.ins().f64const(folded))
                }
                (Some(0.0), _) => Ok(child_vals[1]),
                (_, Some(0.0)) => Ok(child_vals[0]),
                _ => {
                    // FMA peephole (jit_review §4 / simd_review §4):
                    // If the left child is a Mul result whose two factors
                    // were recorded in `mul_factors`, fold `(a*b) + c`
                    // → `fma(a, b, c)`.  Check right side too for
                    // commutativity: `c + (a*b)` → `fma(a, b, c)`.
                    if let Some(&(a, b)) = mul_factors.get(&child_vals[0]) {
                        Ok(builder.ins().fma(a, b, child_vals[1]))
                    } else if let Some(&(a, b)) = mul_factors.get(&child_vals[1]) {
                        Ok(builder.ins().fma(a, b, child_vals[0]))
                    } else {
                        Ok(builder.ins().fadd(child_vals[0], child_vals[1]))
                    }
                }
            }
        }
        OpKind::Sub => {
            if arity != 2 {
                return Err(crate::error::JitError::MalformedNode);
            }
            match (constants[0], constants[1]) {
                (Some(l), Some(r)) => Ok(builder.ins().f64const(l - r)),
                (_, Some(0.0)) => Ok(child_vals[0]),
                _ => Ok(builder.ins().fsub(child_vals[0], child_vals[1])),
            }
        }
        OpKind::Mul => {
            if arity != 2 {
                return Err(crate::error::JitError::MalformedNode);
            }
            // Peephole: `x * 0 → 0`, `x * 1 → x`, `c1 * c2 → const`,
            // `x * 2.0 → x + x` (single fadd is often faster than fmul
            // by a literal 2.0 on modern FP pipelines — jit_review §4).
            match (constants[0], constants[1]) {
                (Some(l), Some(r)) => {
                    let folded = simplify_mul(l, r).unwrap_or(l * r);
                    Ok(builder.ins().f64const(folded))
                }
                (Some(0.0), _) | (_, Some(0.0)) => Ok(builder.ins().f64const(0.0)),
                (Some(1.0), _) => Ok(child_vals[1]),
                (_, Some(1.0)) => Ok(child_vals[0]),
                // `x * 2.0 → x + x` and `2.0 * x → x + x`: replacing a
                // multiply by a power-of-two constant with an additive
                // self-addition avoids an FP multiply unit stall on CPUs
                // where FADD has lower latency than FMUL.
                (Some(2.0), _) => Ok(builder.ins().fadd(child_vals[1], child_vals[1])),
                (_, Some(2.0)) => Ok(builder.ins().fadd(child_vals[0], child_vals[0])),
                _ => {
                    // Record this Mul's factors so an enclosing Add can
                    // optionally fold into FMA instead of separate mul+add.
                    let result = builder.ins().fmul(child_vals[0], child_vals[1]);
                    mul_factors.insert(result, (child_vals[0], child_vals[1]));
                    Ok(result)
                }
            }
        }
        OpKind::Div => {
            if arity != 2 {
                return Err(crate::error::JitError::MalformedNode);
            }
            let lhs = child_vals[0];
            let rhs = child_vals[1];

            // Both operands are compile-time constants: fold entirely.
            if let (Some(lval), Some(rval)) = (constants[0], constants[1]) {
                // IEEE-754: x / 0 == ±Inf or NaN — we map to NaN for
                // consistency with the runtime `select` path below.
                let result = if rval == 0.0 { f64::NAN } else { lval / rval };
                return Ok(builder.ins().f64const(result));
            }
            // Constant zero denominator: no runtime division needed.
            if constants[1] == Some(0.0) {
                return Ok(builder.ins().f64const(f64::NAN));
            }

            // Runtime denominator: emit `select(rhs==0, NaN, lhs/rhs)` so
            // divide-by-zero yields NaN instead of trapping (IEEE-754 §6.2,
            // matches `parallel::solver` behaviour). Using `select` avoids
            // a conditional branch and lets the CPU execute both sides.
            let zero = builder.ins().f64const(0.0);
            let nan_val = builder.ins().f64const(f64::NAN);
            let is_zero = builder.ins().fcmp(FloatCC::Equal, rhs, zero);
            let div_result = builder.ins().fdiv(lhs, rhs);
            Ok(builder.ins().select(is_zero, nan_val, div_result))
        }
        OpKind::Pow => {
            if arity != 2 {
                return Err(crate::error::JitError::MalformedNode);
            }
            // Peephole: `x ^ 0 → 1`, `x ^ 1 → x`, `c1 ^ c2 → const`.
            match (constants[0], constants[1]) {
                (Some(l), Some(r)) => Ok(builder.ins().f64const(l.powf(r))),
                (_, Some(0.0)) => Ok(builder.ins().f64const(1.0)),
                (_, Some(1.0)) => Ok(child_vals[0]),
                _ => {
                    let call = builder.ins().call(powf_func_ref, child_vals);
                    Ok(builder.inst_results(call)[0])
                }
            }
        }
        OpKind::Mod => {
            if arity != 2 {
                return Err(crate::error::JitError::MalformedNode);
            }
            let lhs = child_vals[0];
            let rhs = child_vals[1];
            if let (Some(lval), Some(rval)) = (constants[0], constants[1]) {
                let result = if rval == 0.0 { f64::NAN } else { lval % rval };
                return Ok(builder.ins().f64const(result));
            }
            if constants[1] == Some(0.0) {
                return Ok(builder.ins().f64const(f64::NAN));
            }
            // Cranelift has no native frem for f64; call jit_fmod helper.
            // Guard runtime zero-denominator with select(rhs==0, NaN, fmod(lhs,rhs)).
            let zero = builder.ins().f64const(0.0);
            let nan_val = builder.ins().f64const(f64::NAN);
            let is_zero = builder.ins().fcmp(FloatCC::Equal, rhs, zero);
            let call = builder.ins().call(fmod_func_ref, &[lhs, rhs]);
            let rem_result = builder.inst_results(call)[0];
            Ok(builder.ins().select(is_zero, nan_val, rem_result))
        }
        OpKind::Neg => {
            if arity != 1 {
                return Err(crate::error::JitError::MalformedNode);
            }
            if let Some(c) = constants[0] {
                return Ok(builder.ins().f64const(-c));
            }
            Ok(builder.ins().fneg(child_vals[0]))
        }
    }
    .inspect(|_| {
        // Reference `ast_node` to keep the parameter alive for callers
        // that future-extend with kind-specific peepholes.
        let _ = ast_node;
    })
}

/// Returns the constant `f64` behind `v` if `v` is the result of a
/// `f64const` instruction; otherwise `None`.
///
/// Used by the IR-time peephole pass so `x + 0 → x`, `x * 0 → 0` etc.
/// hold even when one side is a freshly emitted constant Value.
fn constant_behind(builder: &FunctionBuilder<'_>, v: Value) -> Option<f64> {
    use cranelift_codegen::ir::{InstructionData, ValueDef};
    let dfg = &builder.func.dfg;
    let ValueDef::Result(inst, _) = dfg.value_def(v) else {
        return None;
    };
    if let InstructionData::UnaryIeee64 { imm, .. } = dfg.insts[inst] {
        Some(f64::from_bits(imm.bits()))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::convert::dag_to_ast;
    use crate::dag::builder::DagBuilder;
    use crate::parser::expr::parse_expression;

    #[test]
    fn test_jit_compile_and_execute() {
        let mut builder = DagBuilder::new();
        let id = parse_expression("x * 2.5 + y ^ 2.0", &mut builder).unwrap();
        let ast = dag_to_ast(builder.arena(), id);

        let mut compiler = JitCompiler::new();
        let compiled_fn = compiler.compile(&ast).unwrap();

        let vars = vec![3.0, 4.0];
        let result = compiled_fn(vars.as_ptr());
        let expected = 3.0 * 2.5 + 4.0_f64.powf(2.0);
        assert!((result - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jit_divide_by_zero_yields_nan() {
        let mut builder = DagBuilder::new();
        let id = parse_expression("x / y", &mut builder).unwrap();
        let ast = dag_to_ast(builder.arena(), id);

        let mut compiler = JitCompiler::new();
        let compiled_fn = compiler.compile(&ast).unwrap();

        let safe_vars = vec![10.0, 2.0];
        let safe_res = compiled_fn(safe_vars.as_ptr());
        assert!((safe_res - 5.0).abs() < f64::EPSILON);

        // Division by zero must return NaN (not trap).
        let zero_vars = vec![10.0, 0.0];
        let nan_res = compiled_fn(zero_vars.as_ptr());
        assert!(nan_res.is_nan(), "x/0 must be NaN; got {nan_res}");
    }

    #[test]
    fn test_jit_deep_chain_no_stack_overflow() {
        // Build (((x+x)+x)+x)... 2500 deep.
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let mut acc = x;
        for _ in 0..2500 {
            acc = b.add(acc, x);
        }
        let ast = dag_to_ast(b.arena(), acc);

        let mut compiler = JitCompiler::new();
        let compiled_fn = compiler.compile(&ast).unwrap();
        let vars = vec![1.0];
        let result = compiled_fn(vars.as_ptr());
        // (((1+1)+1)+1)... = 1 * (2501 ones).
        let expected = 2501.0;
        assert!((result - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn test_peephole_x_plus_zero() {
        let mut b = DagBuilder::new();
        let id = parse_expression("x + 0", &mut b).unwrap();
        let ast = dag_to_ast(b.arena(), id);
        let mut compiler = JitCompiler::new();
        let f = compiler.compile(&ast).unwrap();
        assert!((f([3.5_f64].as_ptr()) - 3.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_peephole_x_times_zero() {
        let mut b = DagBuilder::new();
        let id = parse_expression("x * 0", &mut b).unwrap();
        let ast = dag_to_ast(b.arena(), id);
        let mut compiler = JitCompiler::new();
        let f = compiler.compile(&ast).unwrap();
        assert!(f([42.0_f64].as_ptr()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_peephole_constant_fold() {
        let mut b = DagBuilder::new();
        // 3 + 4 should compile to a constant `7.0` (peephole folds it).
        let id = parse_expression("3 + 4", &mut b).unwrap();
        let ast = dag_to_ast(b.arena(), id);
        let mut compiler = JitCompiler::new();
        let f = compiler.compile(&ast).unwrap();
        let r = f([].as_ptr());
        assert!((r - 7.0).abs() < f64::EPSILON);
    }

    extern "C" fn rssn_test_double(x: f64) -> f64 {
        x * 2.0
    }

    #[test]
    fn test_custom_function_jit_round_trip() {
        use crate::dag::metadata::NodeFlags;
        use crate::dag::symbol::{FnId, SymbolKind as SK};

        let mut b = DagBuilder::new();
        let x = b.variable("x");
        // Build `double(x)` via a custom function id #42.
        let fn_id = FnId(42);
        let expr = b.operator(SK::Function(fn_id), &[x], NodeFlags::EMPTY);
        let ast = dag_to_ast(b.arena(), expr);

        let mut compiler = JitCompiler::new();
        compiler.register_custom_function(fn_id, rssn_test_double);
        let f = compiler.compile(&ast).expect("compile with custom fn");

        let vars = vec![3.5_f64];
        let r = f(vars.as_ptr());
        assert!((r - 7.0).abs() < f64::EPSILON, "expected 3.5 * 2 = 7.0");
    }

    #[test]
    fn test_custom_function_unregistered_fails_cleanly() {
        use crate::dag::metadata::NodeFlags;
        use crate::dag::symbol::{FnId, SymbolKind as SK};

        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let expr = b.operator(SK::Function(FnId(99)), &[x], NodeFlags::EMPTY);
        let ast = dag_to_ast(b.arena(), expr);

        let mut compiler = JitCompiler::new();
        // No `register_custom_function` call → compile must error.
        let err = compiler.compile(&ast).expect_err("must error");
        assert_eq!(
            err,
            crate::error::JitError::UnknownFunction,
            "unregistered fn id must yield UnknownFunction; got: {err:?}"
        );
    }
}
