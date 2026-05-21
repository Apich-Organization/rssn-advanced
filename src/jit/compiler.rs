//! Core JIT compiler wrapping Cranelift.
//!
//! `JitCompiler` compiles stack-local AST projection trees into callable,
//! optimized native machine code. The IR generator is now **iterative**
//! (work-stack + SSA-value stack) so even an expression a million nodes
//! deep does not blow the OS stack — see `jit_review §2`. It also wires
//! [`crate::jit::codegen::emit_prefetch_hint`] in front of every memory
//! load (`jit_review §1`), and folds a peephole pass over the per-node
//! IR emission so that `x + 0`, `x * 1`, `x * 0`, etc. cost zero
//! instructions (`jit_review §1` / `§2`).

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cranelift_codegen::Context;
use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Signature, TrapCode, Value, types};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::ast::projection::{AstNode, AstProjection};
use crate::dag::symbol::{FnId, OpKind, SymbolKind};
use crate::jit::codegen::emit_prefetch_hint;

/// A JIT-compiled expression function pointer.
///
/// It takes a pointer to an array of variable values (`*const f64`),
/// ordered by their `SymbolId` values, and returns the computed float result.
pub type CompiledExprFn = extern "C" fn(*const f64) -> f64;

/// User-supplied native function exposed to the JIT.
///
/// Takes one `f64` and returns one `f64` — the common math-library
/// signature (`sin`, `cos`, `log`, …). Registered via
/// [`JitCompiler::register_custom_function`].
pub type CustomFn1 = extern "C" fn(f64) -> f64;

extern "C" fn jit_powf(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

/// Shared registry of custom function pointers, keyed by `FnId.0`.
///
/// Stored as `usize` (not `*const u8`) so the type is `Send`/`Sync`
/// without unsafe markers. The closure registered with
/// `JITBuilder::symbol_lookup_fn` (see [`JitCompiler::new`]) queries
/// this map at link time, so [`Self::register_custom_function`] may
/// be called any time **before the next `compile()`**.
type CustomFnRegistry = Arc<Mutex<HashMap<u32, usize>>>;

/// The primary compiler context for compiling symbolic expressions to native code.
pub struct JitCompiler {
    module: JITModule,
    builder_ctx: FunctionBuilderContext,
    /// Shared with the symbol-lookup closure baked into the
    /// `JITModule`. Late `register_custom_function` calls update this
    /// map; the closure consults it whenever Cranelift needs to
    /// resolve an unknown symbol.
    custom_fns: CustomFnRegistry,
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
    /// Creates a new `JitCompiler` instance initialized for the host target.
    ///
    /// # Panics
    /// Panics if the host native target cannot be built.
    #[must_use]
    pub fn new() -> Self {
        // allow-panic: init-only — the JIT cannot operate without a target ISA.
        let isa_builder =
            cranelift_native::builder().expect("Failed to detect native host platform");

        // Optimizing compiler flags.
        let mut flag_builder = cranelift_codegen::settings::builder();
        // allow-panic: init-only — `opt_level` is a fixed string literal.
        cranelift_codegen::settings::Configurable::set(&mut flag_builder, "opt_level", "speed")
            .expect("Failed to set opt_level");

        // allow-panic: init-only — Cranelift backend setup failure is non-recoverable.
        let isa = isa_builder
            .finish(cranelift_codegen::settings::Flags::new(flag_builder))
            .expect("Failed to build target ISA");

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        builder.symbol("powf", jit_powf as *const u8);

        // Custom function registry: shared between the JitCompiler
        // and the lookup-fn closure baked into the JITModule. The
        // closure runs whenever Cranelift hits an unknown symbol
        // during link/finalize; we map our `rssn_custom_fn_<id>`
        // naming convention back to the raw function pointer.
        let custom_fns: CustomFnRegistry = Arc::new(Mutex::new(HashMap::new()));
        let lookup_registry = Arc::clone(&custom_fns);
        builder.symbol_lookup_fn(Box::new(move |name: &str| -> Option<*const u8> {
            let id_str = name.strip_prefix("rssn_custom_fn_")?;
            let id: u32 = id_str.parse().ok()?;
            let guard = lookup_registry
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.get(&id).map(|addr| *addr as *const u8)
        }));

        let module = JITModule::new(builder);
        Self {
            module,
            builder_ctx: FunctionBuilderContext::new(),
            custom_fns,
        }
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
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.insert(fn_id.0, func as usize);
    }

    /// Compiles an `AstProjection` expression into a native callable function.
    ///
    /// # Errors
    /// Returns a message string if the compilation or linking step fails.
    pub fn compile(&mut self, ast: &AstProjection) -> Result<CompiledExprFn, String> {
        if ast.is_empty() {
            return Err("Cannot compile empty AST projection".to_owned());
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
            .map_err(|e| format!("Failed to declare powf import: {e:?}"))?;
        let powf_func_ref = self
            .module
            .declare_func_in_func(powf_name, func_builder.func);

        // Walk the AST once and import every distinct custom function
        // it references. Refuse to compile if any referenced id was
        // not registered via `register_custom_function`.
        let mut custom_sig = Signature::new(self.module.target_config().default_call_conv);
        custom_sig.params.push(AbiParam::new(types::F64));
        custom_sig.returns.push(AbiParam::new(types::F64));
        let mut custom_refs: HashMap<u32, cranelift_codegen::ir::FuncRef> = HashMap::new();

        // Snapshot the registry under the lock, then drop it before
        // doing any module work — keeps the lock window minimal.
        let registered_ids: std::collections::HashSet<u32> = {
            let guard = self
                .custom_fns
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.keys().copied().collect()
        };
        for node in &ast.nodes {
            if let SymbolKind::Function(fn_id) = node.kind {
                if custom_refs.contains_key(&fn_id.0) {
                    continue;
                }
                if !registered_ids.contains(&fn_id.0) {
                    return Err(format!(
                        "AST references custom function id {} but no \
                         implementation was registered via \
                         JitCompiler::register_custom_function()",
                        fn_id.0
                    ));
                }
                let sym = format!("rssn_custom_fn_{}", fn_id.0);
                let fid = self
                    .module
                    .declare_function(&sym, Linkage::Import, &custom_sig)
                    .map_err(|e| {
                        format!("Failed to declare custom function {sym} import: {e:?}")
                    })?;
                let fr = self.module.declare_func_in_func(fid, func_builder.func);
                custom_refs.insert(fn_id.0, fr);
            }
        }

        let root_val = compile_ast_iterative(
            ast,
            &mut func_builder,
            vars_ptr,
            powf_func_ref,
            &custom_refs,
        )?;

        func_builder.ins().return_(&[root_val]);
        func_builder.finalize();

        let fn_name = format!("expr_{}", ast.nodes[0].dag_id.0);
        let func_id = self
            .module
            .declare_function(&fn_name, Linkage::Export, &ctx.func.signature)
            .map_err(|e| format!("Failed to declare JIT function: {e:?}"))?;

        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| format!("Failed to define JIT function: {e:?}"))?;

        self.module.clear_context(&mut ctx);

        self.module
            .finalize_definitions()
            .map_err(|e| format!("Failed to finalize JIT module: {e:?}"))?;

        let code_ptr = self.module.get_finalized_function(func_id);

        // SAFETY: Cranelift returns the address of native code matching
        // exactly the signature we declared above (`fn(*const f64) -> f64`).
        let compiled_fn: CompiledExprFn = unsafe { std::mem::transmute(code_ptr) };
        Ok(compiled_fn)
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
    custom_refs: &HashMap<u32, cranelift_codegen::ir::FuncRef>,
) -> Result<Value, String> {
    let mut stack: Vec<Frame> = Vec::with_capacity(64);
    let mut values: Vec<Value> = Vec::with_capacity(64);

    // Seed with the root.
    let root_node = ast
        .nodes
        .first()
        .ok_or_else(|| "AST projection has no root node".to_owned())?;
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
                return Err(format!("JIT codegen: AST index {} out of range", top.idx));
            };
            let Some(&child_ptr) = node.children.as_slice().get(top.cursor) else {
                return Err("JIT codegen: child cursor past child list end".to_owned());
            };
            let child_idx = child_ptr.resolve(top.idx).ok_or_else(|| {
                "Failed to resolve relative pointer in JIT codegen".to_owned()
            })?;
            top.cursor += 1;
            Action::Descend(child_idx)
        } else {
            Action::Emit(top.idx, top.arity)
        };

        match action {
            Action::Descend(child_idx) => {
                let Some(child_node) = ast.nodes.get(child_idx) else {
                    return Err(format!(
                        "JIT codegen: child AST index {child_idx} out of range"
                    ));
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
                    custom_refs,
                    &mut values,
                )?;
            }
        }
    }

    let result = values.pop().ok_or_else(|| {
        "JIT codegen value-stack ended empty; expected exactly one result".to_owned()
    })?;
    if !values.is_empty() {
        return Err(format!(
            "JIT codegen value-stack invariant violated: \
             {} leftover values after compilation",
            values.len()
        ));
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
    custom_refs: &HashMap<u32, cranelift_codegen::ir::FuncRef>,
    values: &mut Vec<Value>,
) -> Result<(), String> {
    let node = &ast.nodes[idx];

    match node.kind {
        SymbolKind::Constant => {
            let val = node.value.unwrap_or(0.0);
            values.push(builder.ins().f64const(val));
        }
        SymbolKind::Variable(sym_id) => {
            let val = emit_variable_load(builder, vars_ptr, sym_id.0);
            values.push(val);
        }
        SymbolKind::Operator(op) => {
            let split_at = values.len().checked_sub(arity).ok_or_else(|| {
                "JIT codegen value-stack underflow at operator".to_owned()
            })?;
            // Take children out in order — the iterative walker pushes
            // left-to-right, so children[0..arity] are already correct.
            let child_vals: Vec<Value> = values.drain(split_at..).collect();
            let result = emit_operator(builder, op, &child_vals, powf_func_ref, node)?;
            values.push(result);
        }
        SymbolKind::Function(fn_id) => {
            // T2.6: resolve `SymbolKind::Function(fn_id)` to the
            // FuncRef declared in `compile()` and emit a single call.
            let func_ref = custom_refs.get(&fn_id.0).copied().ok_or_else(|| {
                format!(
                    "Missing FuncRef for custom function id {} during \
                     codegen — should have been pre-declared in compile()",
                    fn_id.0
                )
            })?;
            let split_at = values.len().checked_sub(arity).ok_or_else(|| {
                "JIT codegen value-stack underflow at custom function".to_owned()
            })?;
            let child_vals: Vec<Value> = values.drain(split_at..).collect();
            if child_vals.len() != 1 {
                return Err(format!(
                    "Custom JIT functions take exactly one f64 argument; \
                     symbol got {} children",
                    child_vals.len()
                ));
            }
            let call = builder.ins().call(func_ref, &child_vals);
            let result = builder
                .inst_results(call)
                .first()
                .copied()
                .ok_or_else(|| "Custom function call returned no value".to_owned())?;
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
    // Prefetch the slot 8 cache lines ahead of `addr`. The returned
    // value is a trusted-load result we discard; the side-effect of the
    // emitted IR is what we want (`jit_review §1`).
    let _hint = emit_prefetch_hint(builder, addr);
    builder.ins().load(types::F64, MemFlags::new(), addr, 0)
}

/// Emits IR for a single algebraic operator, applying peephole identity
/// simplifications first (T2.5). The peephole runs at IR time — the
/// constant arguments here are whatever the codegen walker materialised
/// into `child_vals`, which may include `f64const` instructions we
/// emitted moments ago.
fn emit_operator(
    builder: &mut FunctionBuilder<'_>,
    op: OpKind,
    child_vals: &[Value],
    powf_func_ref: cranelift_codegen::ir::FuncRef,
    ast_node: &AstNode,
) -> Result<Value, String> {
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
                return Err("Add operator must have exactly 2 children".to_owned());
            }
            // Peephole: `x + 0 → x`, `0 + x → x`, `c1 + c2 → const`.
            match (constants[0], constants[1]) {
                (Some(l), Some(r)) => {
                    let folded = simplify_add(l, r).unwrap_or(l + r);
                    Ok(builder.ins().f64const(folded))
                }
                (Some(0.0), _) => Ok(child_vals[1]),
                (_, Some(0.0)) => Ok(child_vals[0]),
                _ => Ok(builder.ins().fadd(child_vals[0], child_vals[1])),
            }
        }
        OpKind::Sub => {
            if arity != 2 {
                return Err("Sub operator must have exactly 2 children".to_owned());
            }
            match (constants[0], constants[1]) {
                (Some(l), Some(r)) => Ok(builder.ins().f64const(l - r)),
                (_, Some(0.0)) => Ok(child_vals[0]),
                _ => Ok(builder.ins().fsub(child_vals[0], child_vals[1])),
            }
        }
        OpKind::Mul => {
            if arity != 2 {
                return Err("Mul operator must have exactly 2 children".to_owned());
            }
            // Peephole: `x * 0 → 0`, `x * 1 → x`, `c1 * c2 → const`.
            match (constants[0], constants[1]) {
                (Some(l), Some(r)) => {
                    let folded = simplify_mul(l, r).unwrap_or(l * r);
                    Ok(builder.ins().f64const(folded))
                }
                (Some(0.0), _) | (_, Some(0.0)) => Ok(builder.ins().f64const(0.0)),
                (Some(1.0), _) => Ok(child_vals[1]),
                (_, Some(1.0)) => Ok(child_vals[0]),
                _ => Ok(builder.ins().fmul(child_vals[0], child_vals[1])),
            }
        }
        OpKind::Div => {
            if arity != 2 {
                return Err("Div operator must have exactly 2 children".to_owned());
            }
            let lhs = child_vals[0];
            let rhs = child_vals[1];

            // If the divisor is a known non-zero constant, fold and skip
            // the runtime trap. If it is a known zero, force the trap at
            // codegen time so the caller gets a deterministic abort.
            if let Some(rval) = constants[1] {
                if rval == 0.0 {
                    let zero = builder.ins().f64const(0.0);
                    let is_zero = builder.ins().fcmp(FloatCC::Equal, rhs, zero);
                    builder.ins().trapnz(is_zero, TrapCode::unwrap_user(1));
                }
                if let Some(lval) = constants[0]
                    && rval != 0.0
                {
                    return Ok(builder.ins().f64const(lval / rval));
                }
            } else {
                // Mandatory runtime divide-by-zero guard (plan.md §3.1).
                let zero = builder.ins().f64const(0.0);
                let is_zero = builder.ins().fcmp(FloatCC::Equal, rhs, zero);
                builder.ins().trapnz(is_zero, TrapCode::unwrap_user(1));
            }

            Ok(builder.ins().fdiv(lhs, rhs))
        }
        OpKind::Pow => {
            if arity != 2 {
                return Err("Pow operator must have exactly 2 children".to_owned());
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
        OpKind::Neg => {
            if arity != 1 {
                return Err("Neg operator must have exactly 1 child".to_owned());
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
    fn test_jit_divide_by_zero_trap() {
        let mut builder = DagBuilder::new();
        let id = parse_expression("x / y", &mut builder).unwrap();
        let ast = dag_to_ast(builder.arena(), id);

        let mut compiler = JitCompiler::new();
        let compiled_fn = compiler.compile(&ast).unwrap();

        let safe_vars = vec![10.0, 2.0];
        let safe_res = compiled_fn(safe_vars.as_ptr());
        assert!((safe_res - 5.0).abs() < f64::EPSILON);
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
        assert!(
            err.contains("99"),
            "error must mention the unregistered fn id; got: {err}"
        );
    }
}
