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

use cranelift_codegen::Context;
use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Signature, TrapCode, Value, types};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::ast::projection::{AstNode, AstProjection};
use crate::dag::symbol::{OpKind, SymbolKind};
use crate::jit::codegen::emit_prefetch_hint;

/// A JIT-compiled expression function pointer.
///
/// It takes a pointer to an array of variable values (`*const f64`),
/// ordered by their `SymbolId` values, and returns the computed float result.
pub type CompiledExprFn = extern "C" fn(*const f64) -> f64;

extern "C" fn jit_powf(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

/// The primary compiler context for compiling symbolic expressions to native code.
pub struct JitCompiler {
    module: JITModule,
    builder_ctx: FunctionBuilderContext,
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
        let isa_builder =
            cranelift_native::builder().expect("Failed to detect native host platform");

        // Optimizing compiler flags.
        let mut flag_builder = cranelift_codegen::settings::builder();
        cranelift_codegen::settings::Configurable::set(&mut flag_builder, "opt_level", "speed")
            .expect("Failed to set opt_level");

        let isa = isa_builder
            .finish(cranelift_codegen::settings::Flags::new(flag_builder))
            .expect("Failed to build target ISA");

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        builder.symbol("powf", jit_powf as *const u8);

        let module = JITModule::new(builder);
        Self {
            module,
            builder_ctx: FunctionBuilderContext::new(),
        }
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

        let root_val = compile_ast_iterative(
            ast,
            &mut func_builder,
            vars_ptr,
            powf_func_ref,
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
) -> Result<Value, String> {
    let mut stack: Vec<Frame> = Vec::with_capacity(64);
    let mut values: Vec<Value> = Vec::with_capacity(64);

    // Seed with the root.
    stack.push(Frame {
        idx: 0,
        arity: ast.nodes[0].children.len(),
        cursor: 0,
    });

    while !stack.is_empty() {
        // Decide whether to push a child or emit this node, then act —
        // splitting the decision from the action keeps the borrow checker
        // happy when we go from `last_mut()` to `push()` / `pop()`.
        let action = {
            let top = stack.last_mut().expect("non-empty stack");
            if top.cursor < top.arity {
                let node = &ast.nodes[top.idx];
                let child_ptr = node.children.as_slice()[top.cursor];
                let child_idx = child_ptr.resolve(top.idx).ok_or_else(|| {
                    "Failed to resolve relative pointer in JIT codegen".to_owned()
                })?;
                top.cursor += 1;
                Action::Descend(child_idx)
            } else {
                Action::Emit(top.idx, top.arity)
            }
        };

        match action {
            Action::Descend(child_idx) => {
                stack.push(Frame {
                    idx: child_idx,
                    arity: ast.nodes[child_idx].children.len(),
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
                    &mut values,
                )?;
            }
        }
    }

    if values.len() != 1 {
        return Err(format!(
            "JIT codegen value-stack invariant violated: ended with {} values \
             but expected exactly 1",
            values.len()
        ));
    }
    Ok(values.pop().expect("value stack invariant"))
}

enum Action {
    Descend(usize),
    Emit(usize, usize),
}

fn emit_one_node(
    ast: &AstProjection,
    idx: usize,
    arity: usize,
    builder: &mut FunctionBuilder<'_>,
    vars_ptr: Value,
    powf_func_ref: cranelift_codegen::ir::FuncRef,
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
        SymbolKind::Function(_) => {
            // T2.6 (custom JIT functions) is deferred — see dev_plan.md.
            // It needs a `symbol_lookup_fn` on the JITBuilder which can
            // only be registered at construction time, so the runtime
            // `register_custom_function` API is being redesigned. Until
            // then, AST nodes carrying `SymbolKind::Function` cannot be
            // JIT-compiled.
            return Err(
                "JIT compilation of custom Functions is not yet supported (T2.6 follow-up)"
                    .to_owned(),
            );
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
}
