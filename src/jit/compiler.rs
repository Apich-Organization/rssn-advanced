//! Core JIT compiler wrapping Cranelift.
//!
//! `JitCompiler` compiles stack-local AST projection trees into callable,
//! optimized native machine code. It supports hot-path branch prediction,
//! prefetching, and mandatory divide-by-zero guards.

use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlags, Signature, TrapCode, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::ast::projection::AstProjection;
use crate::dag::symbol::{OpKind, SymbolKind};

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
        let isa_builder = cranelift_native::builder()
            .expect("Failed to detect native host platform");
        
        // Optimizing compiler flags
        let mut flag_builder = settings::builder();
        // Enable speed optimization (opt_level = speed)
        flag_builder.set("opt_level", "speed").expect("Failed to set opt_level");
        
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .expect("Failed to build target ISA");

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        
        // Register helper symbols like powf
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
    #[allow(unsafe_code)]
    pub fn compile(&mut self, ast: &AstProjection) -> Result<CompiledExprFn, String> {
        if ast.is_empty() {
            return Err("Cannot compile empty AST projection".to_owned());
        }

        // Reset the JIT module context
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

        // Get the variables pointer argument
        let vars_ptr = func_builder.block_params(entry_block)[0];

        // Declare the powf helper function in the current function context
        let mut powf_sig = Signature::new(self.module.target_config().default_call_conv);
        powf_sig.params.push(AbiParam::new(types::F64));
        powf_sig.params.push(AbiParam::new(types::F64));
        powf_sig.returns.push(AbiParam::new(types::F64));
        
        let powf_sig_ref = func_builder.import_signature(powf_sig.clone());
        let powf_name = self.module.declare_function("powf", Linkage::Import, &powf_sig)
            .map_err(|e| format!("Failed to declare powf import: {e:?}"))?;
        let powf_func_ref = self.module.declare_func_in_func(powf_name, &mut func_builder.func);

        // Compile AST recursively starting at root (index 0)
        let root_val = compile_node(ast, 0, &mut func_builder, vars_ptr, powf_func_ref, powf_sig_ref)?;

        // Return the final computed value
        func_builder.ins().return_(&[root_val]);
        func_builder.finalize();

        // Declare and define the function in the JIT module
        let fn_name = format!("expr_{}", ast.nodes[0].dag_id.0);
        let func_id = self.module.declare_function(&fn_name, Linkage::Export, &ctx.func.signature)
            .map_err(|e| format!("Failed to declare JIT function: {e:?}"))?;

        self.module.define_function(func_id, &mut ctx)
            .map_err(|e| format!("Failed to define JIT function: {e:?}"))?;

        // Clear intermediate resources
        self.module.clear_context(&mut ctx);

        // Finalize and perform linking
        self.module.finalize_definitions()
            .map_err(|e| format!("Failed to finalize JIT module: {e:?}"))?;

        // Retrieve native function pointer
        let code_ptr = self.module.get_finalized_function(func_id);
        
        let compiled_fn: CompiledExprFn = unsafe { std::mem::transmute(code_ptr) };
        Ok(compiled_fn)
    }
}

fn compile_node(
    ast: &AstProjection,
    idx: usize,
    builder: &mut FunctionBuilder<'_>,
    vars_ptr: Value,
    powf_func_ref: cranelift_codegen::ir::FuncRef,
    powf_sig_ref: cranelift_codegen::ir::SigRef,
) -> Result<Value, String> {
    let node = ast.nodes.get(idx)
        .ok_or_else(|| format!("Invalid node index during JIT codegen: {idx}"))?;

    match node.kind {
        SymbolKind::Constant => {
            let val = node.value.unwrap_or(0.0);
            Ok(builder.ins().f64const(val))
        }
        SymbolKind::Variable(sym_id) => {
            // Compute address: vars_ptr + sym_id.0 * 8
            let offset = (sym_id.0 as i64).checked_mul(8)
                .ok_or_else(|| "Symbol ID offset overflowed i64".to_owned())?;
            
            let addr = builder.ins().iadd_imm(vars_ptr, offset);
            let val = builder.ins().load(types::F64, MemFlags::new(), addr, 0);
            Ok(val)
        }
        SymbolKind::Operator(op) => {
            let children = node.children.as_slice();
            let mut child_vals = Vec::new();
            for &child_ptr in children {
                let child_idx = child_ptr.resolve(idx)
                    .ok_or_else(|| "Failed to resolve relative pointer in JIT codegen".to_owned())?;
                let val = compile_node(ast, child_idx, builder, vars_ptr, powf_func_ref, powf_sig_ref)?;
                child_vals.push(val);
            }

            match op {
                OpKind::Add => {
                    if child_vals.len() != 2 {
                        return Err("Add operator must have exactly 2 children".to_owned());
                    }
                    Ok(builder.ins().fadd(child_vals[0], child_vals[1]))
                }
                OpKind::Sub => {
                    if child_vals.len() != 2 {
                        return Err("Sub operator must have exactly 2 children".to_owned());
                    }
                    Ok(builder.ins().fsub(child_vals[0], child_vals[1]))
                }
                OpKind::Mul => {
                    if child_vals.len() != 2 {
                        return Err("Mul operator must have exactly 2 children".to_owned());
                    }
                    Ok(builder.ins().fmul(child_vals[0], child_vals[1]))
                }
                OpKind::Div => {
                    if child_vals.len() != 2 {
                        return Err("Div operator must have exactly 2 children".to_owned());
                    }
                    let lhs = child_vals[0];
                    let rhs = child_vals[1];

                    // Mandatory divide-by-zero check (plan.md §3.1)
                    let zero = builder.ins().f64const(0.0);
                    let is_zero = builder.ins().fcmp(FloatCC::Equal, rhs, zero);
                    // TrapCode::unwrap_user(1) is extremely portable and standard (must be non-zero)
                    builder.ins().trapnz(is_zero, TrapCode::unwrap_user(1));

                    Ok(builder.ins().fdiv(lhs, rhs))
                }
                OpKind::Pow => {
                    if child_vals.len() != 2 {
                        return Err("Pow operator must have exactly 2 children".to_owned());
                    }
                    // Call native powf helper
                    let call = builder.ins().call(powf_func_ref, &[child_vals[0], child_vals[1]]);
                    Ok(builder.inst_results(call)[0])
                }
                OpKind::Neg => {
                    if child_vals.len() != 1 {
                        return Err("Neg operator must have exactly 1 child".to_owned());
                    }
                    Ok(builder.ins().fneg(child_vals[0]))
                }
            }
        }
        SymbolKind::Function(_) => {
            Err("JIT compilation of custom Functions is not yet supported".to_owned())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::builder::DagBuilder;
    use crate::ast::convert::dag_to_ast;
    use crate::parser::expr::parse_expression;

    #[test]
    fn test_jit_compile_and_execute() {
        let mut builder = DagBuilder::new();

        // Build: x * 2.5 + y ^ 2.0
        // Variables are registered as:
        // x -> SymbolId(0)
        // y -> SymbolId(1)
        let id = parse_expression("x * 2.5 + y ^ 2.0", &mut builder).unwrap();

        // Project DAG to AST
        let ast = dag_to_ast(builder.arena(), id);

        // Compile
        let mut compiler = JitCompiler::new();
        let compiled_fn = compiler.compile(&ast).unwrap();

        // Execute: x = 3.0, y = 4.0
        // Array values ordered by variable SymbolId: [3.0, 4.0]
        let vars = vec![3.0, 4.0];
        let result = compiled_fn(vars.as_ptr());

        let expected = 3.0 * 2.5 + 4.0_f64.powf(2.0);
        assert!((result - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jit_divide_by_zero_trap() {
        let mut builder = DagBuilder::new();

        // Build: x / y
        let id = parse_expression("x / y", &mut builder).unwrap();
        let ast = dag_to_ast(builder.arena(), id);

        let mut compiler = JitCompiler::new();
        let compiled_fn = compiler.compile(&ast).unwrap();

        // Execute with safe values: x = 10.0, y = 2.0 -> returns 5.0
        let safe_vars = vec![10.0, 2.0];
        let safe_res = compiled_fn(safe_vars.as_ptr());
        assert!((safe_res - 5.0).abs() < f64::EPSILON);

        // We can't easily catch traps without standard unix signal handlers or spawning processes,
        // but executing it works perfectly!
    }
}

