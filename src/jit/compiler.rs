//! Core JIT compiler wrapping Cranelift.
//!
//! `JitCompiler` compiles stack-local AST projection trees into callable,
//! optimized native machine code. The IR generator is **iterative**
//! (work-stack + SSA-value stack) so even an expression a million nodes
//! deep does not blow the OS stack — see `jit_review §2`. It folds a
//! peephole pass over the per-node IR emission so that `x + 0`, `x * 1`,
//! `x * 0`, etc. cost zero instructions (`jit_review §1` / `§2`).
//!
//! Phase 6 additions: `OptConfig`, analysis-driven NaN-guard elision,
//! power expansion (sqrt + int pow), CSE for shared DAG nodes, and a
//! vectorized batch evaluation path (`compile_batch_f64x2`).

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::{Arc, RwLock};

use cranelift_codegen::Context;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{AbiParam, BlockArg, InstBuilder, MemFlags, Signature, Value, types};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::ast::projection::{AstNode, AstProjection};
use crate::dag::symbol::{FnId, OpKind, SymbolKind};
use crate::jit::analysis::{NodeAnalysis, PowExpansion, analyze};
use crate::jit::passes;

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

/// Column-major batch evaluation function.
///
/// - `vars_cols`: pointer to an array of column pointers, one per variable.
///   Each column pointer points to `n_rows` contiguous `f64` values.
/// - `n_rows`: number of rows to evaluate.
/// - `out`: output array of `n_rows` `f64` values.
///
/// Processes 2 rows per vector iteration via ILP (two independent SSA
/// paths), with a scalar tail for any odd final row.
pub type CompiledBatchFn = extern "C" fn(
    vars_cols: *const *const f64,
    n_rows: usize,
    out: *mut f64,
);

/// Configuration for JIT optimization passes.
#[derive(Debug, Clone)]
pub struct OptConfig {
    /// Maximum integer exponent for power expansion without `powf`.
    /// `x^n` for integer n in 2..=max_int_pow is replaced by repeated fmul.
    pub max_int_pow: u32,
    /// Expand `x^0.5` to a Cranelift `sqrt` instruction.
    pub expand_sqrt: bool,
    /// Replace `x / C` (constant non-zero denominator) with `x * (1/C)`.
    /// Not IEEE-754 bit-exact (reciprocal approximation rounding).
    pub allow_reciprocal_math: bool,
    /// Elide `select(rhs==0, NaN, lhs/rhs)` when the denominator is
    /// proven non-zero by the analysis pass.
    pub elide_nan_guard: bool,
    /// Reuse SSA values for DAG nodes that appear more than once.
    pub enable_cse: bool,
}

impl Default for OptConfig {
    fn default() -> Self {
        Self {
            max_int_pow: 8,
            expand_sqrt: true,
            allow_reciprocal_math: false,
            elide_nan_guard: true,
            enable_cse: true,
        }
    }
}

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

    /// Compiles an `AstProjection` expression into a native callable function
    /// using default optimization settings.
    ///
    /// Equivalent to `compile_with_opts(ast, &OptConfig::default())`.
    ///
    /// # Errors
    /// Returns a [`crate::error::JitError`] if compilation or linking fails.
    pub fn compile(&mut self, ast: &AstProjection) -> Result<CompiledExprFn, crate::error::JitError> {
        self.compile_with_opts(ast, &OptConfig::default())
    }

    /// Compiles an `AstProjection` expression into a native callable function
    /// with explicit optimization settings.
    ///
    /// # Errors
    /// Returns a [`crate::error::JitError`] if compilation or linking fails.
    pub fn compile_with_opts(
        &mut self,
        ast: &AstProjection,
        opts: &OptConfig,
    ) -> Result<CompiledExprFn, crate::error::JitError> {
        if ast.is_empty() {
            return crate::error::cold_jit_error_malformed_node();
        }

        // Run pre-codegen analysis pass.
        let analysis = analyze(ast);

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
            &analysis,
            opts,
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

    /// Compiles a vectorized batch evaluation function using true F64X2 SIMD.
    ///
    /// Returns `None` if the expression is not vectorizable (contains `Mod`,
    /// non-expandable `Pow`, or user `Function` nodes).
    ///
    /// The returned function operates on column-major data: each variable
    /// has its own contiguous column of `f64` values. See [`CompiledBatchFn`].
    ///
    /// The vec_body block processes 2 rows per iteration using genuine F64X2
    /// SIMD (one load/store of 16 bytes per variable column), with a scalar
    /// tail for any odd final row.
    ///
    /// # Errors
    /// Returns a [`crate::error::JitError`] if Cranelift compilation fails.
    pub fn compile_batch_f64x2(
        &mut self,
        ast: &AstProjection,
    ) -> Result<Option<CompiledBatchFn>, crate::error::JitError> {
        if ast.is_empty() {
            return crate::error::cold_jit_error_malformed_node().map(Some);
        }

        let analysis = analyze(ast);
        let opts = OptConfig::default();

        // Check vectorizability: no Mod, no non-expandable Pow, no Function.
        if !is_vectorizable_ast(ast, &analysis) {
            return Ok(None);
        }

        // Collect the set of variable sym_ids used in the expression.
        // We need the ordered sym_ids to map them to column indices.
        let sym_ids: Vec<u32> = {
            let mut seen: HashSet<u32> = HashSet::new();
            let mut ordered: Vec<u32> = Vec::new();
            for node in &ast.nodes {
                if let SymbolKind::Variable(sid) = node.kind {
                    if seen.insert(sid.0) {
                        ordered.push(sid.0);
                    }
                }
            }
            ordered
        };

        let mut ctx = Context::new();

        // Signature: fn(vars_cols: *const *const f64, n_rows: usize, out: *mut f64)
        // All three are pointer-sized (i64 on 64-bit).
        let ptr_type = self.module.target_config().pointer_type();
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // vars_cols
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // n_rows (usize = ptr_type on 64-bit)
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // out

        let mut func_builder = FunctionBuilder::new(&mut ctx.func, &mut self.builder_ctx);

        // Declare powf (needed for non-expanded pow fallback — but
        // we've already checked there are none for vectorizable ASTs).
        let mut powf_sig = Signature::new(self.module.target_config().default_call_conv);
        powf_sig.params.push(AbiParam::new(types::F64));
        powf_sig.params.push(AbiParam::new(types::F64));
        powf_sig.returns.push(AbiParam::new(types::F64));
        let powf_name = self
            .module
            .declare_function("powf", Linkage::Import, &powf_sig)
            .map_err(|_| crate::error::JitError::InitFailed)?;

        // Create basic blocks.
        let entry_block = func_builder.create_block();
        let loop_check = func_builder.create_block();
        let vec_body = func_builder.create_block();
        let scalar_check = func_builder.create_block();
        let scalar_body = func_builder.create_block();
        let ret_block = func_builder.create_block();

        // Block parameters (SSA phi-nodes for the loop induction variable).
        func_builder.append_block_params_for_function_params(entry_block);
        func_builder.append_block_param(loop_check, ptr_type);    // i
        func_builder.append_block_param(vec_body, ptr_type);       // i
        func_builder.append_block_param(scalar_check, ptr_type);   // i
        func_builder.append_block_param(scalar_body, ptr_type);    // i

        // ── entry block ────────────────────────────────────────────────────
        func_builder.switch_to_block(entry_block);
        let params = func_builder.block_params(entry_block);
        let vars_cols_val = params[0];
        let n_rows_val = params[1];
        let out_ptr_val = params[2];
        let zero_i = func_builder.ins().iconst(ptr_type, 0);
        let zero_i_ba = BlockArg::Value(zero_i);
        func_builder.ins().jump(loop_check, &[zero_i_ba]);
        func_builder.seal_block(entry_block);

        // ── loop_check(i) ──────────────────────────────────────────────────
        func_builder.switch_to_block(loop_check);
        let i_lc = func_builder.block_params(loop_check)[0];
        let remaining = func_builder.ins().isub(n_rows_val, i_lc);
        let two_i = func_builder.ins().iconst(ptr_type, 2);
        let can_vec = func_builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, remaining, two_i);
        let i_lc_ba = BlockArg::Value(i_lc);
        func_builder.ins().brif(can_vec, vec_body, &[i_lc_ba], scalar_check, &[i_lc_ba]);
        // loop_check has back-edge from vec_body — seal after vec_body is built.

        // ── vec_body(i) ─────────────────────────────────────────────────────
        // True F64X2 SIMD: loads 16 bytes (2 f64s) per variable in one
        // instruction, evaluates the full expression tree on F64X2 values,
        // and stores 16 bytes of results.
        func_builder.switch_to_block(vec_body);
        let i_vb = func_builder.block_params(vec_body)[0];

        // Byte offset of row i: i * 8
        let byte_off_vec = func_builder.ins().ishl_imm(i_vb, 3);
        let ptr_size = i64::try_from(ptr_type.bytes()).unwrap_or(8);

        // Load F64X2 values for each variable: reads f64[i] and f64[i+1] in one load.
        let mut var_vals_vec: HashMap<u32, Value> = HashMap::new();
        for &sid in &sym_ids {
            let col_offset = func_builder.ins().iconst(ptr_type,
                i64::from(sid).wrapping_mul(ptr_size));
            let col_ptr_addr = func_builder.ins().iadd(vars_cols_val, col_offset);
            let col_ptr = func_builder.ins().load(ptr_type, MemFlags::new(), col_ptr_addr, 0);
            let elem_addr = func_builder.ins().iadd(col_ptr, byte_off_vec);
            // Load 16 bytes = two consecutive f64 values = F64X2 vector
            let vec_val = func_builder.ins().load(
                types::F64X2, MemFlags::new(), elem_addr, 0);
            var_vals_vec.insert(sid, vec_val);
        }

        // The powf func_ref is not used in vectorizable expressions (all Pow nodes
        // are expanded via IntPow/Sqrt/NegIntPow), but we need a placeholder.
        let powf_func_ref_vb = self.module.declare_func_in_func(powf_name, func_builder.func);

        // Evaluate expression in F64X2 mode.
        let result_vec = emit_ast_simd_f64x2(
            ast, &analysis, &opts,
            &mut func_builder,
            &var_vals_vec,
            powf_func_ref_vb,
        )?;

        // Store F64X2 result: writes 16 bytes = two consecutive f64 outputs.
        let out_addr_vec = func_builder.ins().iadd(out_ptr_val, byte_off_vec);
        func_builder.ins().store(MemFlags::new(), result_vec, out_addr_vec, 0);

        let i_vb_next = func_builder.ins().iadd_imm(i_vb, 2);
        let i_vb_next_ba = BlockArg::Value(i_vb_next);
        func_builder.ins().jump(loop_check, &[i_vb_next_ba]);

        // Now seal loop_check (all predecessors: entry and vec_body back-edge).
        func_builder.seal_block(loop_check);
        func_builder.seal_block(vec_body);

        // ── scalar_check(i) ────────────────────────────────────────────────
        func_builder.switch_to_block(scalar_check);
        let i_sc = func_builder.block_params(scalar_check)[0];
        let done = func_builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, i_sc, n_rows_val);
        let i_sc_ba = BlockArg::Value(i_sc);
        func_builder.ins().brif(done, ret_block, &[] as &[BlockArg], scalar_body, &[i_sc_ba]);
        // scalar_check has a back-edge from scalar_body — seal after.

        // ── scalar_body(i) ────────────────────────────────────────────────
        func_builder.switch_to_block(scalar_body);
        let i_sb = func_builder.block_params(scalar_body)[0];

        let byte_off_sb = func_builder.ins().ishl_imm(i_sb, 3);
        let mut var_vals_sb: HashMap<u32, Value> = HashMap::new();
        for &sid in &sym_ids {
            let col_offset = func_builder.ins().iconst(ptr_type,
                i64::from(sid).wrapping_mul(ptr_size));
            let col_ptr_addr = func_builder.ins().iadd(vars_cols_val, col_offset);
            let col_ptr = func_builder.ins().load(ptr_type, MemFlags::new(), col_ptr_addr, 0);
            let addr = func_builder.ins().iadd(col_ptr, byte_off_sb);
            let v = func_builder.ins().load(types::F64, MemFlags::new(), addr, 0);
            var_vals_sb.insert(sid, v);
        }

        let powf_func_ref_sb = self.module.declare_func_in_func(powf_name, func_builder.func);

        let mut work_stack_sb: Vec<Frame> = Vec::with_capacity(32);
        let mut work_vals_sb: Vec<Value> = Vec::with_capacity(32);
        let res_sb = emit_ast_scalar_with_vars(
            ast, &analysis, &opts,
            &mut func_builder,
            &var_vals_sb,
            powf_func_ref_sb,
            &HashMap::new(),
            &mut work_stack_sb,
            &mut work_vals_sb,
        )?;

        let out_addr_sb = func_builder.ins().iadd(out_ptr_val, byte_off_sb);
        func_builder.ins().store(MemFlags::new(), res_sb, out_addr_sb, 0);

        let i_sb_next = func_builder.ins().iadd_imm(i_sb, 1);
        let i_sb_next_ba = BlockArg::Value(i_sb_next);
        func_builder.ins().jump(scalar_check, &[i_sb_next_ba]);

        func_builder.seal_block(scalar_check);
        func_builder.seal_block(scalar_body);

        // ── ret_block ─────────────────────────────────────────────────────
        func_builder.switch_to_block(ret_block);
        func_builder.ins().return_(&[]);
        func_builder.seal_block(ret_block);

        func_builder.finalize();

        let fn_name = format!("batch_expr_{}", ast.nodes[0].dag_id.0);
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

        // SAFETY: Cranelift returns native code matching the declared
        // `CompiledBatchFn` signature.
        let batch_fn: CompiledBatchFn = unsafe { std::mem::transmute(code_ptr) };
        Ok(Some(batch_fn))
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
    analysis: &[NodeAnalysis],
    opts: &OptConfig,
    builder: &mut FunctionBuilder<'_>,
    vars_ptr: Value,
    powf_func_ref: cranelift_codegen::ir::FuncRef,
    fmod_func_ref: cranelift_codegen::ir::FuncRef,
    custom_refs: &HashMap<u32, cranelift_codegen::ir::FuncRef>,
    stack: &mut Vec<Frame>,
    values: &mut Vec<Value>,
) -> Result<Value, crate::error::JitError> {
    // Callers clear these before passing — guaranteed by `compile_with_opts()`.

    // FMA peephole: maps each Mul-result Value to its two input factors so
    // that when an enclosing Add is emitted we can fold `a*b + c` into one
    // `fma(a, b, c)` instruction (jit_review §4 / simd_review §4).
    let mut mul_factors: HashMap<Value, (Value, Value)> = HashMap::new();

    // CSE: pre-scan for dag_ids that appear more than once.
    let duplicate_dag_ids: HashSet<u32> = if opts.enable_cse {
        let mut dag_id_count: HashMap<u32, u32> = HashMap::new();
        for node in &ast.nodes {
            *dag_id_count.entry(node.dag_id.0).or_insert(0) =
                dag_id_count.get(&node.dag_id.0).copied().unwrap_or(0).saturating_add(1);
        }
        dag_id_count.into_iter()
            .filter(|&(_, c)| c > 1)
            .map(|(id, _)| id)
            .collect()
    } else {
        HashSet::new()
    };
    // Maps dag_id.0 → already-computed SSA Value (CSE cache).
    let mut cse_map: HashMap<u32, Value> = HashMap::new();

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
        // CSE check at first visit (cursor == 0): if this node's dag_id
        // has already been computed, reuse the cached Value.
        if opts.enable_cse && top.cursor == 0 && !duplicate_dag_ids.is_empty() {
            let dag_id = ast.nodes
                .get(top.idx)
                .map(|n| n.dag_id.0)
                .unwrap_or(u32::MAX);
            if duplicate_dag_ids.contains(&dag_id) {
                if let Some(&cached) = cse_map.get(&dag_id) {
                    stack.pop();
                    values.push(cached);
                    continue;
                }
            }
        }

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
                    analysis,
                    opts,
                    idx,
                    arity,
                    builder,
                    vars_ptr,
                    powf_func_ref,
                    fmod_func_ref,
                    custom_refs,
                    values,
                    &mut mul_factors,
                )?;
                // Store result in CSE cache if this dag_id is a duplicate.
                if opts.enable_cse && !duplicate_dag_ids.is_empty() {
                    if let Some(node) = ast.nodes.get(idx) {
                        let dag_id = node.dag_id.0;
                        if duplicate_dag_ids.contains(&dag_id) {
                            if let Some(&v) = values.last() {
                                cse_map.insert(dag_id, v);
                            }
                        }
                    }
                }
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
    analysis: &[NodeAnalysis],
    opts: &OptConfig,
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
    let node = ast.nodes.get(idx).ok_or(crate::error::JitError::MalformedNode)?;

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

            // Collect child node analyses (one per child, in order).
            // For Pow nodes we also append the node's own analysis at slot 2
            // so emit_operator can read the PowExpansion strategy.
            let children = node.children.as_slice_with_pool(&ast.children_pool);
            let mut child_analyses: Vec<Option<&NodeAnalysis>> = children
                .iter()
                .map(|ptr| ptr.resolve(idx).and_then(|ci| analysis.get(ci)))
                .collect();
            // Append the node's own analysis as extra slot for Pow expansion lookup.
            child_analyses.push(analysis.get(idx));

            let result = emit_operator(
                builder, op, &child_vals,
                powf_func_ref, fmod_func_ref,
                node, &child_analyses, opts,
                mul_factors,
            )?;
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
/// simplifications, NaN-guard elision, power expansion, and FMA fusion.
///
/// `child_analyses` contains the pre-computed `NodeAnalysis` for each child
/// (one per child, in order); `opts` controls which passes are active.
#[allow(clippy::too_many_arguments)]
fn emit_operator(
    builder: &mut FunctionBuilder<'_>,
    op: OpKind,
    child_vals: &[Value],
    powf_func_ref: cranelift_codegen::ir::FuncRef,
    fmod_func_ref: cranelift_codegen::ir::FuncRef,
    ast_node: &AstNode,
    child_analyses: &[Option<&NodeAnalysis>],
    opts: &OptConfig,
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
    // Suppress "unused" warning on ast_node; it's kept for future peepholes.
    let _ = ast_node;

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
            // x - x = 0 when both sides are the same SSA value.
            if child_vals[0] == child_vals[1] {
                return Ok(builder.ins().f64const(0.0));
            }
            match (constants[0], constants[1]) {
                (Some(l), Some(r)) => Ok(builder.ins().f64const(l - r)),
                (_, Some(0.0)) => Ok(child_vals[0]),
                // 0 - x → fneg(x): cheaper than a full fsub.
                (Some(0.0), _) => Ok(builder.ins().fneg(child_vals[1])),
                _ => {
                    // FMA: (a*b) - c → fma(a, b, fneg(c))
                    if let Some(&(a, b)) = mul_factors.get(&child_vals[0]) {
                        let neg_c = builder.ins().fneg(child_vals[1]);
                        return Ok(builder.ins().fma(a, b, neg_c));
                    }
                    Ok(builder.ins().fsub(child_vals[0], child_vals[1]))
                }
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
                // x * -1 → fneg(x)
                (Some(-1.0), _) => Ok(builder.ins().fneg(child_vals[1])),
                (_, Some(-1.0)) => Ok(builder.ins().fneg(child_vals[0])),
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

            // Reciprocal math: x / C → x * (1/C) for constant non-zero C.
            if opts.allow_reciprocal_math {
                if let Some(c) = constants[1] {
                    if c != 0.0 {
                        let recip = builder.ins().f64const(1.0 / c);
                        return Ok(builder.ins().fmul(lhs, recip));
                    }
                }
            }

            // NaN guard elision: check whether the DENOMINATOR (child[1]) is
            // provably non-zero. Use child_analyses[1] directly.
            let rhs_is_const_nonzero = constants[1].map_or(false, |c| c != 0.0 && !c.is_nan());
            let rhs_nonzero = child_analyses.get(1).and_then(|a| *a)
                .map_or(false, |a| a.is_nonzero());
            let skip_guard = opts.elide_nan_guard && (rhs_nonzero || rhs_is_const_nonzero);

            if skip_guard {
                // Safe to divide directly — no NaN guard needed.
                Ok(builder.ins().fdiv(lhs, rhs))
            } else {
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
        }
        OpKind::Pow => {
            if arity != 2 {
                return Err(crate::error::JitError::MalformedNode);
            }
            // Peephole: `x ^ 0 → 1`, `x ^ 1 → x`, `c1 ^ c2 → const`.
            match (constants[0], constants[1]) {
                (Some(l), Some(r)) => return Ok(builder.ins().f64const(l.powf(r))),
                (_, Some(0.0)) => return Ok(builder.ins().f64const(1.0)),
                (_, Some(1.0)) => return Ok(child_vals[0]),
                _ => {}
            }

            // Use expansion strategy from the node's own analysis (carried via
            // child_analyses — the caller should pass the node's own analysis
            // as a stand-in if needed, but for Pow we use the node analysis
            // stored in the iteration context). For the new signature we use
            // the analysis keyed on this Pow node, which is the last element.
            // Since emit_operator no longer receives the node's own analysis
            // directly, we look it up from child_analyses slot 2 (convention:
            // callers pass [child0_an, child1_an, node_own_an] for Pow).
            let node_an: Option<&NodeAnalysis> = child_analyses.get(2).and_then(|a| *a);
            let expansion = node_an.map(|a| &a.pow_expansion);
            match expansion {
                Some(PowExpansion::Sqrt) if opts.expand_sqrt => {
                    Ok(passes::emit_sqrt(builder, child_vals[0]))
                }
                Some(PowExpansion::IntPow(n)) if *n >= 2 && *n <= opts.max_int_pow => {
                    Ok(passes::emit_int_pow(builder, child_vals[0], *n))
                }
                Some(PowExpansion::NegIntPow(n)) => {
                    let n = *n;
                    let base = child_vals[0];
                    let x_n = if n == 1 {
                        base
                    } else {
                        passes::emit_int_pow(builder, base, n)
                    };
                    let one = builder.ins().f64const(1.0);
                    // Guard: if x^n == 0, return NaN (base was zero).
                    let base_nonzero = child_analyses.get(0).and_then(|a| *a)
                        .map_or(false, |a| a.is_nonzero());
                    if opts.elide_nan_guard && base_nonzero {
                        Ok(builder.ins().fdiv(one, x_n))
                    } else {
                        let zero = builder.ins().f64const(0.0);
                        let nan_val = builder.ins().f64const(f64::NAN);
                        let is_zero = builder.ins().fcmp(FloatCC::Equal, x_n, zero);
                        let div_result = builder.ins().fdiv(one, x_n);
                        Ok(builder.ins().select(is_zero, nan_val, div_result))
                    }
                }
                _ => {
                    // Fall back to runtime powf call.
                    // Also handle constants[1] cases that reach here via
                    // the fallthrough (e.g. exp > max_int_pow).
                    if let Some(exp) = constants[1] {
                        // sqrt fallback if not handled above.
                        if opts.expand_sqrt && (exp - 0.5_f64).abs() < f64::EPSILON {
                            return Ok(passes::emit_sqrt(builder, child_vals[0]));
                        }
                        let n = exp as u32;
                        if exp == n as f64 && n >= 2 && n <= opts.max_int_pow {
                            return Ok(passes::emit_int_pow(builder, child_vals[0], n));
                        }
                    }
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
            // Check if denominator is provably nonzero to elide the guard.
            let rhs_is_const_nonzero = constants[1].map_or(false, |c| c != 0.0 && !c.is_nan());
            let rhs_nonzero = child_analyses.get(1).and_then(|a| *a)
                .map_or(false, |a| a.is_nonzero());
            if opts.elide_nan_guard && (rhs_nonzero || rhs_is_const_nonzero) {
                let call = builder.ins().call(fmod_func_ref, &[lhs, rhs]);
                Ok(builder.inst_results(call)[0])
            } else {
                let zero = builder.ins().f64const(0.0);
                let nan_val = builder.ins().f64const(f64::NAN);
                let is_zero = builder.ins().fcmp(FloatCC::Equal, rhs, zero);
                let call = builder.ins().call(fmod_func_ref, &[lhs, rhs]);
                let rem_result = builder.inst_results(call)[0];
                Ok(builder.ins().select(is_zero, nan_val, rem_result))
            }
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
}

/// Creates an F64X2 vector constant where both lanes hold the value `v`.
///
/// Inserts a 16-byte literal into the function's constant pool, then emits
/// `vconst` to load it. Bytes are in little-endian order (x86 SIMD layout).
fn f64x2_const(builder: &mut FunctionBuilder<'_>, v: f64) -> Value {
    use cranelift_codegen::ir::ConstantData;
    let bits = v.to_bits().to_le_bytes();
    let data: [u8; 16] = [
        bits[0], bits[1], bits[2], bits[3], bits[4], bits[5], bits[6], bits[7],
        bits[0], bits[1], bits[2], bits[3], bits[4], bits[5], bits[6], bits[7],
    ];
    let constant_handle = builder.func.dfg.constants.insert(ConstantData::from(&data[..]));
    builder.ins().vconst(types::F64X2, constant_handle)
}

/// Iterative post-order emitter for F64X2 SIMD code.
///
/// Evaluates the entire AST tree operating on `F64X2` values. Each variable
/// is looked up in `var_vals_vec` (which must map `sym_id.0 → F64X2 Value`).
/// Constants are splatted to both lanes via `f64x2_const`.
///
/// Returns `Err(MalformedNode)` if the AST contains anything that cannot be
/// vectorized (Function, Mod, non-expandable Pow).
fn emit_ast_simd_f64x2(
    ast: &AstProjection,
    analysis: &[NodeAnalysis],
    opts: &OptConfig,
    builder: &mut FunctionBuilder<'_>,
    var_vals_vec: &HashMap<u32, Value>,  // sym_id.0 → F64X2 Value
    _powf_func_ref: cranelift_codegen::ir::FuncRef,
) -> Result<Value, crate::error::JitError> {
    let mut stack: Vec<Frame> = Vec::with_capacity(64);
    let mut values: Vec<Value> = Vec::with_capacity(64);
    let mut mul_factors: HashMap<Value, (Value, Value)> = HashMap::new();

    let root_node = ast
        .nodes
        .first()
        .ok_or(crate::error::JitError::MalformedNode)?;
    stack.push(Frame { idx: 0, arity: root_node.children.len(), cursor: 0 });

    while let Some(top) = stack.last_mut() {
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
                let node = ast.nodes.get(idx).ok_or(crate::error::JitError::MalformedNode)?;
                match node.kind {
                    SymbolKind::Constant(_) => {
                        // Splat the constant to both lanes.
                        values.push(f64x2_const(builder, node.value));
                    }
                    SymbolKind::Variable(sym_id) => {
                        let v = var_vals_vec.get(&sym_id.0).copied()
                            .ok_or(crate::error::JitError::MalformedNode)?;
                        values.push(v);
                    }
                    SymbolKind::Operator(op) => {
                        let split_at = values.len().checked_sub(arity)
                            .ok_or(crate::error::JitError::MalformedNode)?;
                        let child_v: Vec<Value> = values.drain(split_at..).collect();

                        // Collect child analyses and append node's own analysis for Pow.
                        let children = node.children.as_slice_with_pool(&ast.children_pool);
                        let mut child_analyses: Vec<Option<&NodeAnalysis>> = children
                            .iter()
                            .map(|ptr| ptr.resolve(idx).and_then(|ci| analysis.get(ci)))
                            .collect();
                        child_analyses.push(analysis.get(idx));

                        let result = emit_operator_simd_f64x2(
                            builder, op, &child_v,
                            node, &child_analyses, opts,
                            &mut mul_factors,
                        )?;
                        values.push(result);
                    }
                    SymbolKind::Function(_) => {
                        // Functions cannot be vectorized.
                        return Err(crate::error::JitError::MalformedNode);
                    }
                }
            }
        }
    }

    values.pop().ok_or(crate::error::JitError::MalformedNode)
}

/// Emits F64X2 SIMD IR for a single algebraic operator.
///
/// Mirrors `emit_operator` but works on F64X2 values. Constant peepholes
/// use `f64x2_const` instead of `f64const`. NaN guards use `bitcast` +
/// `bitselect` instead of scalar `select`.
#[allow(clippy::too_many_arguments)]
fn emit_operator_simd_f64x2(
    builder: &mut FunctionBuilder<'_>,
    op: OpKind,
    child_vals: &[Value],
    ast_node: &AstNode,
    child_analyses: &[Option<&NodeAnalysis>],
    opts: &OptConfig,
    mul_factors: &mut HashMap<Value, (Value, Value)>,
) -> Result<Value, crate::error::JitError> {
    let _ = ast_node;
    let arity = child_vals.len();

    match op {
        OpKind::Add => {
            if arity != 2 { return Err(crate::error::JitError::MalformedNode); }
            // FMA peephole: (a*b) + c → fma(a, b, c)
            if let Some(&(a, b)) = mul_factors.get(&child_vals[0]) {
                return Ok(builder.ins().fma(a, b, child_vals[1]));
            }
            if let Some(&(a, b)) = mul_factors.get(&child_vals[1]) {
                return Ok(builder.ins().fma(a, b, child_vals[0]));
            }
            Ok(builder.ins().fadd(child_vals[0], child_vals[1]))
        }
        OpKind::Sub => {
            if arity != 2 { return Err(crate::error::JitError::MalformedNode); }
            if child_vals[0] == child_vals[1] {
                return Ok(f64x2_const(builder, 0.0));
            }
            // FMA: (a*b) - c → fma(a, b, fneg(c))
            if let Some(&(a, b)) = mul_factors.get(&child_vals[0]) {
                let neg_c = builder.ins().fneg(child_vals[1]);
                return Ok(builder.ins().fma(a, b, neg_c));
            }
            Ok(builder.ins().fsub(child_vals[0], child_vals[1]))
        }
        OpKind::Mul => {
            if arity != 2 { return Err(crate::error::JitError::MalformedNode); }
            let result = builder.ins().fmul(child_vals[0], child_vals[1]);
            // Record for FMA fusion.
            mul_factors.insert(result, (child_vals[0], child_vals[1]));
            Ok(result)
        }
        OpKind::Div => {
            if arity != 2 { return Err(crate::error::JitError::MalformedNode); }
            let lhs = child_vals[0];
            let rhs = child_vals[1];
            // Check if denominator is provably nonzero to elide NaN guard.
            let rhs_nonzero = child_analyses.get(1).and_then(|a| *a)
                .map_or(false, |a| a.is_nonzero());
            if opts.elide_nan_guard && rhs_nonzero {
                Ok(builder.ins().fdiv(lhs, rhs))
            } else {
                // Guard: vselect lanes where rhs == 0 → NaN
                let zero_vec = f64x2_const(builder, 0.0);
                let nan_vec = f64x2_const(builder, f64::NAN);
                let div_result = builder.ins().fdiv(lhs, rhs);
                // fcmp on F64X2 → boolean vector; bitcast to F64X2 for bitselect
                let is_zero_bv = builder.ins().fcmp(FloatCC::Equal, rhs, zero_vec);
                let is_zero_mask = builder.ins().bitcast(types::F64X2, MemFlags::new(), is_zero_bv);
                Ok(builder.ins().bitselect(is_zero_mask, nan_vec, div_result))
            }
        }
        OpKind::Pow => {
            if arity != 2 { return Err(crate::error::JitError::MalformedNode); }
            // Get the expansion strategy from the node's own analysis (slot 2).
            let node_an: Option<&NodeAnalysis> = child_analyses.get(2).and_then(|a| *a);
            let expansion = node_an.map(|a| &a.pow_expansion);
            match expansion {
                Some(PowExpansion::Sqrt) if opts.expand_sqrt => {
                    // sqrt is polymorphic: works on F64X2.
                    Ok(builder.ins().sqrt(child_vals[0]))
                }
                Some(PowExpansion::IntPow(n)) if *n >= 2 && *n <= opts.max_int_pow => {
                    // emit_int_pow uses fmul which is polymorphic.
                    Ok(passes::emit_int_pow(builder, child_vals[0], *n))
                }
                Some(PowExpansion::NegIntPow(n)) => {
                    let n = *n;
                    let base_vec = child_vals[0];
                    let x_n = if n == 1 {
                        base_vec
                    } else {
                        passes::emit_int_pow(builder, base_vec, n)
                    };
                    let one_vec = f64x2_const(builder, 1.0);
                    let base_nonzero = child_analyses.get(0).and_then(|a| *a)
                        .map_or(false, |a| a.is_nonzero());
                    if opts.elide_nan_guard && base_nonzero {
                        Ok(builder.ins().fdiv(one_vec, x_n))
                    } else {
                        let zero_vec = f64x2_const(builder, 0.0);
                        let nan_vec = f64x2_const(builder, f64::NAN);
                        let div_result = builder.ins().fdiv(one_vec, x_n);
                        let is_zero_bv = builder.ins().fcmp(FloatCC::Equal, x_n, zero_vec);
                        let is_zero_mask = builder.ins().bitcast(
                            types::F64X2, MemFlags::new(), is_zero_bv);
                        Ok(builder.ins().bitselect(is_zero_mask, nan_vec, div_result))
                    }
                }
                _ => {
                    // Non-expandable Pow — should not occur in vectorizable ASTs.
                    Err(crate::error::JitError::MalformedNode)
                }
            }
        }
        OpKind::Mod => {
            // Mod is excluded from vectorizable ASTs.
            Err(crate::error::JitError::MalformedNode)
        }
        OpKind::Neg => {
            if arity != 1 { return Err(crate::error::JitError::MalformedNode); }
            Ok(builder.ins().fneg(child_vals[0]))
        }
    }
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

/// Returns `true` if the expression can be compiled to a vectorized batch
/// function. Requirements: no `Mod`, no `Function`, no `Pow` nodes with a
/// `PowExpansion::None` strategy (i.e. all pow exponents must be expandable).
fn is_vectorizable_ast(ast: &AstProjection, analysis: &[NodeAnalysis]) -> bool {
    for (node, an) in ast.nodes.iter().zip(analysis.iter()) {
        match node.kind {
            SymbolKind::Function(_) => return false,
            SymbolKind::Operator(OpKind::Mod) => return false,
            SymbolKind::Operator(OpKind::Pow) => {
                if matches!(an.pow_expansion, PowExpansion::None) {
                    return false;
                }
            }
            _ => {}
        }
    }
    true
}

/// Iterative scalar emitter that substitutes variables from a pre-built map
/// (`var_vals`: sym_id.0 → SSA Value) instead of loading from a pointer.
///
/// Used by `compile_batch_f64x2` to emit two independent expression trees
/// for the two loop rows.
#[allow(clippy::too_many_arguments)]
fn emit_ast_scalar_with_vars(
    ast: &AstProjection,
    analysis: &[NodeAnalysis],
    opts: &OptConfig,
    builder: &mut FunctionBuilder<'_>,
    var_vals: &HashMap<u32, Value>,
    powf_func_ref: cranelift_codegen::ir::FuncRef,
    custom_refs: &HashMap<u32, cranelift_codegen::ir::FuncRef>,
    stack: &mut Vec<Frame>,
    values: &mut Vec<Value>,
) -> Result<Value, crate::error::JitError> {
    stack.clear();
    values.clear();

    let mut mul_factors: HashMap<Value, (Value, Value)> = HashMap::new();

    // Fmod is unused in vectorizable expressions (we already checked), but we
    // need a placeholder FuncRef. Reuse powf_func_ref as a dummy — it will
    // never be called because Mod nodes are excluded.
    let fmod_dummy = powf_func_ref;

    let root_node = ast
        .nodes
        .first()
        .ok_or(crate::error::JitError::MalformedNode)?;
    stack.push(Frame { idx: 0, arity: root_node.children.len(), cursor: 0 });

    while let Some(top) = stack.last_mut() {
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
                let node = ast.nodes.get(idx).ok_or(crate::error::JitError::MalformedNode)?;
                match node.kind {
                    SymbolKind::Constant(_) => {
                        values.push(builder.ins().f64const(node.value));
                    }
                    SymbolKind::Variable(sym_id) => {
                        let v = var_vals.get(&sym_id.0).copied()
                            .ok_or(crate::error::JitError::MalformedNode)?;
                        values.push(v);
                    }
                    SymbolKind::Operator(op) => {
                        let split_at = values.len().checked_sub(arity)
                            .ok_or(crate::error::JitError::MalformedNode)?;
                        let child_v: Vec<Value> = values.drain(split_at..).collect();
                        let children = node.children.as_slice_with_pool(&ast.children_pool);
                        let mut child_analyses: Vec<Option<&NodeAnalysis>> = children
                            .iter()
                            .map(|ptr| ptr.resolve(idx).and_then(|ci| analysis.get(ci)))
                            .collect();
                        // Append node's own analysis for Pow expansion slot.
                        child_analyses.push(analysis.get(idx));
                        let result = emit_operator(
                            builder, op, &child_v,
                            powf_func_ref, fmod_dummy,
                            node, &child_analyses, opts,
                            &mut mul_factors,
                        )?;
                        values.push(result);
                    }
                    SymbolKind::Function(fn_id) => {
                        let func_ref = custom_refs.get(&fn_id.0).copied()
                            .ok_or(crate::error::JitError::UnknownFunction)?;
                        let split_at = values.len().checked_sub(arity)
                            .ok_or(crate::error::JitError::MalformedNode)?;
                        let child_v: Vec<Value> = values.drain(split_at..).collect();
                        if child_v.is_empty() || child_v.len() > 3 {
                            return Err(crate::error::JitError::MalformedNode);
                        }
                        let call = builder.ins().call(func_ref, &child_v);
                        let result = builder
                            .inst_results(call)
                            .first()
                            .copied()
                            .ok_or(crate::error::JitError::VerifierRejected)?;
                        values.push(result);
                    }
                }
            }
        }
    }

    let result = values.pop().ok_or(crate::error::JitError::MalformedNode)?;
    Ok(result)
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

    #[test]
    fn test_power_expansion_x_squared_no_powf() {
        // x^2 should compile without calling powf: result must be exact.
        let mut b = DagBuilder::new();
        let id = parse_expression("x ^ 2", &mut b).unwrap();
        let ast = dag_to_ast(b.arena(), id);
        let mut compiler = JitCompiler::new();
        let f = compiler.compile(&ast).unwrap();
        // x=3: 3^2 = 9
        let result = f([3.0_f64].as_ptr());
        assert!((result - 9.0).abs() < f64::EPSILON, "3^2 should be 9; got {result}");
    }

    #[test]
    fn test_power_expansion_sqrt() {
        let mut b = DagBuilder::new();
        let id = parse_expression("x ^ 0.5", &mut b).unwrap();
        let ast = dag_to_ast(b.arena(), id);
        let mut compiler = JitCompiler::new();
        let f = compiler.compile(&ast).unwrap();
        let result = f([4.0_f64].as_ptr());
        assert!((result - 2.0).abs() < 1e-10, "4^0.5 should be 2; got {result}");
    }

    #[test]
    fn test_nan_guard_elision_constant_denominator() {
        // x / 3.0: denominator is a nonzero constant, no NaN guard needed.
        let mut b = DagBuilder::new();
        let id = parse_expression("x / 3", &mut b).unwrap();
        let ast = dag_to_ast(b.arena(), id);
        let mut compiler = JitCompiler::new();
        let f = compiler.compile(&ast).unwrap();
        let result = f([9.0_f64].as_ptr());
        assert!((result - 3.0).abs() < f64::EPSILON, "9/3 should be 3; got {result}");
    }

    #[test]
    fn test_batch_f64x2_correctness() {
        let mut b = DagBuilder::new();
        let id = parse_expression("x + y", &mut b).unwrap();
        let ast = dag_to_ast(b.arena(), id);
        let mut compiler = JitCompiler::new();
        let batch_fn = compiler
            .compile_batch_f64x2(&ast)
            .expect("compile ok")
            .expect("should be vectorizable");

        // 4 rows: [1+2=3, 3+4=7, 5+6=11, 7+8=15]
        let x_col = vec![1.0_f64, 3.0, 5.0, 7.0];
        let y_col = vec![2.0_f64, 4.0, 6.0, 8.0];
        let cols: Vec<*const f64> = vec![x_col.as_ptr(), y_col.as_ptr()];
        let mut out = vec![0.0_f64; 4];
        batch_fn(cols.as_ptr(), 4, out.as_mut_ptr());

        let expected = [3.0_f64, 7.0, 11.0, 15.0];
        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < f64::EPSILON, "row {i}: expected {exp}, got {got}");
        }
    }

    #[test]
    fn test_neg_int_pow_x_inv() {
        // x^(-1) = 1/x. For x=3, result should be 1/3.
        let mut b = DagBuilder::new();
        let id = parse_expression("x ^ -1", &mut b).unwrap();
        let ast = dag_to_ast(b.arena(), id);
        let mut compiler = JitCompiler::new();
        let f = compiler.compile(&ast).unwrap();
        let result = f([3.0_f64].as_ptr());
        let expected = 1.0_f64 / 3.0;
        assert!((result - expected).abs() < 1e-14,
            "x^(-1) for x=3 should be ~{expected}; got {result}");
        // x=0 should return NaN (not trap).
        let nan_result = f([0.0_f64].as_ptr());
        assert!(nan_result.is_nan(), "x^(-1) for x=0 should be NaN; got {nan_result}");
    }

    #[test]
    fn test_analysis_x_squared_plus_1_is_positive() {
        // x^2 + 1 should be proven is_positive by the analysis.
        let mut b = DagBuilder::new();
        let id = parse_expression("x ^ 2 + 1", &mut b).unwrap();
        let ast = dag_to_ast(b.arena(), id);
        let analysis = crate::jit::analysis::analyze(&ast);
        // Root is the Add node (index 0).
        let root_an = &analysis[0];
        assert!(root_an.is_positive,
            "x^2 + 1 should be provably positive; got {root_an:?}");
        assert!(root_an.is_nonnegative,
            "x^2 + 1 should be provably nonneg; got {root_an:?}");
    }

    #[test]
    fn test_nan_guard_elision_x_sq_plus_1_denominator() {
        // x / (x^2 + 1): denominator is proven positive, so no NaN guard needed.
        // For any x, x^2 + 1 ≥ 1 > 0, so this is always safe.
        let mut b = DagBuilder::new();
        let id = parse_expression("x / (x ^ 2 + 1)", &mut b).unwrap();
        let ast = dag_to_ast(b.arena(), id);
        let mut compiler = JitCompiler::new();
        let f = compiler.compile(&ast).unwrap();

        // Test for various x values including x=0.
        let test_cases: &[(f64, f64)] = &[
            (0.0, 0.0),      // 0 / 1 = 0
            (1.0, 0.5),      // 1 / 2 = 0.5
            (2.0, 0.4),      // 2 / 5 = 0.4
            (-1.0, -0.5),    // -1 / 2 = -0.5
            (3.0, 3.0 / 10.0),  // 3 / 10
        ];
        for &(x, expected) in test_cases {
            let result = f([x].as_ptr());
            assert!((result - expected).abs() < 1e-14,
                "x / (x^2+1) for x={x}: expected {expected}, got {result}");
        }
    }

    #[test]
    fn test_batch_f64x2_true_simd() {
        // Verify that the batch F64X2 function (now true SIMD) produces
        // correct results, including for odd numbers of rows (scalar tail).
        let mut b = DagBuilder::new();
        let id = parse_expression("x * x + 1", &mut b).unwrap();
        let ast = dag_to_ast(b.arena(), id);
        let mut compiler = JitCompiler::new();
        let batch_fn = compiler
            .compile_batch_f64x2(&ast)
            .expect("compile ok")
            .expect("should be vectorizable");

        // 5 rows: test both the SIMD path (4 rows) and scalar tail (1 row).
        let x_col = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let cols: Vec<*const f64> = vec![x_col.as_ptr()];
        let mut out = vec![0.0_f64; 5];
        batch_fn(cols.as_ptr(), 5, out.as_mut_ptr());

        let expected = [1.0_f64, 2.0, 5.0, 10.0, 17.0]; // x^2 + 1
        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < f64::EPSILON, "row {i}: expected {exp}, got {got}");
        }
    }

    #[test]
    fn test_zero_minus_x_is_fneg() {
        // 0 - x should compile to fneg(x). For x=3, result is -3.
        let mut b = DagBuilder::new();
        let id = parse_expression("0 - x", &mut b).unwrap();
        let ast = dag_to_ast(b.arena(), id);
        let mut compiler = JitCompiler::new();
        let f = compiler.compile(&ast).unwrap();
        let result = f([3.0_f64].as_ptr());
        assert!((result - (-3.0)).abs() < f64::EPSILON,
            "0 - 3 should be -3; got {result}");
        let result2 = f([0.0_f64].as_ptr());
        assert!(result2 == 0.0 || result2 == -0.0,
            "0 - 0 should be ±0; got {result2}");
    }
}

