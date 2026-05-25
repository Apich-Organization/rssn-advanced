//! Unified custom-operator extension system.
//!
//! This module is the single entry point for defining operators that plug into
//! all three RSSN-Advanced pipelines simultaneously:
//!
//! | Pipeline | Integration method |
//! |---|---|
//! | **JIT compilation** | [`CustomOpRegistry::apply_to_jit`] — registers `eval_fn` pointers; enables `compile_batch_f64x2` for pure operators |
//! | **Heuristic simplifier** | [`CustomOpRegistry::build_rule_registry`] — produces a [`RuleRegistry`] from attached [`SimplifyRule`]s |
//! | **E-graph saturation** | [`CustomOpRegistry::apply_to_egraph`] — injects [`EGraphRule`]s into an [`EGraph`][crate::egraph::egraph::EGraph] |
//!
//! ## Previous fragmented API vs. this module
//!
//! Before this module, extending the pipeline required three independent
//! registration calls that could easily fall out of sync:
//!
//! ```text
//! // OLD — three disconnected registrations:
//! compiler.register_custom_function(fn_id, fn_ptr);          // JIT
//! rule_registry.register_named("name", closure, priority, …); // simplifier
//! egraph.add_rule(closure);                                   // e-graph
//! ```
//!
//! With `CustomOpRegistry` the same operator is described once:
//!
//! ```rust,ignore
//! extern "C" fn my_sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
//!
//! let desc = CustomOpDescriptor::builder(FnId(10), "sigmoid", EvalFn::Arity1(my_sigmoid))
//!     .vectorizable()
//!     .simplify_rule("sigmoid identity", 5, |_b, _k, _c| None)
//!     .build();
//!
//! let mut reg = CustomOpRegistry::new();
//! reg.register(desc)?;
//! let reg = Arc::new(reg);
//!
//! reg.apply_to_jit(&mut compiler);             // JIT + batch vectorisation
//! let rule_reg = reg.build_rule_registry();    // heuristic simplifier
//! reg.apply_to_egraph(&mut egraph);            // e-graph rules
//! ```
//!
//! ## C FFI
//!
//! C/C++ callers use the `rssn_custom_op_*` family of functions in
//! [`crate::ffi::c_api`]:
//!
//! ```c
//! RssnCustomOpRegistry* reg = rssn_custom_op_registry_new();
//! rssn_custom_op_register_fn1(reg, 10, "sigmoid", my_sigmoid_c, /*vectorizable=*/1);
//! rssn_custom_op_add_simplify_rule(reg, 10, "no-op rule", 5, my_rule_cb, NULL);
//!
//! // Use registry in each pipeline step:
//! rssn_dag_compile_with_custom_ops(builder, root, reg, &fn_ptr);
//! rssn_dag_simplify_with_custom_ops(builder, root, reg, &out_id);
//! rssn_dag_egraph_with_custom_ops(builder, root, config, reg, &out_id);
//!
//! rssn_custom_op_registry_free(reg);
//! ```

pub mod descriptor;

pub use descriptor::{
    CustomOpDescriptor, CustomOpDescriptorBuilder, CustomOpError, CustomOpRegistry, EGraphRule,
    EGraphRuleArc, EvalFn, EvalFn1, EvalFn2, EvalFn3, SimplifyRule, SimplifyRuleArc,
};
