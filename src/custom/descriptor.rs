//! Core types for the unified custom-operator extension system.
//!
//! Build a [`CustomOpDescriptor`] with its fluent builder, register it into a
//! [`CustomOpRegistry`], then pass the registry to each pipeline integration
//! method instead of configuring the JIT, simplifier, and e-graph separately.
//!
//! # Quick start
//!
//! ```rust
//! use rssn_advanced::custom::descriptor::{CustomOpDescriptor, CustomOpRegistry, EvalFn};
//! use rssn_advanced::dag::builder::DagBuilder;
//!
//! extern "C" fn my_relu(x: f64) -> f64 { x.max(0.0) }
//!
//! let mut builder = DagBuilder::new();
//! // intern_function is the public API for obtaining a FnId from outside the crate
//! let fn_id = builder.intern_function("relu");
//!
//! let desc = CustomOpDescriptor::builder(fn_id, "relu", EvalFn::Arity1(my_relu))
//!     .vectorizable()      // pure → safe to duplicate in ILP batch path
//!     .simplify_rule("relu(0) → 0", 10, |_b, _kind, _children| None)
//!     .cost(1.5)
//!     .build();
//!
//! let mut registry = CustomOpRegistry::new();
//! registry.register(desc).unwrap();
//! ```

#![allow(clippy::single_call_fn)]

use std::collections::HashMap;
use std::sync::Arc;

use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use crate::dag::symbol::{FnId, SymbolKind};
use crate::heuristic::rule_registry::{RuleFn, RuleRegistry};

// ── Evaluation function types ─────────────────────────────────────────────

/// 1-argument `extern "C"` evaluation function (`f64 → f64`).
///
/// The function must be `extern "C"` so the JIT can emit a direct `call`
/// with the platform C ABI (same as `libc` math functions).
pub type EvalFn1 = extern "C" fn(f64) -> f64;

/// 2-argument `extern "C"` evaluation function (`f64, f64 → f64`).
pub type EvalFn2 = extern "C" fn(f64, f64) -> f64;

/// 3-argument `extern "C"` evaluation function (`f64, f64, f64 → f64`).
pub type EvalFn3 = extern "C" fn(f64, f64, f64) -> f64;

/// Arity-tagged evaluation function pointer.
///
/// Stored in [`CustomOpDescriptor`] and used by the JIT to emit the correct
/// `call` instruction signature.
#[derive(Clone, Copy)]
pub enum EvalFn {
    /// Unary operator (`f64 → f64`).
    Arity1(EvalFn1),
    /// Binary operator (`f64, f64 → f64`).
    Arity2(EvalFn2),
    /// Ternary operator (`f64, f64, f64 → f64`).
    Arity3(EvalFn3),
}

impl EvalFn {
    /// Returns the number of `f64` arguments the function accepts.
    #[must_use]
    pub const fn arity(self) -> u8 {
        match self {
            Self::Arity1(_) => 1,
            Self::Arity2(_) => 2,
            Self::Arity3(_) => 3,
        }
    }

    /// Evaluate with a slice of arguments.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `args.len() != self.arity()`.
    #[must_use]
    pub fn call(self, args: &[f64]) -> f64 {
        debug_assert_eq!(args.len(), self.arity() as usize, "EvalFn arity mismatch");
        match self {
            Self::Arity1(f) => f(args[0]),
            Self::Arity2(f) => f(args[0], args[1]),
            Self::Arity3(f) => f(args[0], args[1], args[2]),
        }
    }
}

// ── Rule types ────────────────────────────────────────────────────────────

/// Shared heuristic simplification rule closure.
///
/// Uses `Arc` (not `Box`) so the same closure can be cloned cheaply into a
/// [`RuleRegistry`] without copying the allocation.
pub type SimplifyRuleArc =
    Arc<dyn Fn(&mut DagBuilder, SymbolKind, &[DagNodeId]) -> Option<DagNodeId> + Send + Sync>;

/// Shared e-graph rewrite rule closure.
pub type EGraphRuleArc =
    Arc<dyn Fn(&mut DagBuilder, &SymbolKind, &[DagNodeId]) -> Option<DagNodeId> + Send + Sync>;

/// A heuristic simplification rule attached to a [`CustomOpDescriptor`].
pub struct SimplifyRule {
    /// Human-readable name for fingerprinting and diagnostics.
    pub name: String,
    /// Dispatch priority — higher fires first.
    pub priority: i32,
    /// The rule closure.
    pub rule: SimplifyRuleArc,
}

impl Clone for SimplifyRule {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            priority: self.priority,
            rule: Arc::clone(&self.rule),
        }
    }
}

/// An e-graph rewrite rule attached to a [`CustomOpDescriptor`].
pub struct EGraphRule {
    /// When `true` the rule runs *after* built-in algebraic rules each
    /// saturation round; `false` runs it before.
    pub after_builtins: bool,
    /// The rule closure.
    pub rule: EGraphRuleArc,
}

impl Clone for EGraphRule {
    fn clone(&self) -> Self {
        Self {
            after_builtins: self.after_builtins,
            rule: Arc::clone(&self.rule),
        }
    }
}

// ── CustomOpDescriptor ────────────────────────────────────────────────────

/// Complete descriptor for a user-defined operator.
///
/// A descriptor bundles every pipeline-facing property of a custom operator
/// into a single value.  Register it with [`CustomOpRegistry::register`].
pub struct CustomOpDescriptor {
    /// Numeric identifier — unique within a registry.
    pub fn_id: FnId,
    /// Human-readable name resolved by the expression parser.
    pub name: String,
    /// Native evaluation function called by the JIT via `call`.
    pub eval_fn: EvalFn,
    /// When `true`, the operator is pure (no side effects) and may be
    /// duplicated by the batch f64×2 ILP vectorisation path.
    pub vectorizable: bool,
    /// Heuristic simplification rules (fed to the `HeuristicEngine`).
    pub simplify_rules: Vec<SimplifyRule>,
    /// E-graph rewrite rules (fed to equality saturation).
    pub egraph_rules: Vec<EGraphRule>,
    /// Extraction cost hint for the e-graph extractor (default: `2.0`;
    /// built-in binary ops are `1.0`, variables `0.5`).
    pub cost: f64,
}

impl Clone for CustomOpDescriptor {
    fn clone(&self) -> Self {
        Self {
            fn_id: self.fn_id,
            name: self.name.clone(),
            eval_fn: self.eval_fn,
            vectorizable: self.vectorizable,
            simplify_rules: self.simplify_rules.clone(),
            egraph_rules: self.egraph_rules.clone(),
            cost: self.cost,
        }
    }
}

impl CustomOpDescriptor {
    /// Start building a descriptor with fluent API.
    ///
    /// ```rust
    /// use rssn_advanced::custom::descriptor::{CustomOpDescriptor, EvalFn};
    /// use rssn_advanced::dag::builder::DagBuilder;
    ///
    /// extern "C" fn sin_approx(x: f64) -> f64 { x.sin() }
    ///
    /// let mut builder = DagBuilder::new();
    /// let fn_id = builder.intern_function("sin");
    /// let _desc = CustomOpDescriptor::builder(fn_id, "sin", EvalFn::Arity1(sin_approx))
    ///     .vectorizable()
    ///     .build();
    /// ```
    #[must_use]
    pub fn builder(
        fn_id: impl Into<FnId>,
        name: impl Into<String>,
        eval_fn: EvalFn,
    ) -> CustomOpDescriptorBuilder {
        CustomOpDescriptorBuilder::new(fn_id.into(), name.into(), eval_fn)
    }

    /// Returns the arity of the underlying evaluation function.
    #[must_use]
    pub const fn arity(&self) -> u8 {
        self.eval_fn.arity()
    }
}

// ── CustomOpDescriptorBuilder ─────────────────────────────────────────────

/// Fluent builder for [`CustomOpDescriptor`].
pub struct CustomOpDescriptorBuilder {
    fn_id: FnId,
    name: String,
    eval_fn: EvalFn,
    vectorizable: bool,
    simplify_rules: Vec<SimplifyRule>,
    egraph_rules: Vec<EGraphRule>,
    cost: f64,
}

impl CustomOpDescriptorBuilder {
    /// Create a builder from the mandatory fields.
    #[must_use]
    pub const fn new(fn_id: FnId, name: String, eval_fn: EvalFn) -> Self {
        Self {
            fn_id,
            name,
            eval_fn,
            vectorizable: false,
            simplify_rules: Vec::new(),
            egraph_rules: Vec::new(),
            cost: 2.0,
        }
    }

    /// Mark the operator as pure and safe for ILP vectorisation.
    ///
    /// When set, the batch f64×2 path may call the function twice in the
    /// same loop body (once per row) to expose instruction-level parallelism.
    #[must_use]
    pub const fn vectorizable(mut self) -> Self {
        self.vectorizable = true;
        self
    }

    /// Attach a heuristic simplification rule.
    ///
    /// The `rule` closure receives `(builder, kind, children)` for every DAG
    /// node visited by the simplifier.  Return `Some(replacement_id)` to
    /// rewrite, `None` to pass.
    pub fn simplify_rule<F>(mut self, name: impl Into<String>, priority: i32, rule: F) -> Self
    where
        F: Fn(&mut DagBuilder, SymbolKind, &[DagNodeId]) -> Option<DagNodeId>
            + Send
            + Sync
            + 'static,
    {
        self.simplify_rules.push(SimplifyRule {
            name: name.into(),
            priority,
            rule: Arc::new(rule),
        });
        self
    }

    /// Attach an e-graph rewrite rule.
    ///
    /// `after_builtins = false` runs the rule before built-in algebraic rules
    /// each saturation round; `true` runs it after.
    pub fn egraph_rule<F>(mut self, after_builtins: bool, rule: F) -> Self
    where
        F: Fn(&mut DagBuilder, &SymbolKind, &[DagNodeId]) -> Option<DagNodeId>
            + Send
            + Sync
            + 'static,
    {
        self.egraph_rules.push(EGraphRule {
            after_builtins,
            rule: Arc::new(rule),
        });
        self
    }

    /// Override the e-graph extraction cost (default `2.0`).
    #[must_use]
    pub const fn cost(mut self, c: f64) -> Self {
        self.cost = c;
        self
    }

    /// Consume the builder and produce the descriptor.
    #[must_use]
    pub fn build(self) -> CustomOpDescriptor {
        CustomOpDescriptor {
            fn_id: self.fn_id,
            name: self.name,
            eval_fn: self.eval_fn,
            vectorizable: self.vectorizable,
            simplify_rules: self.simplify_rules,
            egraph_rules: self.egraph_rules,
            cost: self.cost,
        }
    }
}

// ── CustomOpError ─────────────────────────────────────────────────────────

/// Errors returned by [`CustomOpRegistry::register`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CustomOpError {
    /// A descriptor with the same `FnId` is already registered.
    DuplicateFnId(FnId),
    /// A descriptor with the same name is already registered.
    DuplicateName(String),
}

impl std::fmt::Display for CustomOpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateFnId(id) => write!(f, "custom op fn_id {id:?} already registered"),
            Self::DuplicateName(name) => {
                write!(f, "custom op name {name:?} already registered")
            }
        }
    }
}

impl std::error::Error for CustomOpError {}

// ── CustomOpRegistry ──────────────────────────────────────────────────────

/// Registry of [`CustomOpDescriptor`]s — the single source of truth for all
/// custom operators in the pipeline.
///
/// Create one registry per compilation unit (or share via `Arc`), register
/// your operators, then feed it to each pipeline step:
///
/// ```rust
/// # // cfg guard: skip when cranelift-jit feature is absent
/// # #[cfg(not(feature = "cranelift-jit"))] fn main() {}
/// # #[cfg(feature = "cranelift-jit")] fn main() {
/// use std::sync::Arc;
/// use rssn_advanced::custom::descriptor::{CustomOpDescriptor, CustomOpRegistry, EvalFn};
/// use rssn_advanced::dag::builder::DagBuilder;
/// use rssn_advanced::egraph::egraph::{EGraph, EGraphConfig};
/// use rssn_advanced::jit::compiler::JitCompiler;
///
/// extern "C" fn double_it(x: f64) -> f64 { x * 2.0 }
///
/// let mut builder = DagBuilder::new();
/// let fn_id = builder.intern_function("double");
/// let desc = CustomOpDescriptor::builder(fn_id, "double", EvalFn::Arity1(double_it)).build();
///
/// let mut registry = CustomOpRegistry::new();
/// registry.register(desc).unwrap();
/// let registry = Arc::new(registry);
///
/// // JIT: register eval_fn pointers + enable batch vectorisation
/// registry.apply_to_jit(&mut JitCompiler::try_new().unwrap());
///
/// // Simplifier: generate a RuleRegistry from all simplify_rules
/// let _rule_reg = registry.build_rule_registry();
///
/// // E-graph: inject all egraph_rules into an EGraph instance
/// {
///     let mut egraph = EGraph::new(&mut builder, EGraphConfig::default());
///     registry.apply_to_egraph(&mut egraph);
/// }
/// # }
/// ```
pub struct CustomOpRegistry {
    /// Descriptors keyed by numeric `FnId`.
    ops: HashMap<FnId, CustomOpDescriptor>,
    /// Name → `FnId` lookup for the expression parser.
    name_to_id: HashMap<String, FnId>,
}

impl Default for CustomOpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomOpRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ops: HashMap::new(),
            name_to_id: HashMap::new(),
        }
    }

    /// Register a custom operator descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`CustomOpError::DuplicateFnId`] or [`CustomOpError::DuplicateName`]
    /// if the `fn_id` or `name` is already registered.
    pub fn register(&mut self, desc: CustomOpDescriptor) -> Result<(), CustomOpError> {
        if self.ops.contains_key(&desc.fn_id) {
            return Err(CustomOpError::DuplicateFnId(desc.fn_id));
        }
        if self.name_to_id.contains_key(&desc.name) {
            return Err(CustomOpError::DuplicateName(desc.name));
        }
        self.name_to_id.insert(desc.name.clone(), desc.fn_id);
        self.ops.insert(desc.fn_id, desc);
        Ok(())
    }

    /// Look up a descriptor by numeric `FnId`.
    #[must_use]
    pub fn get(&self, fn_id: FnId) -> Option<&CustomOpDescriptor> {
        self.ops.get(&fn_id)
    }

    /// Look up a descriptor by operator name.
    #[must_use]
    pub fn get_by_name(&self, name: &str) -> Option<&CustomOpDescriptor> {
        self.name_to_id.get(name).and_then(|id| self.ops.get(id))
    }

    /// Mutably look up a descriptor by numeric `FnId`.
    ///
    /// Used by the C FFI rule-attachment functions to push additional
    /// [`SimplifyRule`]s and [`EGraphRule`]s into an already-registered
    /// descriptor during the build phase (before the registry is shared
    /// with the JIT via [`Arc`]).
    pub fn get_mut(&mut self, fn_id: FnId) -> Option<&mut CustomOpDescriptor> {
        self.ops.get_mut(&fn_id)
    }

    /// Returns `true` if the operator is registered and flagged vectorizable.
    ///
    /// Queried by `JitCompiler::compile_batch_f64x2` before emitting the ILP
    /// loop — if any `Function` node in the expression is not vectorizable the
    /// batch path is skipped.
    #[must_use]
    pub fn is_vectorizable(&self, fn_id: FnId) -> bool {
        self.ops.get(&fn_id).is_some_and(|d| d.vectorizable)
    }

    /// Iterate over all registered descriptors.
    ///
    /// Used internally by [`JitCompiler::set_custom_op_registry`] to populate
    /// the JIT's function-pointer table.
    pub fn ops_iter(&self) -> impl Iterator<Item = &CustomOpDescriptor> {
        self.ops.values()
    }

    /// Pre-intern all custom operator names in a `DagBuilder`.
    ///
    /// Call this before parsing expressions that reference custom operators,
    /// so the parser resolves names to the correct `FnId` values.
    pub fn register_with_builder(&self, builder: &mut DagBuilder) {
        for desc in self.ops.values() {
            let _id = builder.intern_function(&desc.name);
        }
    }

    // ── Pipeline integrations ─────────────────────────────────────────────

    /// Feed all custom operator evaluation functions into a [`JitCompiler`].
    ///
    /// This populates the compiler's internal custom-function registry (so
    /// the JIT can emit `call rssn_custom_fn_N` instructions) and stores the
    /// registry reference so `compile_batch_f64x2` can check `is_vectorizable`.
    #[cfg(feature = "cranelift-jit")]
    pub fn apply_to_jit(
        self: &std::sync::Arc<Self>,
        compiler: &mut crate::jit::compiler::JitCompiler,
    ) {
        compiler.set_custom_op_registry(std::sync::Arc::clone(self));
    }

    /// Build a [`RuleRegistry`] populated with every simplification rule from
    /// all registered descriptors.
    ///
    /// Pass the result to `HeuristicEngine::with_rule_registry` or to
    /// `rssn_dag_simplify_with_rules`.
    #[must_use]
    pub fn build_rule_registry(&self) -> RuleRegistry {
        let mut registry = RuleRegistry::new();
        for desc in self.ops.values() {
            for rule in &desc.simplify_rules {
                // Clone the Arc cheaply; the trampoline Box captures it.
                let arc = Arc::clone(&rule.rule);
                let trampoline: RuleFn = Box::new(move |b, k, c| arc(b, k, c));
                registry.register_named(&rule.name, trampoline, rule.priority, None);
            }
        }
        registry
    }

    /// Inject all e-graph rewrite rules from all registered descriptors into
    /// an [`EGraph`][crate::egraph::egraph::EGraph] instance.
    ///
    /// Rules with `after_builtins = false` are added before built-in rules;
    /// rules with `after_builtins = true` are added after.
    pub fn apply_to_egraph(&self, egraph: &mut crate::egraph::egraph::EGraph<'_>) {
        for desc in self.ops.values() {
            for rule in &desc.egraph_rules {
                let arc = Arc::clone(&rule.rule);
                if rule.after_builtins {
                    egraph.add_rule_after_builtins(move |b, k, c| arc(b, k, c));
                } else {
                    egraph.add_rule(move |b, k, c| arc(b, k, c));
                }
            }
        }
    }
}
