//! User-extensible rule registry for the heuristic engine.
//!
//! `RuleRegistry` allows external callers to register custom pattern-match
//! and replacement closures without modifying the library source.
//!
//! Each rule is a closure that receives the `DagBuilder`, the current node's
//! `SymbolKind`, and its (already-rewritten) children, and returns either
//! `Some(replacement_id)` when the rule fires or `None` to pass.
//!
//! Rules support priority ordering and optional kind filters for efficient dispatch.

use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use crate::dag::symbol::SymbolKind;

/// Signature for a custom rewrite rule closure.
///
/// Returns `Some(replacement)` when the rule fires, `None` to defer to the
/// next rule (or the engine's built-in patterns).
pub type RuleFn = Box<dyn Fn(&mut DagBuilder, SymbolKind, &[DagNodeId]) -> Option<DagNodeId> + Send + Sync>;

struct PrioritizedRule {
    func: RuleFn,
    priority: i32,
    kind_filter: Option<SymbolKind>,
}

/// Registry of user-defined algebraic rewrite rules.
///
/// Rules support priority ordering (higher priority tried first) and optional
/// kind filters that skip rules inapplicable to the current node kind.
///
/// # Example
///
/// ```rust
/// use rssn_advanced::heuristic::rule_registry::RuleRegistry;
/// use rssn_advanced::dag::symbol::{SymbolKind, OpKind};
///
/// let mut reg = RuleRegistry::new();
/// // x - x → 0
/// reg.register(|builder, kind, children| {
///     if kind == SymbolKind::Operator(OpKind::Sub) && children.len() == 2 && children[0] == children[1] {
///         Some(builder.constant(0.0))
///     } else {
///         None
///     }
/// });
/// ```
pub struct RuleRegistry {
    rules: Vec<PrioritizedRule>,
}

impl std::fmt::Debug for RuleRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuleRegistry")
            .field("rules_count", &self.rules.len())
            .finish()
    }
}

impl Default for RuleRegistry { fn default() -> Self { Self::new() } }

impl RuleRegistry {
    /// Creates a new, empty `RuleRegistry`.
    #[must_use]
    pub fn new() -> Self { Self { rules: Vec::new() } }

    /// Register a rule with default priority (0) and no kind filter (applies to all).
    pub fn register<F>(&mut self, rule: F)
    where F: Fn(&mut DagBuilder, SymbolKind, &[DagNodeId]) -> Option<DagNodeId> + Send + Sync + 'static {
        self.register_with_priority(rule, 0, None);
    }

    /// Register a rule with explicit priority and optional kind filter.
    ///
    /// Higher priority rules are tried first. Rules with a `kind_filter` are only
    /// tried when the node's `SymbolKind` matches — this skips irrelevant rules.
    pub fn register_with_priority<F>(&mut self, rule: F, priority: i32, kind_filter: Option<SymbolKind>)
    where F: Fn(&mut DagBuilder, SymbolKind, &[DagNodeId]) -> Option<DagNodeId> + Send + Sync + 'static {
        self.rules.push(PrioritizedRule { func: Box::new(rule), priority, kind_filter });
        // Keep descending priority order; stable sort preserves insertion order within same priority.
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Returns the number of registered rules.
    #[must_use]
    pub fn len(&self) -> usize { self.rules.len() }

    /// Returns `true` if no rules have been registered.
    #[must_use]
    pub fn is_empty(&self) -> bool { self.rules.is_empty() }

    /// Tries each registered rule in priority order for `(kind, children)`.
    ///
    /// Rules with a `kind_filter` that doesn't match `kind` are skipped entirely.
    /// Returns the replacement node from the first rule that fires, or `None` if
    /// no rule matches.
    pub fn try_apply(&self, builder: &mut DagBuilder, kind: SymbolKind, children: &[DagNodeId]) -> Option<DagNodeId> {
        for rule in &self.rules {
            // Skip rules that don't apply to this kind.
            if let Some(ref filter) = rule.kind_filter {
                if *filter != kind { continue; }
            }
            if let Some(result) = (rule.func)(builder, kind, children) {
                return Some(result);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::builder::DagBuilder;
    use crate::dag::symbol::{OpKind, SymbolKind};

    #[test]
    fn custom_rule_fires() {
        let mut reg = RuleRegistry::new();
        // Register: x - x → 0
        reg.register(|builder, kind, children| {
            if kind == SymbolKind::Operator(OpKind::Sub)
                && children.len() == 2
                && children[0] == children[1]
            {
                Some(builder.constant(0.0))
            } else {
                None
            }
        });

        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let result = reg.try_apply(&mut b, SymbolKind::Operator(OpKind::Sub), &[x, x]);
        assert!(result.is_some(), "x - x rule should fire");
        let node = b.arena().get(result.unwrap()).unwrap();
        assert!(matches!(node.kind, SymbolKind::Constant(_)));
    }

    #[test]
    fn multiple_rules_first_wins() {
        let mut reg = RuleRegistry::new();
        reg.register(|_builder, kind, _children| {
            if kind == SymbolKind::Operator(OpKind::Add) {
                None // pass
            } else {
                None
            }
        });
        reg.register(|builder, kind, children| {
            if kind == SymbolKind::Operator(OpKind::Add) && children.len() == 2 && children[0] == children[1] {
                Some(builder.constant(999.0))
            } else {
                None
            }
        });
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let result = reg.try_apply(&mut b, SymbolKind::Operator(OpKind::Add), &[x, x]);
        assert!(result.is_some());
    }

    #[test]
    fn priority_ordering_respected() {
        let mut reg = RuleRegistry::new();
        // Low-priority rule returns 1.0.
        reg.register_with_priority(|builder, kind, _children| {
            if matches!(kind, SymbolKind::Operator(OpKind::Add)) {
                Some(builder.constant(1.0))
            } else { None }
        }, -10, None);
        // High-priority rule returns 2.0.
        reg.register_with_priority(|builder, kind, _children| {
            if matches!(kind, SymbolKind::Operator(OpKind::Add)) {
                Some(builder.constant(2.0))
            } else { None }
        }, 10, None);

        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let result = reg.try_apply(&mut b, SymbolKind::Operator(OpKind::Add), &[x, x]);
        let result_id = result.expect("should fire");
        let node = b.arena().get(result_id).unwrap();
        // High-priority rule (2.0) should win.
        assert!(matches!(node.kind, SymbolKind::Constant(v) if (v - 2.0).abs() < f64::EPSILON));
    }

    #[test]
    fn kind_filter_skips_non_matching_rules() {
        let mut reg = RuleRegistry::new();
        let fired = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let fired_clone = std::sync::Arc::clone(&fired);
        // Rule only for Sub — should NOT fire for Mul.
        reg.register_with_priority(move |_builder, _kind, _children| {
            fired_clone.store(true, std::sync::atomic::Ordering::SeqCst);
            None
        }, 0, Some(SymbolKind::Operator(OpKind::Sub)));

        let mut b = DagBuilder::new();
        let x = b.variable("x");
        // Apply for Mul — the Sub-filtered rule must be skipped.
        let result = reg.try_apply(&mut b, SymbolKind::Operator(OpKind::Mul), &[x, x]);
        assert!(result.is_none());
        assert!(!fired.load(std::sync::atomic::Ordering::SeqCst), "kind-filtered rule must not fire");
    }
}
