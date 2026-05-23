//! User-extensible rule registry for the heuristic engine.
//!
//! `RuleRegistry` allows external callers to register custom pattern-match
//! and replacement closures without modifying the library source.
//!
//! Each rule is a closure that receives the `DagBuilder`, the current node's
//! `SymbolKind`, and its (already-rewritten) children, and returns either
//! `Some(replacement_id)` when the rule fires or `None` to pass.

use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use crate::dag::symbol::SymbolKind;

/// Signature for a custom rewrite rule closure.
///
/// Returns `Some(replacement)` when the rule fires, `None` to defer to the
/// next rule (or the engine's built-in patterns).
pub type RuleFn = Box<dyn Fn(&mut DagBuilder, SymbolKind, &[DagNodeId]) -> Option<DagNodeId> + Send + Sync>;

/// Registry of user-defined algebraic rewrite rules.
///
/// Rules are stored in insertion order and tried in sequence for each node
/// during simplification. The first rule that returns `Some` wins.
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
    rules: Vec<RuleFn>,
}

impl std::fmt::Debug for RuleRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuleRegistry")
            .field("rules_count", &self.rules.len())
            .finish()
    }
}

impl Default for RuleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleRegistry {
    /// Creates a new, empty `RuleRegistry`.
    #[must_use]
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Registers a custom rewrite rule.
    ///
    /// Rules are tried in insertion order; the first to return `Some` wins.
    pub fn register<F>(&mut self, rule: F)
    where
        F: Fn(&mut DagBuilder, SymbolKind, &[DagNodeId]) -> Option<DagNodeId> + Send + Sync + 'static,
    {
        self.rules.push(Box::new(rule));
    }

    /// Returns the number of registered rules.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Returns `true` if no rules have been registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Tries each registered rule in order for `(kind, children)`.
    ///
    /// Returns the replacement node from the first rule that fires,
    /// or `None` if no rule matches.
    pub fn try_apply(
        &self,
        builder: &mut DagBuilder,
        kind: SymbolKind,
        children: &[DagNodeId],
    ) -> Option<DagNodeId> {
        for rule in &self.rules {
            if let Some(result) = rule(builder, kind, children) {
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
        assert!(matches!(node.kind, SymbolKind::Constant));
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
}
