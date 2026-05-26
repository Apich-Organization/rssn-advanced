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
//!
//! # Rule-set fingerprinting
//!
//! Because rules are closures they cannot be serialised. To detect whether the
//! rule set changed between a `PackedArenaImage` serialisation and the current
//! load, [`RuleRegistry::rule_set_fingerprint`] hashes the ordered sequence of
//! rule names. If the fingerprint stored in the image header differs from the
//! current registry's fingerprint, CANONICAL bits in the loaded arena should be
//! cleared — they were computed under a different set of rewrites.

use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use crate::dag::symbol::SymbolKind;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Signature for a custom rewrite rule closure.
///
/// Returns `Some(replacement)` when the rule fires, `None` to defer to the
/// next rule (or the engine's built-in patterns).
pub type RuleFn =
    Box<dyn Fn(&mut DagBuilder, SymbolKind, &[DagNodeId]) -> Option<DagNodeId> + Send + Sync>;

struct PrioritizedRule {
    func: RuleFn,
    priority: i32,
    #[allow(dead_code)]
    kind_filter: Option<SymbolKind>,
    /// Human-readable name used by [`RuleRegistry::rule_set_fingerprint`].
    name: String,
}

/// Rule registry with `O(rules_for_kind)` dispatch.
///
/// Rules are indexed at registration time: each `SymbolKind`-filtered rule is
/// stored in a per-kind bucket; unfiltered rules go in a wildcard bucket. Both
/// buckets are sorted by descending priority so the first match wins with no
/// sorting overhead at match time.
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
    // Index: kind_disc → sorted indices into `rules` (descending priority)
    kind_index: HashMap<u8, Vec<usize>>,
    // Unfiltered rules (applied to every kind), sorted descending priority
    wildcard_indices: Vec<usize>,
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

/// Map `SymbolKind` to a stable u8 discriminant for the index key.
const fn kind_disc(k: &SymbolKind) -> u8 {
    match k {
        SymbolKind::Variable(_) => 0,
        SymbolKind::Constant(_) => 1,
        SymbolKind::Operator(op) => 2 + (*op as u8),
        SymbolKind::Function(_) => 16,
    }
}

impl RuleRegistry {
    /// Creates a new, empty `RuleRegistry`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            kind_index: HashMap::new(),
            wildcard_indices: Vec::new(),
        }
    }

    /// Register a rule with default priority (0) and no kind filter (applies to all).
    pub fn register<F>(&mut self, rule: F)
    where
        F: Fn(&mut DagBuilder, SymbolKind, &[DagNodeId]) -> Option<DagNodeId>
            + Send
            + Sync
            + 'static,
    {
        self.register_named_impl(format!("rule#{}", self.rules.len()), rule, 0, None);
    }

    /// Register a rule with explicit priority and optional kind filter.
    ///
    /// Higher priority rules are tried first. Rules with a `kind_filter` are only
    /// tried when the node's `SymbolKind` matches — this avoids virtual calls for
    /// inapplicable rules entirely (`O(rules_for_kind)` instead of `O(total_rules)`).
    pub fn register_with_priority<F>(
        &mut self,
        rule: F,
        priority: i32,
        kind_filter: Option<SymbolKind>,
    ) where
        F: Fn(&mut DagBuilder, SymbolKind, &[DagNodeId]) -> Option<DagNodeId>
            + Send
            + Sync
            + 'static,
    {
        self.register_named_impl(
            format!("rule#{}", self.rules.len()),
            rule,
            priority,
            kind_filter,
        );
    }

    /// Register a named rule with explicit priority and optional kind filter.
    ///
    /// The `name` participates in [`Self::rule_set_fingerprint`], enabling
    /// detection of rule-set changes across serialise/deserialise round-trips.
    pub fn register_named<F>(
        &mut self,
        name: &str,
        rule: F,
        priority: i32,
        kind_filter: Option<SymbolKind>,
    ) where
        F: Fn(&mut DagBuilder, SymbolKind, &[DagNodeId]) -> Option<DagNodeId>
            + Send
            + Sync
            + 'static,
    {
        self.register_named_impl(name.to_owned(), rule, priority, kind_filter);
    }

    fn register_named_impl<F>(
        &mut self,
        name: String,
        rule: F,
        priority: i32,
        kind_filter: Option<SymbolKind>,
    ) where
        F: Fn(&mut DagBuilder, SymbolKind, &[DagNodeId]) -> Option<DagNodeId>
            + Send
            + Sync
            + 'static,
    {
        let idx = self.rules.len();
        self.rules.push(PrioritizedRule {
            func: Box::new(rule),
            priority,
            kind_filter,
            name,
        });

        if let Some(ref k) = kind_filter {
            let disc = kind_disc(k);
            let bucket = self.kind_index.entry(disc).or_default();
            bucket.push(idx);
            // Keep descending priority order.
            bucket.sort_by(|&a, &b| self.rules[b].priority.cmp(&self.rules[a].priority));
        } else {
            self.wildcard_indices.push(idx);
            self.wildcard_indices
                .sort_by(|&a, &b| self.rules[b].priority.cmp(&self.rules[a].priority));
        }
    }

    /// Returns a stable `u64` fingerprint derived from the ordered list of rule names.
    ///
    /// Rules are ordered by registration sequence (not by priority bucket). The
    /// hash is computed with `rapidhash` over the concatenation of names — stable
    /// across invocations as long as the same rules are registered in the same
    /// order.
    ///
    /// A value of `0` means "no registry" when stored in a `PackedArenaImage`
    /// header. This function never returns `0` for a non-empty registry (the
    /// rapidhash seed ensures non-zero output for any non-empty input).
    ///
    /// If fingerprints differ between the stored image and the current registry,
    /// CANONICAL bits should be cleared from all loaded nodes.
    #[must_use]
    pub fn rule_set_fingerprint(&self) -> u64 {
        if self.rules.is_empty() {
            return 0;
        }
        let mut h = rapidhash::fast::RapidHasher::default();
        for rule in &self.rules {
            rule.name.hash(&mut h);
        }
        // Guarantee non-zero so the "no registry" sentinel (0) is distinguishable.
        let fp = h.finish();
        if fp == 0 { u64::MAX } else { fp }
    }

    /// Returns an iterator over rule names in registration order.
    pub fn named_rules(&self) -> impl Iterator<Item = &str> {
        self.rules.iter().map(|r| r.name.as_str())
    }

    /// Returns the number of registered rules.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.rules.len()
    }

    /// Returns `true` if no rules have been registered.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// `O(rules_for_kind` + `wildcard_rules`) dispatch — skips all rules
    /// registered for a different kind entirely.
    ///
    /// Kind-specific rules (higher specificity) are tried before wildcard rules.
    /// Returns the replacement node from the first rule that fires, or `None` if
    /// no rule matches.
    pub fn try_apply(
        &self,
        builder: &mut DagBuilder,
        kind: SymbolKind,
        children: &[DagNodeId],
    ) -> Option<DagNodeId> {
        let disc = kind_disc(&kind);

        // Kind-specific rules first (higher specificity = higher effective priority).
        if let Some(indices) = self.kind_index.get(&disc) {
            for &idx in indices {
                if let Some(result) = (self.rules[idx].func)(builder, kind, children) {
                    return Some(result);
                }
            }
        }

        // Wildcard rules.
        for &idx in &self.wildcard_indices {
            if let Some(result) = (self.rules[idx].func)(builder, kind, children) {
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
            if kind == SymbolKind::Operator(OpKind::Add)
                && children.len() == 2
                && children[0] == children[1]
            {
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
        reg.register_with_priority(
            |builder, kind, _children| {
                if matches!(kind, SymbolKind::Operator(OpKind::Add)) {
                    Some(builder.constant(1.0))
                } else {
                    None
                }
            },
            -10,
            None,
        );
        // High-priority rule returns 2.0.
        reg.register_with_priority(
            |builder, kind, _children| {
                if matches!(kind, SymbolKind::Operator(OpKind::Add)) {
                    Some(builder.constant(2.0))
                } else {
                    None
                }
            },
            10,
            None,
        );

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
        reg.register_with_priority(
            move |_builder, _kind, _children| {
                fired_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                None
            },
            0,
            Some(SymbolKind::Operator(OpKind::Sub)),
        );

        let mut b = DagBuilder::new();
        let x = b.variable("x");
        // Apply for Mul — the Sub-filtered rule must be skipped.
        let result = reg.try_apply(&mut b, SymbolKind::Operator(OpKind::Mul), &[x, x]);
        assert!(result.is_none());
        assert!(
            !fired.load(std::sync::atomic::Ordering::SeqCst),
            "kind-filtered rule must not fire"
        );
    }

    #[test]
    fn indexed_dispatch_only_consults_matching_kind_rules() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let add_counter = Arc::new(AtomicUsize::new(0));
        let mul_counter = Arc::new(AtomicUsize::new(0));

        let mut reg = RuleRegistry::new();

        // Register 100 Add-filtered rules that count how many times they run.
        for _ in 0..100 {
            let c = Arc::clone(&add_counter);
            reg.register_with_priority(
                move |_b, _k, _ch| {
                    c.fetch_add(1, Ordering::SeqCst);
                    None
                },
                0,
                Some(SymbolKind::Operator(OpKind::Add)),
            );
        }

        // Register 100 Mul-filtered rules that count how many times they run.
        for _ in 0..100 {
            let c = Arc::clone(&mul_counter);
            reg.register_with_priority(
                move |_b, _k, _ch| {
                    c.fetch_add(1, Ordering::SeqCst);
                    None
                },
                0,
                Some(SymbolKind::Operator(OpKind::Mul)),
            );
        }

        let mut b = DagBuilder::new();
        let x = b.variable("x");

        // Dispatch for Mul — only the 100 Mul rules should run.
        let _ = reg.try_apply(&mut b, SymbolKind::Operator(OpKind::Mul), &[x, x]);

        assert_eq!(
            add_counter.load(Ordering::SeqCst),
            0,
            "Add-filtered rules must NOT run when dispatching for Mul"
        );
        assert_eq!(
            mul_counter.load(Ordering::SeqCst),
            100,
            "All 100 Mul-filtered rules must run"
        );
    }
}
