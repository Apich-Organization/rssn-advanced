//! Algebraic identity patterns used by the heuristic engine.
//!
//! The `heuristic_review §1` audit found the previous engine "performed
//! a recursive traversal" without doing any actual pattern matching.
//! This module fills that gap: each pattern is a single
//! `try_apply(builder, kind, children) -> Option<DagNodeId>` that
//! returns `Some(replacement)` when its rule fires, `None` otherwise.
//!
//! Every replacement goes through `DagBuilder::*` constructors, which
//! call `DedupMap::get_or_insert` — so even aggressive rewriting
//! preserves the hash-cons invariant (`heuristic_review §1`).

// `0.0` and `1.0` have exact `f64` representations; the strict-equality
// lint is misleading for the identity rules in this file.
#![allow(clippy::float_cmp)]

use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use crate::dag::symbol::{OpKind, SymbolKind};

/// Reusable type for pattern application results.
pub type PatternResult = Option<DagNodeId>;

/// Returns the `f64` constant a node represents, or `None` for
/// non-constant nodes.
fn constant_value(builder: &DagBuilder, id: DagNodeId) -> Option<f64> {
    let node = builder.arena().get(id)?;
    if matches!(node.kind, SymbolKind::Constant) {
        node.value
    } else {
        None
    }
}

/// `x + 0 → x` and `0 + x → x`.
#[must_use]
pub fn add_zero(builder: &DagBuilder, children: &[DagNodeId]) -> PatternResult {
    if children.len() != 2 {
        return None;
    }
    let lhs = children[0];
    let rhs = children[1];
    if matches!(constant_value(builder, lhs), Some(v) if v == 0.0) {
        return Some(rhs);
    }
    if matches!(constant_value(builder, rhs), Some(v) if v == 0.0) {
        return Some(lhs);
    }
    None
}

/// `x - 0 → x` and `x - x → 0`.
pub fn sub_identity(builder: &mut DagBuilder, children: &[DagNodeId]) -> PatternResult {
    if children.len() != 2 {
        return None;
    }
    let lhs = children[0];
    let rhs = children[1];
    if matches!(constant_value(builder, rhs), Some(v) if v == 0.0) {
        return Some(lhs);
    }
    if lhs == rhs {
        return Some(builder.constant(0.0));
    }
    None
}

/// `x * 0 → 0`, `x * 1 → x`, `1 * x → x`.
pub fn mul_identity(builder: &mut DagBuilder, children: &[DagNodeId]) -> PatternResult {
    if children.len() != 2 {
        return None;
    }
    let lhs = children[0];
    let rhs = children[1];

    let lhs_val = constant_value(builder, lhs);
    let rhs_val = constant_value(builder, rhs);

    if matches!(lhs_val, Some(v) if v == 0.0) || matches!(rhs_val, Some(v) if v == 0.0) {
        return Some(builder.constant(0.0));
    }
    if matches!(lhs_val, Some(v) if v == 1.0) {
        return Some(rhs);
    }
    if matches!(rhs_val, Some(v) if v == 1.0) {
        return Some(lhs);
    }
    None
}

/// `x / 1 → x`, `0 / x → 0`, `x / x → 1` (only when `x` is a constant
/// known not to be zero — the symbolic case is generally unsafe).
pub fn div_identity(builder: &mut DagBuilder, children: &[DagNodeId]) -> PatternResult {
    if children.len() != 2 {
        return None;
    }
    let lhs = children[0];
    let rhs = children[1];

    if matches!(constant_value(builder, rhs), Some(v) if v == 1.0) {
        return Some(lhs);
    }
    if matches!(constant_value(builder, lhs), Some(v) if v == 0.0) {
        return Some(builder.constant(0.0));
    }
    // x / x → 1 only when we can prove x ≠ 0. The safe approximation:
    // both sides are the same id AND that id is a non-zero constant.
    if lhs == rhs
        && let Some(v) = constant_value(builder, lhs)
        && v != 0.0
    {
        return Some(builder.constant(1.0));
    }
    None
}

/// `x ^ 0 → 1`, `x ^ 1 → x`, `1 ^ x → 1`, `0 ^ x → 0` (for x > 0).
pub fn pow_identity(builder: &mut DagBuilder, children: &[DagNodeId]) -> PatternResult {
    if children.len() != 2 {
        return None;
    }
    let lhs = children[0];
    let rhs = children[1];

    let lhs_val = constant_value(builder, lhs);
    let rhs_val = constant_value(builder, rhs);

    if matches!(rhs_val, Some(v) if v == 0.0) {
        return Some(builder.constant(1.0));
    }
    if matches!(rhs_val, Some(v) if v == 1.0) {
        return Some(lhs);
    }
    if matches!(lhs_val, Some(v) if v == 1.0) {
        return Some(builder.constant(1.0));
    }
    if let (Some(base), Some(exp)) = (lhs_val, rhs_val)
        && base == 0.0
        && exp > 0.0
    {
        return Some(builder.constant(0.0));
    }
    None
}

/// `--x → x`.
#[must_use]
pub fn neg_double(builder: &DagBuilder, children: &[DagNodeId]) -> PatternResult {
    if children.len() != 1 {
        return None;
    }
    let child = builder.arena().get(children[0])?;
    if child.kind == SymbolKind::Operator(OpKind::Neg) {
        let inner = child.children.as_slice();
        if inner.len() == 1 {
            return Some(inner[0]);
        }
    }
    None
}

/// Dispatches to the appropriate pattern for `kind`. Returns
/// `Some(replacement_id)` when a pattern fires, else `None`.
///
/// All replacements go through `DagBuilder` and therefore preserve
/// structural deduplication.
pub fn try_apply(
    builder: &mut DagBuilder,
    kind: SymbolKind,
    children: &[DagNodeId],
) -> PatternResult {
    match kind {
        SymbolKind::Operator(OpKind::Add) => add_zero(builder, children),
        SymbolKind::Operator(OpKind::Sub) => sub_identity(builder, children),
        SymbolKind::Operator(OpKind::Mul) => mul_identity(builder, children),
        SymbolKind::Operator(OpKind::Div) => div_identity(builder, children),
        SymbolKind::Operator(OpKind::Pow) => pow_identity(builder, children),
        SymbolKind::Operator(OpKind::Neg) => neg_double(builder, children),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_zero_folds() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let zero = b.constant(0.0);
        assert_eq!(add_zero(&b, &[x, zero]), Some(x));
        assert_eq!(add_zero(&b, &[zero, x]), Some(x));
        // No zero → no fold.
        let y = b.variable("y");
        assert_eq!(add_zero(&b, &[x, y]), None);
    }

    #[test]
    fn mul_identities_fire() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let zero = b.constant(0.0);
        let one = b.constant(1.0);
        // x * 0 → 0
        assert_eq!(mul_identity(&mut b, &[x, zero]), Some(zero));
        // 1 * x → x
        assert_eq!(mul_identity(&mut b, &[one, x]), Some(x));
        // x * 1 → x
        assert_eq!(mul_identity(&mut b, &[x, one]), Some(x));
    }

    #[test]
    fn sub_x_minus_x_is_zero() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let result = sub_identity(&mut b, &[x, x]);
        assert!(result.is_some(), "x - x should fire");
        let result = result.expect("just asserted");
        assert_eq!(constant_value(&b, result), Some(0.0));
    }

    #[test]
    fn pow_zero_is_one() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let zero = b.constant(0.0);
        let result = pow_identity(&mut b, &[x, zero]);
        let id = result.expect("x^0 should fire");
        assert_eq!(constant_value(&b, id), Some(1.0));
    }

    #[test]
    fn neg_double_unwraps() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let neg_x = b.neg(x);
        let neg_neg_x = b.neg(neg_x);
        // try_apply on `Neg(Neg(x))` should unwrap to x.
        let result = neg_double(&b, &[neg_x]);
        assert_eq!(result, Some(x), "expected --x to unwrap to x");
        // Top-level dispatcher should pick the same pattern.
        assert_eq!(
            try_apply(&mut b, SymbolKind::Operator(OpKind::Neg), &[neg_x]),
            Some(x)
        );
        let _ = neg_neg_x;
    }
}
