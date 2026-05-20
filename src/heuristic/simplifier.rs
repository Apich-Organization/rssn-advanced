//! Approximate simplification for graceful degradation.
//!
//! Replaces the previous "fold deep subtrees to 1.0" stub
//! (`heuristic_review §2`) — which was mathematically unsound — with
//! a coefficient-based pruning pass:
//!
//! * Walks additive chains and drops terms whose absolute coefficient
//!   falls below a threshold derived from `aggressiveness`.
//! * Leaves non-additive structure intact.
//! * Goes through [`DagBuilder`] for every replacement, so dedup is
//!   never bypassed.
//!
//! At `aggressiveness == 0.0` the pass is a no-op; at `1.0` the
//! threshold is so aggressive that only the dominant terms survive.

use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use crate::dag::symbol::{OpKind, SymbolKind};

/// Performs lossy approximate simplification to keep symbol explosion
/// in check.
///
/// `aggressiveness ∈ [0.0, 1.0]`:
/// * < 0.1: no-op (returns `root` unchanged).
/// * 0.1–0.5: prunes terms whose coefficient is below ~1e-3.
/// * 0.5–0.8: prunes terms whose coefficient is below ~1e-2.
/// * 0.8–1.0: prunes terms whose coefficient is below ~1e-1.
#[must_use]
pub fn approximate_simplify(
    builder: &mut DagBuilder,
    root: DagNodeId,
    aggressiveness: f64,
) -> DagNodeId {
    if aggressiveness < 0.1 || root.is_none() {
        return root;
    }

    let threshold = coefficient_threshold(aggressiveness);
    prune_additive_chain(builder, root, threshold)
}

fn coefficient_threshold(aggressiveness: f64) -> f64 {
    let clamped = aggressiveness.clamp(0.0, 1.0);
    // Map [0, 1] to roughly [1e-4, 1e-1] log-linearly. We err on the
    // conservative side so the lossy pass doesn't accidentally erase
    // meaningful terms with small numeric coefficients.
    let exponent = -4.0 + clamped * 3.0; // -4..-1
    10.0_f64.powf(exponent)
}

/// Walks a (potentially nested) additive chain rooted at `root` and
/// rebuilds it without any operand whose effective coefficient is
/// below `threshold`. Non-additive subtrees are passed through.
fn prune_additive_chain(
    builder: &mut DagBuilder,
    root: DagNodeId,
    threshold: f64,
) -> DagNodeId {
    let Some(node) = builder.arena().get(root) else {
        return root;
    };
    // Only act on Add nodes; everything else is structurally returned.
    if !matches!(node.kind, SymbolKind::Operator(OpKind::Add)) {
        return root;
    }

    // Collect every term in the addition chain, iteratively. We treat
    // adjacent `+` nodes as one big variadic sum.
    let mut terms: Vec<DagNodeId> = Vec::new();
    let mut stack: Vec<DagNodeId> = vec![root];
    while let Some(id) = stack.pop() {
        match builder.arena().get(id) {
            Some(n) if matches!(n.kind, SymbolKind::Operator(OpKind::Add)) => {
                for child in n.children.as_slice().iter().rev() {
                    stack.push(*child);
                }
            }
            _ => terms.push(id),
        }
    }

    // Filter terms by coefficient magnitude. Variables/operator nodes
    // get `meta.coefficient`; constants get their literal value.
    let kept: Vec<DagNodeId> = terms
        .into_iter()
        .filter(|&id| {
            let coef = magnitude_of_term(builder, id);
            coef.abs() >= threshold
        })
        .collect();

    match kept.len() {
        0 => builder.constant(0.0),
        1 => kept[0],
        _ => {
            // Fold a left-associative `add` chain through the builder.
            let mut acc = kept[0];
            for &term in &kept[1..] {
                acc = builder.add(acc, term);
            }
            acc
        }
    }
}

fn magnitude_of_term(builder: &DagBuilder, id: DagNodeId) -> f64 {
    builder.arena().get(id).map_or(0.0, |n| match n.kind {
        SymbolKind::Constant => n.value.unwrap_or(0.0).abs(),
        _ => n.meta.coefficient.abs(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_op_below_threshold() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let y = b.variable("y");
        let sum = b.add(x, y);
        let result = approximate_simplify(&mut b, sum, 0.05);
        assert_eq!(result, sum, "aggressiveness < 0.1 should be a no-op");
    }

    #[test]
    fn prunes_tiny_constants() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let tiny = b.constant(1e-6);
        let sum = b.add(x, tiny);
        // At aggressiveness=0.6 the threshold is roughly 10^(-4+3*0.6)
        // = 10^(-2.2) ≈ 6.3e-3, well above 1e-6 so `tiny` is dropped.
        let result = approximate_simplify(&mut b, sum, 0.6);
        assert_eq!(result, x, "tiny constant term should be pruned");
    }

    #[test]
    fn keeps_meaningful_terms() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let y = b.variable("y");
        let sum = b.add(x, y);
        // Aggressiveness 1.0 → threshold ~0.1. Variable coefficients
        // default to 1.0, well above the threshold, so both terms are
        // kept and the original add is reconstructed (via dedup).
        let result = approximate_simplify(&mut b, sum, 1.0);
        assert_eq!(result, sum, "non-tiny terms must survive");
    }

    #[test]
    fn all_pruned_returns_zero() {
        let mut b = DagBuilder::new();
        let tiny1 = b.constant(1e-6);
        let tiny2 = b.constant(2e-6);
        let sum = b.add(tiny1, tiny2);
        let result = approximate_simplify(&mut b, sum, 1.0);
        let v = b
            .arena()
            .get(result)
            .and_then(|n| n.value)
            .expect("constant 0 expected");
        assert!(v.abs() < f64::EPSILON);
    }

    #[test]
    fn non_additive_root_is_passthrough() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let y = b.variable("y");
        let prod = b.mul(x, y);
        let result = approximate_simplify(&mut b, prod, 1.0);
        assert_eq!(result, prod);
    }
}
