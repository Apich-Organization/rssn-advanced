//! Cost model for E-graph extraction.
//!
//! Lower cost = cheaper to evaluate. The extractor visits each e-class and
//! returns the `DagNodeId` whose recursive cost (self + children) is minimal.
//!
//! ## Design rationale
//!
//! We prefer expressions that:
//! 1. Use fewer floating-point operations overall.
//! 2. Replace `powf` (expensive transcendental) with multiply chains.
//! 3. Avoid division when a reciprocal multiplication suffices.
//!
//! The cost numbers are deliberately coarse — they rank operations by
//! typical latency on modern OOO pipelines (add ≈ mul < div ≪ sqrt < powf).

use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use crate::dag::symbol::{OpKind, SymbolKind};

/// Per-operation cost weights for the E-graph extraction pass.
///
/// All values are relative; lower = cheaper. The extractor returns the
/// e-class representative with the minimum total recursive cost.
///
/// Override individual fields to reflect a specific target microarchitecture
/// (e.g. lower `div` for AVX-512 throughput, higher `sqrt` for in-order cores).
#[derive(Debug, Clone, Copy)]
pub struct CostWeights {
    /// Terminal nodes (variable, constant). Default: `1.0`.
    pub terminal: f64,
    /// `fadd` / `fsub`. Default: `1.0`.
    pub add_sub: f64,
    /// `fmul`. Default: `2.0`.
    pub mul: f64,
    /// `fdiv`. Default: `8.0`.
    pub div: f64,
    /// Sign flip / `fneg`. Default: `1.0`.
    pub neg: f64,
    /// Integer-exponent multiply chain (`n ∈ 2..=16`). Default: `4.0`.
    pub pow_int: f64,
    /// `vsqrtpd` / `fsqrt`. Default: `12.0`.
    pub sqrt: f64,
    /// General `powf` (two transcendentals). Default: `40.0`.
    pub pow_general: f64,
    /// User-defined function call. Default: `8.0`.
    pub function: f64,
    /// `fmod`-equivalent. Default: `20.0`.
    pub modulo: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            terminal: 1.0,
            add_sub: 1.0,
            mul: 2.0,
            div: 8.0,
            neg: 1.0,
            pow_int: 4.0,
            sqrt: 12.0,
            pow_general: 40.0,
            function: 8.0,
            modulo: 20.0,
        }
    }
}

/// Computes the **total recursive cost** of the sub-expression rooted at
/// `id`, using the best-available (already extracted) child costs from
/// `child_costs`. Unknown child costs default to `weights.terminal`.
///
/// The `child_costs` map must have been populated for all children before
/// calling this for their parents (bottom-up order). Pass
/// `&CostWeights::default()` for the built-in cost table tuned to a modern
/// OOO pipeline; supply custom weights to reflect a specific target
/// (e.g. lower `div` for AVX-512 throughput).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn node_cost<S: ::std::hash::BuildHasher>(
    builder: &DagBuilder,
    id: DagNodeId,
    child_costs: &std::collections::HashMap<u32, f64, S>,
    weights: &CostWeights,
) -> f64 {
    let Some(node) = builder.arena().get(id) else {
        return weights.terminal;
    };

    let children_total: f64 = node
        .children
        .as_slice()
        .iter()
        .map(|&c| child_costs.get(&c.0).copied().unwrap_or(weights.terminal))
        .sum();

    let own = match &node.kind {
        SymbolKind::Variable(_) | SymbolKind::Constant(_) => weights.terminal,
        SymbolKind::Operator(op) => match op {
            OpKind::Add | OpKind::Sub => weights.add_sub,
            OpKind::Mul => weights.mul,
            OpKind::Div => weights.div,
            OpKind::Neg => weights.neg,
            OpKind::Mod => weights.modulo,
            OpKind::Pow => {
                let is_int_pow = node.children.as_slice().get(1).is_some_and(|&exp_id| {
                    builder.arena().get(exp_id).is_some_and(|exp_node| {
                        if let SymbolKind::Constant(v) = exp_node.kind {
                            let n = v as i64;
                            (2..=16).contains(&n) && (n as f64 - v).abs() < 1e-9
                        } else {
                            false
                        }
                    })
                });
                if is_int_pow {
                    weights.pow_int
                } else {
                    weights.pow_general
                }
            }
        },
        SymbolKind::Function(_) => weights.function,
    };

    // x^0.5 → vsqrtpd: reward sqrt path even when pow_general would normally win.
    let own = if matches!(&node.kind, SymbolKind::Operator(OpKind::Pow)) {
        let is_half_pow = node.children.as_slice().get(1).is_some_and(|&exp_id| {
            builder.arena().get(exp_id).is_some_and(|exp_node| {
                matches!(exp_node.kind, SymbolKind::Constant(v) if (v - 0.5).abs() < 1e-9)
            })
        });
        if is_half_pow { weights.sqrt } else { own }
    } else {
        own
    };

    own + children_total
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn constant_cost_is_terminal() {
        let mut b = DagBuilder::new();
        let c = b.constant(3.14);
        let costs = HashMap::new();
        let w = CostWeights::default();
        assert!((node_cost(&b, c, &costs, &w) - w.terminal).abs() < 1e-9);
    }

    #[test]
    fn add_cost_includes_children() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let y = b.variable("y");
        let s = b.add(x, y);
        let costs = HashMap::new();
        let w = CostWeights::default();
        let expected = w.add_sub + w.terminal + w.terminal;
        assert!((node_cost(&b, s, &costs, &w) - expected).abs() < 1e-9);
    }

    #[test]
    fn div_costs_more_than_mul() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let y = b.variable("y");
        let d = b.div(x, y);
        let m = b.mul(x, y);
        let costs = HashMap::new();
        let w = CostWeights::default();
        assert!(node_cost(&b, d, &costs, &w) > node_cost(&b, m, &costs, &w));
    }

    #[test]
    fn int_pow_costs_less_than_general_pow() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let c2 = b.constant(2.0);
        let c15 = b.constant(15.5); // non-integer exponent
        let p_int = b.pow(x, c2);
        let p_gen = b.pow(x, c15);
        let costs = HashMap::new();
        let w = CostWeights::default();
        assert!(node_cost(&b, p_int, &costs, &w) < node_cost(&b, p_gen, &costs, &w));
    }

    #[test]
    fn custom_weights_change_ordering() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let y = b.variable("y");
        let d = b.div(x, y);
        let m = b.mul(x, y);
        let costs = HashMap::new();
        // With equal div and mul cost, div should NOT cost more.
        let w = CostWeights {
            div: 2.0,
            mul: 2.0,
            ..CostWeights::default()
        };
        assert_eq!(
            (node_cost(&b, d, &costs, &w) - node_cost(&b, m, &costs, &w)).abs() < 1e-9,
            true,
            "div and mul should have equal cost with equal weights"
        );
    }
}
