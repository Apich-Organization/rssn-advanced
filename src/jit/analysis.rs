//! Pre-codegen analysis pass: computes properties of each AST node before
//! IR is emitted. Results drive NaN-guard elision and power expansion.

use crate::ast::projection::AstProjection;
use crate::dag::symbol::{OpKind, SymbolKind};

/// Properties of a single AST node computed before codegen.
#[derive(Debug, Clone)]
pub struct NodeAnalysis {
    /// True if this subtree is provably never zero.
    /// Used to elide the `select(rhs==0, NaN, lhs/rhs)` guard in Div/Mod.
    pub is_nonzero: bool,
    /// If this is a Pow node and the exponent is a constant that can be
    /// lowered to fmul chains or sqrt, this encodes the strategy.
    pub pow_expansion: PowExpansion,
}

/// Expansion strategy for a `Pow` node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PowExpansion {
    /// No expansion (runtime powf call required).
    None,
    /// Expand to `sqrt(lhs)`.
    Sqrt,
    /// Expand to repeated fmul with binary exponentiation.
    /// Inner value: the integer exponent (2..=8).
    IntPow(u32),
}

/// Walks `ast` bottom-up and returns one `NodeAnalysis` per node (same
/// indices as `ast.nodes`). The result vector has the same length as
/// `ast.nodes`.
#[must_use]
pub fn analyze(ast: &AstProjection) -> Vec<NodeAnalysis> {
    let n = ast.nodes.len();
    // Initialise all entries to safe defaults.
    let mut results: Vec<NodeAnalysis> = (0..n)
        .map(|_| NodeAnalysis { is_nonzero: false, pow_expansion: PowExpansion::None })
        .collect();

    // Bottom-up: because the AST is stored in pre-order (root first), we
    // process in reverse order so children are always analysed before their
    // parents.
    for idx in (0..n).rev() {
        let node = &ast.nodes[idx];
        let pool = &ast.children_pool;

        let is_nonzero = match node.kind {
            SymbolKind::Constant(v) => v != 0.0 && !v.is_nan(),
            SymbolKind::Variable(_) => false,
            SymbolKind::Operator(OpKind::Mul) => {
                // Product is non-zero iff both factors are non-zero.
                let children = node.children.as_slice_with_pool(pool);
                children.iter().all(|ptr| {
                    ptr.resolve(idx)
                        .and_then(|ci| results.get(ci))
                        .map_or(false, |a| a.is_nonzero)
                })
            }
            SymbolKind::Operator(OpKind::Div) => {
                // Division result: non-zero iff both numerator and
                // denominator are non-zero (also used for guard elision on
                // the denominator side — callers check the RHS node
                // directly, not this node's analysis).
                let children = node.children.as_slice_with_pool(pool);
                children.iter().all(|ptr| {
                    ptr.resolve(idx)
                        .and_then(|ci| results.get(ci))
                        .map_or(false, |a| a.is_nonzero)
                })
            }
            SymbolKind::Operator(OpKind::Neg) => {
                // neg(x) is non-zero iff x is non-zero.
                node.children
                    .as_slice_with_pool(pool)
                    .first()
                    .and_then(|ptr| ptr.resolve(idx))
                    .and_then(|ci| results.get(ci))
                    .map_or(false, |a| a.is_nonzero)
            }
            SymbolKind::Operator(OpKind::Pow) => {
                // x^e is non-zero iff x is non-zero (we don't special-case
                // even exponents producing non-negative results here because
                // that only tells us ≥ 0, not > 0).
                node.children
                    .as_slice_with_pool(pool)
                    .first()
                    .and_then(|ptr| ptr.resolve(idx))
                    .and_then(|ci| results.get(ci))
                    .map_or(false, |a| a.is_nonzero)
            }
            // Add / Sub / Mod: too hard to prove non-zero in the general case.
            SymbolKind::Operator(OpKind::Add)
            | SymbolKind::Operator(OpKind::Sub)
            | SymbolKind::Operator(OpKind::Mod) => false,
            SymbolKind::Function(_) => false,
        };

        // Compute pow_expansion for Pow nodes.
        let pow_expansion = if let SymbolKind::Operator(OpKind::Pow) = node.kind {
            // Inspect the exponent child (second child).
            let children = node.children.as_slice_with_pool(pool);
            if children.len() >= 2 {
                let exp_expansion = children
                    .get(1)
                    .and_then(|ptr| ptr.resolve(idx))
                    .and_then(|ci| ast.nodes.get(ci))
                    .and_then(|exp_node| {
                        if let SymbolKind::Constant(_) = exp_node.kind {
                            Some(exp_node.value)
                        } else {
                            None
                        }
                    })
                    .map(classify_exponent)
                    .unwrap_or(PowExpansion::None);
                exp_expansion
            } else {
                PowExpansion::None
            }
        } else {
            PowExpansion::None
        };

        if let Some(entry) = results.get_mut(idx) {
            entry.is_nonzero = is_nonzero;
            entry.pow_expansion = pow_expansion;
        }
    }

    results
}

/// Maps a constant exponent value to the appropriate expansion strategy.
#[must_use]
fn classify_exponent(exp: f64) -> PowExpansion {
    // Handled by existing peepholes in the emitter (x^0 → 1, x^1 → x).
    if exp == 0.0 || exp == 1.0 {
        return PowExpansion::None;
    }
    // sqrt
    if (exp - 0.5_f64).abs() < f64::EPSILON {
        return PowExpansion::Sqrt;
    }
    // Integer exponents 2..=8.
    let n = exp as u32;
    if n >= 2 && n <= 8 && (n as f64 - exp).abs() < f64::EPSILON {
        return PowExpansion::IntPow(n);
    }
    PowExpansion::None
}
