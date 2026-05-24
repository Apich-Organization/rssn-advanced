//! Pre-codegen analysis pass: computes properties of each AST node before
//! IR is emitted. Results drive NaN-guard elision and power expansion.
//!
//! This module implements a **full abstract interpretation** framework that
//! tracks value bounds, sign properties, NaN-freedom, and power-expansion
//! strategies in a single bottom-up pass over the AST.

use crate::ast::projection::AstProjection;
use crate::dag::symbol::{OpKind, SymbolKind};

/// Abstract value domain for a single AST node.
///
/// All fields are sound over-approximations: if a field says "true", it is
/// provably true; if it says "false" or "None", we simply do not know.
#[derive(Debug, Clone)]
pub struct NodeAnalysis {
    /// Proven lower bound (value is always ≥ lower_bound if Some).
    pub lower_bound: Option<f64>,
    /// Proven upper bound (value is always ≤ upper_bound if Some).
    pub upper_bound: Option<f64>,
    /// True if the value is provably ≥ 0.0.
    pub is_nonnegative: bool,
    /// True if the value is provably > 0.0 (implies non-zero AND non-negative).
    pub is_positive: bool,
    /// True if the value cannot be NaN (both inputs finite, no div-by-zero).
    pub no_nan: bool,
    /// For Pow nodes: how to expand without calling powf.
    pub pow_expansion: PowExpansion,
}

impl NodeAnalysis {
    /// True if provably non-zero (either positive, or lower_bound > 0, or upper_bound < 0).
    #[inline]
    #[must_use]
    pub fn is_nonzero(&self) -> bool {
        self.is_positive
            || self.lower_bound.map_or(false, |lb| lb > 0.0)
            || self.upper_bound.map_or(false, |ub| ub < 0.0)
    }

    /// Neutral default: nothing is known.
    fn unknown() -> Self {
        Self {
            lower_bound: None,
            upper_bound: None,
            is_nonnegative: false,
            is_positive: false,
            no_nan: false,
            pow_expansion: PowExpansion::None,
        }
    }
}

/// Expansion strategy for a `Pow` node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PowExpansion {
    /// No expansion (runtime powf call required).
    None,
    /// Expand to `sqrt(lhs)`.
    Sqrt,
    /// Expand to repeated fmul using an addition-chain schedule.
    /// Inner value: the integer exponent (2..=16).
    IntPow(u32),
    /// Expand to `1 / x^n` using integer power then reciprocal.
    /// Inner value: the positive integer n (1..=8), representing exponent = -n.
    NegIntPow(u32),
}

/// Walks `ast` bottom-up and returns one `NodeAnalysis` per node (same
/// indices as `ast.nodes`). The result vector has the same length as
/// `ast.nodes`.
#[must_use]
pub fn analyze(ast: &AstProjection) -> Vec<NodeAnalysis> {
    let n = ast.nodes.len();
    // Initialise all entries to safe (unknown) defaults.
    let mut results: Vec<NodeAnalysis> = (0..n).map(|_| NodeAnalysis::unknown()).collect();

    // Bottom-up: because the AST is stored in pre-order (root first), we
    // process in reverse order so children are always analysed before their
    // parents.
    for idx in (0..n).rev() {
        let node = &ast.nodes[idx];
        let pool = &ast.children_pool;
        let children = node.children.as_slice_with_pool(pool);

        let an = match node.kind {
            // ── Constant ──────────────────────────────────────────────────
            SymbolKind::Constant(_) => {
                let v = node.value; // node.value holds the actual f64
                let is_nonneg = v >= 0.0;
                let is_pos = v > 0.0;
                let is_finite = !v.is_nan() && !v.is_infinite();
                NodeAnalysis {
                    lower_bound: Some(v),
                    upper_bound: Some(v),
                    is_nonnegative: is_nonneg,
                    is_positive: is_pos,
                    no_nan: is_finite,
                    pow_expansion: PowExpansion::None,
                }
            }

            // ── Variable ──────────────────────────────────────────────────
            SymbolKind::Variable(_) => NodeAnalysis::unknown(),

            // ── Neg ───────────────────────────────────────────────────────
            SymbolKind::Operator(OpKind::Neg) => {
                let x = child_analysis(&results, children, idx, 0);
                // Flip bounds: -(x) has lb = -x.ub, ub = -x.lb
                let new_lower = x.upper_bound.map(|u| -u);
                let new_upper = x.lower_bound.map(|l| -l);
                let is_nonneg = x.upper_bound.map_or(false, |u| u <= 0.0);
                let is_pos = x.upper_bound.map_or(false, |u| u < 0.0);
                NodeAnalysis {
                    lower_bound: new_lower,
                    upper_bound: new_upper,
                    is_nonnegative: is_nonneg,
                    is_positive: is_pos,
                    no_nan: x.no_nan,
                    pow_expansion: PowExpansion::None,
                }
            }

            // ── Add ───────────────────────────────────────────────────────
            SymbolKind::Operator(OpKind::Add) => {
                let a = child_analysis(&results, children, idx, 0);
                let b = child_analysis(&results, children, idx, 1);
                let lower = a.lower_bound.zip(b.lower_bound).map(|(al, bl)| al + bl);
                let upper = a.upper_bound.zip(b.upper_bound).map(|(au, bu)| au + bu);
                let is_nonneg = a.is_nonnegative && b.is_nonnegative;
                let is_pos =
                    (a.is_positive && b.is_nonnegative) || (a.is_nonnegative && b.is_positive);
                NodeAnalysis {
                    lower_bound: lower,
                    upper_bound: upper,
                    is_nonnegative: is_nonneg,
                    is_positive: is_pos,
                    no_nan: a.no_nan && b.no_nan,
                    pow_expansion: PowExpansion::None,
                }
            }

            // ── Sub ───────────────────────────────────────────────────────
            SymbolKind::Operator(OpKind::Sub) => {
                let a = child_analysis(&results, children, idx, 0);
                let b = child_analysis(&results, children, idx, 1);
                // a - b ≥ a_lb - b_ub  (tightest lower bound)
                let lower = a.lower_bound.zip(b.upper_bound).map(|(al, bu)| al - bu);
                // a - b ≤ a_ub - b_lb  (tightest upper bound)
                let upper = a.upper_bound.zip(b.lower_bound).map(|(au, bl)| au - bl);
                // Derive sign info from bounds
                let is_nonneg = lower.map_or(false, |l| l >= 0.0);
                let is_pos = lower.map_or(false, |l| l > 0.0);
                NodeAnalysis {
                    lower_bound: lower,
                    upper_bound: upper,
                    is_nonnegative: is_nonneg,
                    is_positive: is_pos,
                    no_nan: a.no_nan && b.no_nan,
                    pow_expansion: PowExpansion::None,
                }
            }

            // ── Mul ───────────────────────────────────────────────────────
            SymbolKind::Operator(OpKind::Mul) => {
                let a = child_analysis(&results, children, idx, 0);
                let b = child_analysis(&results, children, idx, 1);

                // 4-corner interval multiplication: when all four bounds are
                // known we compute the exact product interval as the min/max
                // over all combinations (al*bl, al*bu, au*bl, au*bu). This
                // is correct for all sign combinations and tighter than the
                // all-nonneg-only check we used previously.
                let (lower, upper) =
                    match (a.lower_bound, a.upper_bound, b.lower_bound, b.upper_bound) {
                        (Some(al), Some(au), Some(bl), Some(bu)) => {
                            let c0 = al * bl;
                            let c1 = al * bu;
                            let c2 = au * bl;
                            let c3 = au * bu;
                            // Gracefully handle NaN/Inf products by propagating None.
                            if [c0, c1, c2, c3].iter().all(|v| v.is_finite()) {
                                let lo = c0.min(c1).min(c2).min(c3);
                                let hi = c0.max(c1).max(c2).max(c3);
                                (Some(lo), Some(hi))
                            } else {
                                (None, None)
                            }
                        }
                        // At least one bound pair unknown: fall back to sign-only tracking.
                        _ => (None, None),
                    };

                let is_nonneg = a.is_nonnegative && b.is_nonnegative;
                // Sign flip: both negative → positive.
                let is_nonneg_both_neg = a.upper_bound.map_or(false, |u| u < 0.0)
                    && b.upper_bound.map_or(false, |u| u < 0.0);
                let is_nonneg_any =
                    is_nonneg || is_nonneg_both_neg || lower.map_or(false, |l| l >= 0.0);
                let is_pos = a.is_positive && b.is_positive;
                let is_pos_both_neg = a.upper_bound.map_or(false, |u| u < 0.0)
                    && b.upper_bound.map_or(false, |u| u < 0.0);
                let is_pos_any = is_pos || is_pos_both_neg || lower.map_or(false, |l| l > 0.0);

                NodeAnalysis {
                    lower_bound: lower,
                    upper_bound: upper,
                    is_nonnegative: is_nonneg_any,
                    is_positive: is_pos_any,
                    no_nan: a.no_nan && b.no_nan,
                    pow_expansion: PowExpansion::None,
                }
            }

            // ── Div ───────────────────────────────────────────────────────
            SymbolKind::Operator(OpKind::Div) => {
                let a = child_analysis(&results, children, idx, 0);
                let b = child_analysis(&results, children, idx, 1);
                let is_nonneg = a.is_nonnegative && b.is_positive;
                let is_pos = a.is_positive && b.is_positive;
                // no_nan: if denominator is provably nonzero AND both are nan-free
                let no_nan = a.no_nan && b.is_nonzero() && b.no_nan;
                NodeAnalysis {
                    lower_bound: None,
                    upper_bound: None,
                    is_nonnegative: is_nonneg,
                    is_positive: is_pos,
                    no_nan,
                    pow_expansion: PowExpansion::None,
                }
            }

            // ── Mod ───────────────────────────────────────────────────────
            SymbolKind::Operator(OpKind::Mod) => NodeAnalysis::unknown(),

            // ── Pow ───────────────────────────────────────────────────────
            SymbolKind::Operator(OpKind::Pow) => {
                let base = child_analysis(&results, children, idx, 0);
                // Read the exponent's constant value. We check two sources:
                //  1. The exponent node is literally `SymbolKind::Constant`.
                //  2. The exponent node's already-computed analysis has tight
                //     bounds (lower == upper), e.g. for `Neg(Constant(1.0))`
                //     the analysis produces lb = ub = -1.0.
                let exp_child_idx = children.get(1).and_then(|ptr| ptr.resolve(idx));
                let exp_val: Option<f64> = exp_child_idx
                    .and_then(|ci| ast.nodes.get(ci))
                    .and_then(|exp_node| {
                        if let SymbolKind::Constant(_) = exp_node.kind {
                            Some(exp_node.value)
                        } else {
                            None
                        }
                    })
                    // Fallback: check if the exponent's analysis has a tight bound.
                    .or_else(|| {
                        let exp_an = exp_child_idx.and_then(|ci| results.get(ci))?;
                        match (exp_an.lower_bound, exp_an.upper_bound) {
                            (Some(lb), Some(ub)) if (lb - ub).abs() < f64::EPSILON => Some(lb),
                            _ => None,
                        }
                    });

                let pow_expansion = exp_val.map(classify_exponent).unwrap_or(PowExpansion::None);

                // Compute sign/bound properties based on exponent
                let (lower_bound, upper_bound, is_nonnegative, is_positive, no_nan) = match exp_val
                {
                    Some(exp) => {
                        let n = exp as i32;
                        let is_even_int =
                            n >= 2 && (n as f64 - exp).abs() < f64::EPSILON && n % 2 == 0;
                        let is_odd_pos_int =
                            n >= 1 && (n as f64 - exp).abs() < f64::EPSILON && n % 2 != 0;
                        let is_sqrt = (exp - 0.5_f64).abs() < f64::EPSILON;
                        let is_neg_exp = exp < 0.0;

                        if is_even_int {
                            // x^even ≥ 0 always; > 0 if base provably nonzero
                            let is_pos = base.is_nonzero();
                            (Some(0.0), None, true, is_pos, base.no_nan)
                        } else if is_sqrt {
                            // sqrt(x) ≥ 0; no_nan if base ≥ 0
                            (
                                Some(0.0),
                                None,
                                true,
                                base.is_positive,
                                base.no_nan && base.is_nonnegative,
                            )
                        } else if is_odd_pos_int {
                            // sign follows base
                            (
                                None,
                                None,
                                base.is_nonnegative,
                                base.is_positive,
                                base.no_nan,
                            )
                        } else if is_neg_exp {
                            // x^(-n): nonzero if base nonzero; sign follows base if n is even
                            let neg_n = (-exp) as u32;
                            let is_even_neg = neg_n % 2 == 0;
                            let is_nonneg = if is_even_neg {
                                true
                            } else {
                                base.is_nonnegative
                            };
                            let is_pos = if is_even_neg {
                                base.is_nonzero()
                            } else {
                                base.is_positive
                            };
                            (
                                None,
                                None,
                                is_nonneg,
                                is_pos,
                                base.no_nan && base.is_nonzero(),
                            )
                        } else {
                            (None, None, false, false, false)
                        }
                    }
                    None => (None, None, false, false, false),
                };

                NodeAnalysis {
                    lower_bound,
                    upper_bound,
                    is_nonnegative,
                    is_positive,
                    no_nan,
                    pow_expansion,
                }
            }

            // ── Function ──────────────────────────────────────────────────
            SymbolKind::Function(_) => NodeAnalysis::unknown(),
        };

        results[idx] = an;
    }

    results
}

/// Helper: borrow the analysis of child at position `pos` within `children`.
///
/// Returns a reference to `NodeAnalysis::unknown()` static if the child is
/// missing or out of bounds — prevents panics in malformed trees.
fn child_analysis<'a>(
    results: &'a [NodeAnalysis],
    children: &[crate::ast::pointer::RelPtr<crate::ast::projection::AstNode>],
    parent_idx: usize,
    pos: usize,
) -> &'a NodeAnalysis {
    static UNKNOWN: std::sync::OnceLock<NodeAnalysis> = std::sync::OnceLock::new();
    let unknown = UNKNOWN.get_or_init(NodeAnalysis::unknown);
    children
        .get(pos)
        .and_then(|ptr| ptr.resolve(parent_idx))
        .and_then(|ci| results.get(ci))
        .unwrap_or(unknown)
}

/// Maps a constant exponent value to the appropriate expansion strategy.
///
/// Expansions covered:
/// - `0.5` → [`PowExpansion::Sqrt`] (native `sqrtsd` / `fsqrt`)
/// - `2..=16` → [`PowExpansion::IntPow`] (addition-chain of `fmul`s)
/// - `-1..=-8` → [`PowExpansion::NegIntPow`] (`1/x^n` via `emit_int_pow`)
/// - anything else → [`PowExpansion::None`] (runtime `powf` call)
///
/// Exponents `0.0` and `1.0` are excluded because they are handled by
/// cheaper constant-fold peepholes in the IR emitter before this strategy
/// is consulted.
#[must_use]
pub fn classify_exponent(exp: f64) -> PowExpansion {
    // x^0 → 1 and x^1 → x are handled by cheaper constant peepholes.
    if exp == 0.0 || exp == 1.0 {
        return PowExpansion::None;
    }
    // sqrt: x^0.5 → sqrt(x)
    if (exp - 0.5_f64).abs() < f64::EPSILON {
        return PowExpansion::Sqrt;
    }

    // Positive integer exponents 2..=16 → addition-chain fmul sequence.
    if exp > 0.0 {
        let n = exp as u32;
        // Confirm exact integer (no fractional part within epsilon).
        if n >= 2 && n <= 16 && (n as f64 - exp).abs() < f64::EPSILON {
            return PowExpansion::IntPow(n);
        }
    }

    // Negative integer exponents -1..=-8 → 1/x^n.
    if exp < 0.0 {
        let n = (-exp) as u32;
        // Confirm exact negative integer.
        if n >= 1 && n <= 8 && (-(n as f64) - exp).abs() < f64::EPSILON {
            return PowExpansion::NegIntPow(n);
        }
    }

    PowExpansion::None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_covers_positive_ints_2_to_16() {
        for n in 2u32..=16 {
            assert_eq!(
                classify_exponent(n as f64),
                PowExpansion::IntPow(n),
                "classify_exponent({n})"
            );
        }
    }

    #[test]
    fn classify_covers_neg_ints_minus1_to_minus8() {
        for n in 1u32..=8 {
            assert_eq!(
                classify_exponent(-(n as f64)),
                PowExpansion::NegIntPow(n),
                "classify_exponent(-{n})"
            );
        }
    }

    #[test]
    fn classify_sqrt() {
        assert_eq!(classify_exponent(0.5), PowExpansion::Sqrt);
    }

    #[test]
    fn classify_identity_and_zero_return_none() {
        assert_eq!(classify_exponent(0.0), PowExpansion::None);
        assert_eq!(classify_exponent(1.0), PowExpansion::None);
    }

    #[test]
    fn classify_non_integer_fraction_returns_none() {
        assert_eq!(classify_exponent(1.5), PowExpansion::None);
        assert_eq!(classify_exponent(2.7), PowExpansion::None);
        assert_eq!(classify_exponent(-1.5), PowExpansion::None);
    }

    #[test]
    fn classify_large_or_unknown_exponent_returns_none() {
        assert_eq!(classify_exponent(17.0), PowExpansion::None);
        assert_eq!(classify_exponent(-9.0), PowExpansion::None);
        assert_eq!(classify_exponent(100.0), PowExpansion::None);
    }

    #[test]
    fn analysis_constant_node_exact_bounds() {
        // A constant node should have lb == ub == value.
        use crate::ast::convert::dag_to_ast;
        use crate::dag::builder::DagBuilder;
        let mut b = DagBuilder::new();
        let _c = b.constant(3.5);
        let root = b.constant(3.5);
        let ast = dag_to_ast(b.arena(), root);
        let analysis = analyze(&ast);
        let an = &analysis[0];
        assert_eq!(an.lower_bound, Some(3.5));
        assert_eq!(an.upper_bound, Some(3.5));
        assert!(an.is_nonnegative);
        assert!(an.is_positive);
        assert!(an.no_nan);
    }

    #[test]
    fn analysis_neg_constant() {
        use crate::ast::convert::dag_to_ast;
        use crate::dag::builder::DagBuilder;
        let mut b = DagBuilder::new();
        let c = b.constant(-2.0);
        let root = b.neg(c);
        let ast = dag_to_ast(b.arena(), root);
        let analysis = analyze(&ast);
        // Root is Neg(-2.0) → 2.0; lb = ub = 2.0.
        let root_an = &analysis[0];
        assert!(
            root_an
                .lower_bound
                .map_or(false, |l| (l - 2.0).abs() < 1e-12)
        );
        assert!(
            root_an
                .upper_bound
                .map_or(false, |u| (u - 2.0).abs() < 1e-12)
        );
        assert!(root_an.is_nonnegative);
        assert!(root_an.is_positive);
    }

    #[test]
    fn analysis_pow_even_exponent_is_nonneg() {
        use crate::ast::convert::dag_to_ast;
        use crate::dag::builder::DagBuilder;
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let two = b.constant(2.0);
        let root = b.pow(x, two);
        let ast = dag_to_ast(b.arena(), root);
        let analysis = analyze(&ast);
        let root_an = &analysis[0];
        assert!(root_an.is_nonnegative, "x^2 must be non-negative");
        assert_eq!(root_an.lower_bound, Some(0.0));
        assert_eq!(root_an.pow_expansion, PowExpansion::IntPow(2));
    }

    #[test]
    fn analysis_pow_sqrt_expansion() {
        use crate::ast::convert::dag_to_ast;
        use crate::dag::builder::DagBuilder;
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let half = b.constant(0.5);
        let root = b.pow(x, half);
        let ast = dag_to_ast(b.arena(), root);
        let analysis = analyze(&ast);
        let root_an = &analysis[0];
        assert_eq!(root_an.pow_expansion, PowExpansion::Sqrt);
    }
}
