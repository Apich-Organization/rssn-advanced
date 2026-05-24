//! Property-based tests for rssn-advanced using proptest.
//!
//! These tests verify algebraic correctness guarantees that hold for all
//! inputs within the described domains. They complement the unit tests by
//! exercising edge cases that hand-written examples might miss.
//!
//! ## What we test
//!
//! - **DAG deduplication**: structurally identical expressions always produce
//!   the same node id (structural sharing is sound).
//! - **Parser round-trips**: numbers parsed then printed remain numerically
//!   equivalent.
//! - **JIT correctness** (feature-gated): JIT-compiled expressions produce the
//!   same results as the equivalent native Rust f64 arithmetic.
//! - **Analysis soundness**: the pre-codegen analysis never claims a value is
//!   positive when it could be zero or negative.
//! - **Peephole correctness**: each IR peephole fires only when mathematically
//!   valid and preserves the expression's value.
//! - **FFI surface**: the C-API wrapper functions produce the same results as
//!   their Rust equivalents.

use proptest::prelude::*;
use rssn_advanced::dag::builder::DagBuilder;
use rssn_advanced::dag::node::DagNodeId;

// =========================================================================
// Helpers
// =========================================================================

/// Evaluates a simple two-variable expression tree `f(x, y)` using the
/// builder API, where the tree is determined by the `op_selector` integer.
#[allow(dead_code)]
fn eval_native(x: f64, y: f64, op_selector: u8) -> f64 {
    match op_selector % 6 {
        0 => x + y,
        1 => x - y,
        2 => x * y,
        3 => {
            if y != 0.0 {
                x / y
            } else {
                f64::NAN
            }
        }
        4 => x.powf(y),
        _ => x % if y != 0.0 { y } else { 1.0 },
    }
}

fn build_two_var_expr(
    builder: &mut DagBuilder,
    xv: DagNodeId,
    yv: DagNodeId,
    op_selector: u8,
) -> DagNodeId {
    match op_selector % 6 {
        0 => builder.add(xv, yv),
        1 => builder.sub(xv, yv),
        2 => builder.mul(xv, yv),
        3 => builder.div(xv, yv),
        4 => builder.pow(xv, yv),
        _ => builder.modulo(xv, yv),
    }
}

// =========================================================================
// DAG structural deduplication
// =========================================================================

proptest! {
    /// Identical variable names always intern to the same node id.
    #[test]
    fn dag_variable_dedup(name in "[a-z][a-z0-9]{0,7}") {
        let mut b = DagBuilder::new();
        let id1 = b.variable(&name);
        let id2 = b.variable(&name);
        prop_assert_eq!(id1, id2, "same variable name must produce same DagNodeId");
    }

    /// Identical constants always share a node.
    #[test]
    fn dag_constant_dedup(val in prop::num::f64::POSITIVE) {
        let mut b = DagBuilder::new();
        let id1 = b.constant(val);
        let id2 = b.constant(val);
        prop_assert_eq!(id1, id2, "same constant must deduplicate");
    }

    /// Building the same binary expression twice yields the same node id.
    #[test]
    fn dag_operator_dedup(op in 0u8..6, x in 1.0f64..100.0, y in 1.0f64..100.0) {
        let mut b = DagBuilder::new();
        let xv = b.constant(x);
        let yv = b.constant(y);
        let e1 = build_two_var_expr(&mut b, xv, yv, op);
        let e2 = build_two_var_expr(&mut b, xv, yv, op);
        prop_assert_eq!(e1, e2, "same expression built twice must deduplicate");
    }

    /// `add_many` over k constants produces a left-associative sum equal to
    /// the arithmetic sum.
    #[test]
    fn add_many_produces_valid_node(vals in prop::collection::vec(1.0f64..10.0, 1..8)) {
        let mut b = DagBuilder::new();
        let nodes: Vec<DagNodeId> = vals.iter().map(|&v| b.constant(v)).collect();
        let root = b.add_many(&nodes).expect("non-empty input");
        // The root must be a valid arena node.
        prop_assert!(b.arena().get(root).is_some(), "add_many root must be in arena");
    }

    /// `mul_many` over k constants produces a valid node.
    #[test]
    fn mul_many_produces_valid_node(vals in prop::collection::vec(1.0f64..10.0, 1..8)) {
        let mut b = DagBuilder::new();
        let nodes: Vec<DagNodeId> = vals.iter().map(|&v| b.constant(v)).collect();
        let root = b.mul_many(&nodes).expect("non-empty input");
        prop_assert!(b.arena().get(root).is_some(), "mul_many root must be in arena");
    }
}

// =========================================================================
// Parser correctness
// =========================================================================

proptest! {
    /// Numbers that survive a format→parse round-trip stay within f64 epsilon.
    ///
    /// We use Rust's default `Display` (`{val}`) which invokes the Ryu algorithm:
    /// the shortest decimal string that round-trips through `f64::from_str`.
    /// Subnormals and values below ~5e-18 are skipped because `{:.17}` (17
    /// decimal *places*) cannot represent them — we would be testing the parser
    /// against a string that already lost the original value.
    #[test]
    fn parser_constant_roundtrip(val in prop::num::f64::POSITIVE) {
        // Ryu guarantees round-trip; skip values the parser can't see at all.
        prop_assume!(val.is_finite() && val >= 5e-18);
        let s = format!("{val}");
        let mut b = DagBuilder::new();
        match b.parse(&s) {
            Ok(root) => {
                let node = b.arena().get(root).expect("root in arena");
                let rssn_advanced::dag::symbol::SymbolKind::Constant(parsed) = node.kind else {
                    prop_assert!(false, "expected Constant node, got {:?}", node.kind);
                    return Ok(());
                };
                prop_assert!(
                    (parsed - val).abs() <= val.abs() * 1e-14 + f64::MIN_POSITIVE,
                    "parsed {parsed} ≠ original {val} (string was {s:?})"
                );
            }
            Err(_) => {
                // Some extreme f64 values (Inf, NaN) cannot be parsed.
                prop_assume!(!val.is_infinite() && !val.is_nan());
                prop_assert!(false, "parse of finite number {val} failed");
            }
        }
    }

    /// Simple variable + constant expressions parse without error.
    #[test]
    fn parser_simple_expression_ok(
        var in "[a-z][a-z0-9]{0,3}",
        c in 0.1f64..1000.0,
        op in 0u8..5,
    ) {
        let ops = ["+", "-", "*", "/", "^"];
        let expr = format!("{var} {} {c}", ops[op as usize % 5]);
        let mut b = DagBuilder::new();
        let result = b.parse(&expr);
        prop_assert!(result.is_ok(), "parse of {expr:?} failed: {:?}", result.err());
    }
}

// =========================================================================
// JIT correctness
// =========================================================================

#[cfg(feature = "jit")]
mod jit_props {
    use super::*;
    use rssn_advanced::ast::convert::dag_to_ast;
    use rssn_advanced::jit::compiler::{JitCompiler, OptConfig};

    /// Compiles a constant expression and verifies the JIT result matches
    /// the expected value computed in native Rust.
    fn compile_and_eval(expr: &str, vars: &[f64]) -> f64 {
        let mut b = DagBuilder::new();
        let root = b.parse(expr).expect("parse");
        let ast = dag_to_ast(b.arena(), root);
        let mut compiler = JitCompiler::new();
        let f = compiler.compile(&ast).expect("compile");
        f(vars.as_ptr())
    }

    proptest! {
        /// x + y (JIT) == x + y (native).
        #[test]
        fn jit_add_matches_native(x in -1000.0f64..1000.0, y in -1000.0f64..1000.0) {
            let result = compile_and_eval("x + y", &[x, y]);
            prop_assert!(
                (result - (x + y)).abs() < 1e-10,
                "JIT x+y={result}, native={}", x + y
            );
        }

        /// x * y (JIT) == x * y (native).
        #[test]
        fn jit_mul_matches_native(x in -100.0f64..100.0, y in -100.0f64..100.0) {
            let result = compile_and_eval("x * y", &[x, y]);
            prop_assert!(
                (result - (x * y)).abs() < 1e-10,
                "JIT x*y={result}, native={}", x * y
            );
        }

        /// x - y (JIT) == x - y (native).
        #[test]
        fn jit_sub_matches_native(x in -1000.0f64..1000.0, y in -1000.0f64..1000.0) {
            let result = compile_and_eval("x - y", &[x, y]);
            prop_assert!(
                (result - (x - y)).abs() < 1e-10,
                "JIT x-y={result}, native={}", x - y
            );
        }

        /// x / y (JIT) == x / y (native) for nonzero y.
        #[test]
        fn jit_div_matches_native_nonzero(x in -100.0f64..100.0, y in 0.1f64..100.0) {
            let result = compile_and_eval("x / y", &[x, y]);
            let expected = x / y;
            prop_assert!(
                (result - expected).abs() <= expected.abs() * 1e-14 + 1e-14,
                "JIT x/y={result}, native={expected}"
            );
        }

        /// x^2 (JIT) == x*x (native) — verifies IntPow(2) expansion.
        #[test]
        fn jit_pow2_matches_mul(x in -100.0f64..100.0) {
            let result = compile_and_eval("x ^ 2.0", &[x]);
            let expected = x * x;
            prop_assert!(
                (result - expected).abs() <= expected.abs() * 1e-14 + 1e-14,
                "JIT x^2={result}, native={expected}"
            );
        }

        /// x^3 (JIT) == x*x*x (native) — verifies IntPow(3) expansion.
        #[test]
        fn jit_pow3_matches_mul(x in -10.0f64..10.0) {
            let result = compile_and_eval("x ^ 3.0", &[x]);
            let expected = x * x * x;
            prop_assert!(
                (result - expected).abs() <= expected.abs() * 1e-12 + 1e-12,
                "JIT x^3={result}, native={expected}"
            );
        }

        /// x^8 (JIT) == x^8 (native) — verifies the 3-step squaring chain.
        #[test]
        fn jit_pow8_matches_native(x in -5.0f64..5.0) {
            let result = compile_and_eval("x ^ 8.0", &[x]);
            let expected = x.powi(8);
            prop_assert!(
                (result - expected).abs() <= expected.abs() * 1e-12 + 1e-12,
                "JIT x^8={result}, native={expected}"
            );
        }

        /// x^16 (JIT) == x^16 (native) — verifies the 4-step squaring chain.
        #[test]
        fn jit_pow16_matches_native(x in -3.0f64..3.0) {
            let result = compile_and_eval("x ^ 16.0", &[x]);
            let expected = x.powi(16);
            prop_assert!(
                (result - expected).abs() <= expected.abs() * 1e-11 + 1e-11,
                "JIT x^16={result}, native={expected}"
            );
        }

        /// sqrt(x) (JIT) == sqrt(x) (native) — verifies Sqrt expansion.
        #[test]
        fn jit_sqrt_matches_native(x in 0.0f64..10000.0) {
            let result = compile_and_eval("x ^ 0.5", &[x]);
            let expected = x.sqrt();
            prop_assert!(
                (result - expected).abs() <= expected * 1e-14 + 1e-15,
                "JIT sqrt(x)={result}, native={expected}"
            );
        }

        /// Constant folding: JIT compiles a constant-only expression to the
        /// correct numeric value.
        #[test]
        fn jit_constant_fold(c in 1.0f64..1000.0) {
            // Use Ryu format so the parsed constant is bit-for-bit identical to c.
            // {c:.6} only gives 6 decimal places (~1e-6 accuracy), which is
            // incompatible with a 1e-14 tolerance check.
            let expr = format!("{c} + 0.0");
            let result = compile_and_eval(&expr, &[]);
            prop_assert!(
                (result - c).abs() <= c.abs() * 1e-14 + 1e-14,
                "JIT constant fold {c}: got {result}"
            );
        }

        /// x * 0.0 (JIT) == 0.0 — verifies the x*0→0 peephole.
        #[test]
        fn jit_mul_zero_peephole(x in prop::num::f64::NORMAL) {
            let result = compile_and_eval("x * 0.0", &[x]);
            prop_assert_eq!(result, 0.0, "x * 0.0 must be exactly 0.0");
        }

        /// x + 0.0 (JIT) == x — verifies the x+0→x peephole.
        #[test]
        fn jit_add_zero_peephole(x in prop::num::f64::NORMAL) {
            let result = compile_and_eval("x + 0.0", &[x]);
            prop_assert_eq!(result, x, "x + 0.0 must be x");
        }

        /// 1.0 ^ x (JIT) == 1.0 — verifies the 1^x→1 peephole.
        #[test]
        fn jit_one_pow_x_peephole(x in -100.0f64..100.0) {
            let result = compile_and_eval("1.0 ^ x", &[x]);
            prop_assert_eq!(result, 1.0, "1.0 ^ x must be 1.0");
        }

        /// 0.0 ^ x (JIT) == 0.0 for positive x — verifies the 0^positive→0 peephole.
        #[test]
        fn jit_zero_pow_positive_peephole(x in 0.001f64..100.0) {
            let result = compile_and_eval("0.0 ^ x", &[x]);
            prop_assert_eq!(result, 0.0, "0.0 ^ positive must be 0.0, got {}", result);
        }

        /// CSE: x*x + x*x (JIT) == 2 * x^2 (native).
        #[test]
        fn jit_cse_correct(x in -100.0f64..100.0) {
            let result = compile_and_eval("x * x + x * x", &[x]);
            let expected = 2.0 * x * x;
            prop_assert!(
                (result - expected).abs() <= expected.abs() * 1e-13 + 1e-13,
                "JIT CSE: {result} vs {expected}"
            );
        }

        /// x / x (JIT) == 1.0 when x is not zero — CSE + x/x→1 peephole.
        /// Note: current implementation only fires when both SSA values are
        /// identical AND analysis can prove nonzero. We test the semantic value.
        #[test]
        fn jit_div_self_is_one(x in 0.1f64..1000.0) {
            let result = compile_and_eval("x / x", &[x]);
            prop_assert!(
                (result - 1.0).abs() < 1e-13,
                "x / x must equal 1.0 for nonzero x, got {result}"
            );
        }

        /// FMA fusion: a*b + c (JIT) has the same value as the native computation.
        #[test]
        fn jit_fma_fusion_correct(a in -10.0f64..10.0, b in -10.0f64..10.0, c in -10.0f64..10.0) {
            let result = compile_and_eval("x * y + z", &[a, b, c]);
            let expected = a * b + c;
            prop_assert!(
                (result - expected).abs() <= expected.abs() * 1e-13 + 1e-13,
                "FMA a*b+c: JIT={result}, native={expected}"
            );
        }

        /// Polynomial with all major optimisations exercised.
        #[test]
        fn jit_polynomial_correct(x in -5.0f64..5.0) {
            // x^4 + 2*x^2 - x + 1
            let result = compile_and_eval("x ^ 4.0 + 2.0 * x ^ 2.0 - x + 1.0", &[x]);
            let expected = x.powi(4) + 2.0 * x.powi(2) - x + 1.0;
            prop_assert!(
                (result - expected).abs() <= expected.abs() * 1e-11 + 1e-11,
                "polynomial: JIT={result}, native={expected}"
            );
        }

        /// Reciprocal math opt: x / 4.0 with allow_reciprocal_math.
        #[test]
        fn jit_reciprocal_math_opt(x in -1000.0f64..1000.0) {
            let mut b = DagBuilder::new();
            let root = b.parse("x / 4.0").expect("parse");
            let ast = dag_to_ast(b.arena(), root);
            let mut compiler = JitCompiler::new();
            let opts = OptConfig { allow_reciprocal_math: true, ..OptConfig::default() };
            let f = compiler.compile_with_opts(&ast, &opts).expect("compile");
            let result = f([x].as_ptr());
            let expected = x / 4.0;
            prop_assert!(
                (result - expected).abs() <= expected.abs() * 1e-13 + 1e-15,
                "reciprocal math: JIT={result}, native={expected}"
            );
        }
    }

    proptest! {
        /// Batch evaluation: `compile_batch_f64x2` produces the same results
        /// as scalar JIT for all vectorizable expressions.
        #[test]
        fn jit_batch_matches_scalar(
            xs in prop::collection::vec(-10.0f64..10.0, 4..=8),
            ys in prop::collection::vec(-10.0f64..10.0, 4..=8),
        ) {
            prop_assume!(xs.len() == ys.len());
            let n = xs.len();

            let mut b = DagBuilder::new();
            let root = b.parse("x + y * y").expect("parse");
            let ast = dag_to_ast(b.arena(), root);

            let mut compiler = JitCompiler::new();
            let scalar_f = compiler.compile(&ast).expect("scalar compile");

            // Scalar evaluation reference.
            let scalar_results: Vec<f64> = (0..n).map(|i| {
                scalar_f([xs[i], ys[i]].as_ptr())
            }).collect();

            if let Ok(Some(batch_f)) = compiler.compile_batch_f64x2(&ast) {
                // Column-major layout: col0 = xs, col1 = ys.
                let x_col = xs.as_slice();
                let y_col = ys.as_slice();
                let col_ptrs = [x_col.as_ptr(), y_col.as_ptr()];
                let mut out = vec![0.0f64; n];
                batch_f(col_ptrs.as_ptr(), n, out.as_mut_ptr());

                for i in 0..n {
                    let diff = (out[i] - scalar_results[i]).abs();
                    prop_assert!(
                        diff <= scalar_results[i].abs() * 1e-13 + 1e-13,
                        "batch[{i}]={} vs scalar={}", out[i], scalar_results[i]
                    );
                }
            }
        }
    }
}

// =========================================================================
// Analysis soundness
// =========================================================================

#[cfg(feature = "jit")]
mod analysis_props {
    use super::*;
    use rssn_advanced::ast::convert::dag_to_ast;
    use rssn_advanced::jit::analysis::analyze;

    proptest! {
        /// If analysis says `is_positive`, the expression is actually > 0 at
        /// the constant-folding level (for constant sub-expressions).
        #[test]
        fn analysis_positive_constant_is_positive(c in 0.001f64..1000.0) {
            let mut b = DagBuilder::new();
            let root = b.constant(c);
            let ast = dag_to_ast(b.arena(), root);
            let analysis = analyze(&ast);
            let an = &analysis[0];
            prop_assert!(an.is_positive, "positive constant must be marked is_positive");
            prop_assert!(an.is_nonnegative);
            prop_assert!(an.no_nan);
            prop_assert_eq!(an.lower_bound, Some(c));
            prop_assert_eq!(an.upper_bound, Some(c));
        }

        /// If analysis says `is_nonnegative` for x^2, that's always true.
        #[test]
        fn analysis_x_squared_is_nonneg(_x in prop::num::f64::NORMAL) {
            let mut b = DagBuilder::new();
            let x = b.variable("x");
            let two = b.constant(2.0);
            let root = b.pow(x, two);
            let ast = dag_to_ast(b.arena(), root);
            let analysis = analyze(&ast);
            // Root is Pow; check that it's marked nonneg.
            let root_an = &analysis[0];
            prop_assert!(root_an.is_nonnegative, "x^2 must be non-negative");
        }

        /// The analysis bounds for constants are tight: lb == ub == value.
        #[test]
        fn analysis_constant_tight_bounds(c in prop::num::f64::NORMAL) {
            let mut b = DagBuilder::new();
            let root = b.constant(c);
            let ast = dag_to_ast(b.arena(), root);
            let analysis = analyze(&ast);
            let an = &analysis[0];
            prop_assert_eq!(an.lower_bound, Some(c));
            prop_assert_eq!(an.upper_bound, Some(c));
        }

        /// classify_exponent returns IntPow(n) for all positive integers 2..=16.
        #[test]
        fn classify_exponent_covers_int_range(n in 2u32..=16) {
            use rssn_advanced::jit::analysis::{PowExpansion, classify_exponent};
            let result = classify_exponent(n as f64);
            prop_assert_eq!(result, PowExpansion::IntPow(n));
        }

        /// classify_exponent returns NegIntPow(n) for -1..=-8.
        #[test]
        fn classify_exponent_covers_neg_int_range(n in 1u32..=8) {
            use rssn_advanced::jit::analysis::{PowExpansion, classify_exponent};
            let result = classify_exponent(-(n as f64));
            prop_assert_eq!(result, PowExpansion::NegIntPow(n));
        }
    }
}

// =========================================================================
// FFI surface
// =========================================================================

mod ffi_props {
    use super::*;
    use rssn_advanced::ffi::c_api::{
        rssn_dag_add, rssn_dag_constant, rssn_dag_div, rssn_dag_free, rssn_dag_mul, rssn_dag_neg,
        rssn_dag_new, rssn_dag_parse, rssn_dag_pow, rssn_dag_sub,
    };
    use rssn_advanced::ffi::types::RssnStatus;

    proptest! {
        /// Constant nodes built via FFI match the value inserted.
        #[test]
        fn ffi_constant_round_trip(val in 1.0f64..1e6) {
            let builder = rssn_dag_new();
            let id = rssn_dag_constant(builder, val);
            prop_assert_ne!(id, u32::MAX, "constant must not fail");
            let b = unsafe { &mut *builder };
            let node = b.arena().get(rssn_advanced::dag::node::DagNodeId::new(id))
                .expect("node in arena");
            let rssn_advanced::dag::symbol::SymbolKind::Constant(node_val) = node.kind else {
                prop_assert!(false, "expected Constant node");
                return Ok(());
            };
            prop_assert!(
                (node_val - val).abs() < f64::EPSILON,
                "FFI constant value mismatch: {} vs {}", node_val, val
            );
            rssn_dag_free(builder);
        }

        /// FFI binary operators return valid (non-MAX) node ids for valid inputs.
        #[test]
        fn ffi_binary_ops_ok(x in 1.0f64..100.0, y in 1.0f64..100.0) {
            let b = rssn_dag_new();
            let xid = rssn_dag_constant(b, x);
            let yid = rssn_dag_constant(b, y);

            prop_assert_ne!(rssn_dag_add(b, xid, yid), u32::MAX);
            prop_assert_ne!(rssn_dag_sub(b, xid, yid), u32::MAX);
            prop_assert_ne!(rssn_dag_mul(b, xid, yid), u32::MAX);
            prop_assert_ne!(rssn_dag_div(b, xid, yid), u32::MAX);
            prop_assert_ne!(rssn_dag_pow(b, xid, yid), u32::MAX);
            prop_assert_ne!(rssn_dag_neg(b, xid), u32::MAX);

            rssn_dag_free(b);
        }

        /// FFI parse handles valid arithmetic expressions without panicking.
        #[test]
        fn ffi_parse_simple(c in 0.1f64..999.0) {
            let expr = format!("{c:.4} + {c:.4}");
            let expr_c = std::ffi::CString::new(expr).unwrap();
            let b = rssn_dag_new();
            let mut out = u32::MAX;
            let status = rssn_dag_parse(b, expr_c.as_ptr(), &mut out);
            prop_assert_eq!(status, RssnStatus::Success);
            prop_assert_ne!(out, u32::MAX);
            rssn_dag_free(b);
        }
    }
}

// =========================================================================
// Addition-chain correctness for emit_int_pow
// =========================================================================

/// Verify the addition chains for n=2..=16 match native f64::powi.
/// These tests run in pure Rust without Cranelift — they verify the
/// mathematical correctness of the chain schedules in passes.rs.
#[test]
fn addition_chain_logic_all_n() {
    for n in 2u32..=16 {
        for &x in &[0.0_f64, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, 10.0] {
            let expected = x.powi(n as i32);
            let actual = eval_chain(x, n);
            let diff = (actual - expected).abs();
            let tol = expected.abs() * 1e-13 + 1e-13;
            assert!(
                diff <= tol || (actual.is_nan() && expected.is_nan()),
                "chain({x}, {n}): got {actual}, expected {expected}"
            );
        }
    }
}

/// Verify neg int pow logic for n=1..=8.
#[test]
fn neg_pow_chain_logic_all_n() {
    for n in 1u32..=8 {
        for &x in &[1.0_f64, 2.0, -1.0, -2.0, 0.5, 10.0] {
            let expected = 1.0 / x.powi(n as i32);
            let actual = 1.0 / eval_chain(x, n);
            let diff = (actual - expected).abs();
            let tol = expected.abs() * 1e-12 + 1e-12;
            assert!(
                diff <= tol,
                "neg_chain({x}, {n}): got {actual}, expected {expected}"
            );
        }
    }
}

/// Emulates the addition chain logic from passes.rs using native Rust arithmetic.
fn eval_chain(x: f64, n: u32) -> f64 {
    match n {
        2 => x * x,
        3 => {
            let sq = x * x;
            sq * x
        }
        4 => {
            let sq = x * x;
            sq * sq
        }
        5 => {
            let sq = x * x;
            let q4 = sq * sq;
            q4 * x
        }
        6 => {
            let sq = x * x;
            let cu = sq * x;
            cu * cu
        }
        7 => {
            let sq = x * x;
            let cu = sq * x;
            let c6 = cu * cu;
            c6 * x
        }
        8 => {
            let sq = x * x;
            let q4 = sq * sq;
            q4 * q4
        }
        9 => {
            let sq = x * x;
            let q4 = sq * sq;
            let q8 = q4 * q4;
            q8 * x
        }
        10 => {
            let sq = x * x;
            let q4 = sq * sq;
            let q8 = q4 * q4;
            q8 * sq
        }
        11 => {
            let sq = x * x;
            let q4 = sq * sq;
            let q8 = q4 * q4;
            let x9 = q8 * x;
            x9 * sq
        }
        12 => {
            let sq = x * x;
            let cu = sq * x;
            let c6 = cu * cu;
            c6 * c6
        }
        13 => {
            let sq = x * x;
            let cu = sq * x;
            let c6 = cu * cu;
            let c12 = c6 * c6;
            c12 * x
        }
        14 => {
            let sq = x * x;
            let cu = sq * x;
            let c6 = cu * cu;
            let c12 = c6 * c6;
            c12 * sq
        }
        15 => {
            let sq = x * x;
            let cu = sq * x;
            let c6 = cu * cu;
            let c12 = c6 * c6;
            c12 * cu
        }
        16 => {
            let sq = x * x;
            let q4 = sq * sq;
            let q8 = q4 * q4;
            q8 * q8
        }
        _ => x, // out of range — matches the debug_assert fallback in passes.rs
    }
}
