//! Criterion benchmark suite for rssn-advanced.
//!
//! ## What is measured
//!
//! - **DAG construction**: expression parsing and hash-consed node allocation.
//! - **SIMD batch arithmetic**: throughput of `batch_add` / `batch_mul` on
//!   various slice sizes.
//! - **JIT compilation**: time to produce native code from an AST projection
//!   (includes Cranelift IR emission, optimisation, and link-step).
//! - **JIT execution**: steady-state throughput of a compiled function with
//!   no recompilation overhead.
//! - **Batch JIT evaluation**: F64X2 vectorised batch path.
//! - **Heuristic simplification**: wall-clock time for the iterative rewriter.
//!
//! ## Comparison baseline
//!
//! Each JIT benchmark is paired with a **native Rust** baseline that performs
//! the equivalent computation using plain `f64` arithmetic with `#[inline(never)]`
//! to prevent the compiler from optimising away the measurement. This gives a
//! meaningful lower bound: the native baseline is what a hand-optimised
//! implementation would produce.
//!
//! A direct comparison with the `symbolica` crate is not included in this
//! benchmark suite because symbolica requires a commercial licence key for
//! use, which makes automated benchmarking impractical in an open-source CI
//! environment. Readers who hold a symbolica licence can run equivalent
//! benchmarks by following symbolica's own documentation.

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rssn_advanced::dag::builder::DagBuilder;
use rssn_advanced::heuristic::{HeuristicConfig, HeuristicEngine, SearchStrategy};
use rssn_advanced::parser::parse_expression;
use rssn_advanced::simd::{batch_add, batch_mul};

#[cfg(feature = "jit")]
use rssn_advanced::ast::convert::dag_to_ast;
#[cfg(feature = "jit")]
use rssn_advanced::jit::compiler::{JitCompiler, OptConfig};

// =========================================================================
// Native baselines — `#[inline(never)]` prevents over-optimisation.
// =========================================================================

#[inline(never)]
fn native_add_n_vars(vars: &[f64]) -> f64 {
    vars.iter().copied().sum()
}

#[inline(never)]
fn native_polynomial(x: f64) -> f64 {
    x.powi(4) + 2.0 * x.powi(2) - x + 1.0
}

#[inline(never)]
fn native_fma_chain(a: f64, b: f64, c: f64) -> f64 {
    a * b + c
}

// =========================================================================
// 1. DAG construction benchmarks
// =========================================================================

fn bench_dag_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("dag_construction");

    // 1a. Parse + build a short expression from a string.
    let expr_short = "x + y * 2.0 + x * y + 3.5";
    group.bench_function("parse_short_expr", |b| {
        b.iter(|| {
            let mut builder = DagBuilder::new();
            let _root = parse_expression(expr_short, &mut builder).unwrap();
        })
    });

    // 1b. Parse a longer polynomial.
    let expr_long = "a^4 + 4*a^3*b + 6*a^2*b^2 + 4*a*b^3 + b^4";
    group.bench_function("parse_binomial4", |b| {
        b.iter(|| {
            let mut builder = DagBuilder::new();
            let _root = parse_expression(expr_long, &mut builder).unwrap();
        })
    });

    // 1c. Programmatic construction — avoids parser overhead.
    group.bench_function("programmatic_10_node", |b| {
        b.iter(|| {
            let mut builder = DagBuilder::new();
            let x = builder.variable("x");
            let y = builder.variable("y");
            let c1 = builder.constant(2.0);
            let c2 = builder.constant(3.5);
            let t1 = builder.mul(x, c1);
            let t2 = builder.add(x, y);
            let t3 = builder.mul(t2, c2);
            let _root = builder.add(t1, t3);
        })
    });

    // 1d. Structural deduplication: inserting the same sub-expression 100 times.
    group.bench_function("dedup_100_repeated_inserts", |b| {
        b.iter(|| {
            let mut builder = DagBuilder::new();
            let x = builder.variable("x");
            let two = builder.constant(2.0);
            for _ in 0..100 {
                let _ = builder.mul(x, two);
            }
        })
    });

    group.finish();
}

// =========================================================================
// 2. SIMD batch arithmetic
// =========================================================================

fn bench_simd_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_batch_arithmetic");
    group.warm_up_time(std::time::Duration::from_millis(500));

    for &size in &[256usize, 1024, 4096, 16_384, 65_536] {
        let lhs = vec![1.5_f64; size];
        let rhs = vec![2.5_f64; size];
        let mut result = vec![0.0_f64; size];

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("batch_add", size), &size, |b, _| {
            b.iter(|| {
                let _ = batch_add(&lhs, &rhs, &mut result);
            })
        });

        group.bench_with_input(BenchmarkId::new("batch_mul", size), &size, |b, _| {
            b.iter(|| {
                let _ = batch_mul(&lhs, &rhs, &mut result);
            })
        });

        // Native scalar baseline for comparison.
        group.bench_with_input(
            BenchmarkId::new("scalar_add_baseline", size),
            &size,
            |b, _| {
                b.iter(|| {
                    for i in 0..size {
                        // SAFETY: index is in bounds.
                        result[i] = lhs[i] + rhs[i];
                    }
                })
            },
        );
    }

    group.finish();
}

// =========================================================================
// 3. JIT compilation benchmarks
// =========================================================================

#[cfg(feature = "jit")]
fn bench_jit_compile(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_compile");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(5));

    // 3a. Linear sum of 6 variables.
    {
        let mut builder = DagBuilder::new();
        let root = parse_expression("x0 + x1 + x2 + x3 + x4 + x5", &mut builder).unwrap();
        let ast = dag_to_ast(builder.arena(), root);

        group.bench_function("compile_linear_6var", |b| {
            b.iter(|| {
                let mut compiler = JitCompiler::new();
                let _f = compiler.compile(&ast).unwrap();
            })
        });
    }

    // 3b. Polynomial x^4 + 2*x^2 - x + 1 (exercises IntPow expansion).
    {
        let mut builder = DagBuilder::new();
        let root = parse_expression("x ^ 4.0 + 2.0 * x ^ 2.0 - x + 1.0", &mut builder).unwrap();
        let ast = dag_to_ast(builder.arena(), root);

        group.bench_function("compile_polynomial_degree4", |b| {
            b.iter(|| {
                let mut compiler = JitCompiler::new();
                let _f = compiler.compile(&ast).unwrap();
            })
        });
    }

    // 3c. Expression with sqrt.
    {
        let mut builder = DagBuilder::new();
        let root = parse_expression("x ^ 0.5 + y * y", &mut builder).unwrap();
        let ast = dag_to_ast(builder.arena(), root);

        group.bench_function("compile_sqrt_plus_square", |b| {
            b.iter(|| {
                let mut compiler = JitCompiler::new();
                let _f = compiler.compile(&ast).unwrap();
            })
        });
    }

    // 3d. Expression with CSE (x*x appears twice).
    {
        let mut builder = DagBuilder::new();
        let root = parse_expression("x * x + x * x + y", &mut builder).unwrap();
        let ast = dag_to_ast(builder.arena(), root);

        group.bench_function("compile_with_cse", |b| {
            b.iter_batched(
                JitCompiler::new,
                |mut compiler| compiler.compile(&ast).unwrap(),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

#[cfg(not(feature = "jit"))]
fn bench_jit_compile(_c: &mut Criterion) {}

// =========================================================================
// 4. JIT execution benchmarks (steady-state throughput)
// =========================================================================

#[cfg(feature = "jit")]
fn bench_jit_exec(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_exec");
    group.warm_up_time(std::time::Duration::from_millis(300));

    // 4a. Linear sum: JIT vs native.
    {
        let mut builder = DagBuilder::new();
        let root = parse_expression("x0 + x1 + x2 + x3 + x4 + x5", &mut builder).unwrap();
        let ast = dag_to_ast(builder.arena(), root);
        let mut compiler = JitCompiler::new();
        let compiled = compiler.compile(&ast).unwrap();
        let vars = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];

        group.bench_function("exec_linear_6var_jit", |b| {
            b.iter(|| compiled(vars.as_ptr()))
        });
        group.bench_function("exec_linear_6var_native", |b| {
            b.iter(|| native_add_n_vars(&vars))
        });
    }

    // 4b. Polynomial degree-4: JIT vs native.
    {
        let mut builder = DagBuilder::new();
        let root = parse_expression("x ^ 4.0 + 2.0 * x ^ 2.0 - x + 1.0", &mut builder).unwrap();
        let ast = dag_to_ast(builder.arena(), root);
        let mut compiler = JitCompiler::new();
        let compiled = compiler.compile(&ast).unwrap();
        let vars = [2.5_f64];

        group.bench_function("exec_polynomial4_jit", |b| {
            b.iter(|| compiled(vars.as_ptr()))
        });
        group.bench_function("exec_polynomial4_native", |b| {
            b.iter(|| native_polynomial(vars[0]))
        });
    }

    // 4c. FMA chain: JIT vs native.
    {
        let mut builder = DagBuilder::new();
        let root = parse_expression("x * y + z", &mut builder).unwrap();
        let ast = dag_to_ast(builder.arena(), root);
        let mut compiler = JitCompiler::new();
        let compiled = compiler.compile(&ast).unwrap();
        let vars = [3.0_f64, 4.0, 5.0];

        group.bench_function("exec_fma_jit", |b| b.iter(|| compiled(vars.as_ptr())));
        group.bench_function("exec_fma_native", |b| {
            b.iter(|| native_fma_chain(vars[0], vars[1], vars[2]))
        });
    }

    // 4d. x^16 expansion: 4-step squaring chain JIT vs native powi.
    {
        let mut builder = DagBuilder::new();
        let root = parse_expression("x ^ 16.0", &mut builder).unwrap();
        let ast = dag_to_ast(builder.arena(), root);
        let mut compiler = JitCompiler::new();
        let compiled = compiler
            .compile_with_opts(
                &ast,
                &OptConfig {
                    max_int_pow: 16,
                    ..OptConfig::default()
                },
            )
            .unwrap();
        let vars = [1.5_f64];

        group.bench_function("exec_pow16_jit", |b| b.iter(|| compiled(vars.as_ptr())));
        group.bench_function("exec_pow16_native_powi", |b| b.iter(|| vars[0].powi(16)));
    }

    // 4e. Reciprocal 1/x: NegIntPow(1) path.
    {
        let mut builder = DagBuilder::new();
        let root = parse_expression("x ^ -1.0", &mut builder).unwrap();
        let ast = dag_to_ast(builder.arena(), root);
        let mut compiler = JitCompiler::new();
        let compiled = compiler.compile(&ast).unwrap();
        let vars = [3.14_f64];

        group.bench_function("exec_recip_jit", |b| b.iter(|| compiled(vars.as_ptr())));
        group.bench_function("exec_recip_native", |b| b.iter(|| 1.0 / vars[0]));
    }

    group.finish();
}

#[cfg(not(feature = "jit"))]
fn bench_jit_exec(_c: &mut Criterion) {}

// =========================================================================
// 5. Batch (F64X2) JIT evaluation
// =========================================================================

#[cfg(feature = "jit")]
fn bench_jit_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_batch_f64x2");
    group.warm_up_time(std::time::Duration::from_millis(400));

    let mut builder = DagBuilder::new();
    let root = parse_expression("x + y * y", &mut builder).unwrap();
    let ast = dag_to_ast(builder.arena(), root);

    let mut compiler = JitCompiler::new();
    let scalar_f = compiler.compile(&ast).unwrap();
    let batch_f = compiler.compile_batch_f64x2(&ast).unwrap();

    for &n in &[64usize, 256, 1024, 4096] {
        let xs: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let ys: Vec<f64> = (0..n).map(|i| (i as f64) * 0.02 + 1.0).collect();
        let col_ptrs = [xs.as_ptr(), ys.as_ptr()];
        let mut out = vec![0.0_f64; n];

        group.throughput(Throughput::Elements(n as u64));

        if let Some(bf) = batch_f {
            group.bench_with_input(BenchmarkId::new("batch_f64x2", n), &n, |b, _| {
                b.iter(|| {
                    bf(col_ptrs.as_ptr(), n, out.as_mut_ptr());
                })
            });
        }

        group.bench_with_input(BenchmarkId::new("scalar_loop_baseline", n), &n, |b, _| {
            b.iter(|| {
                for i in 0..n {
                    out[i] = scalar_f([xs[i], ys[i]].as_ptr());
                }
            })
        });
    }

    group.finish();
}

#[cfg(not(feature = "jit"))]
fn bench_jit_batch(_c: &mut Criterion) {}

// =========================================================================
// 6. Heuristic simplification
// =========================================================================

fn bench_heuristic_simplify(c: &mut Criterion) {
    let mut group = c.benchmark_group("heuristic_simplify");

    // 6a. Simple algebraic simplification: x + 0.
    {
        let expr = "x + 0.0";
        group.bench_function("simplify_x_plus_zero", |b| {
            b.iter(|| {
                let mut builder = DagBuilder::new();
                let root = parse_expression(expr, &mut builder).unwrap();
                let cfg = HeuristicConfig::default();
                let mut engine = HeuristicEngine::new(cfg, SearchStrategy::Greedy);
                let _simplified = engine.simplify(&mut builder, root);
            })
        });
    }

    // 6b. Polynomial.
    {
        let expr = "x^2 + 2*x + 1";
        group.bench_function("simplify_degree2_polynomial", |b| {
            b.iter(|| {
                let mut builder = DagBuilder::new();
                let root = parse_expression(expr, &mut builder).unwrap();
                let cfg = HeuristicConfig::default();
                let mut engine = HeuristicEngine::new(cfg, SearchStrategy::Greedy);
                let _simplified = engine.simplify(&mut builder, root);
            })
        });
    }

    // 6c. Beam search for a more complex expression.
    {
        let expr = "a*b + b*c + c*a + a^2 + b^2";
        group.bench_function("simplify_beam_search_quadratic", |b| {
            b.iter(|| {
                let mut builder = DagBuilder::new();
                let root = parse_expression(expr, &mut builder).unwrap();
                let cfg = HeuristicConfig::default().max_depth(5);
                let mut engine = HeuristicEngine::new(cfg, SearchStrategy::Greedy);
                let _simplified = engine.simplify(&mut builder, root);
            })
        });
    }

    group.finish();
}

// =========================================================================
// 7. OptConfig variations
// =========================================================================

#[cfg(feature = "jit")]
fn bench_opt_config_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_opt_config_comparison");

    let mut builder = DagBuilder::new();
    let root = parse_expression("x ^ 4.0 + x ^ 8.0 + x ^ 16.0", &mut builder).unwrap();
    let ast = dag_to_ast(builder.arena(), root);

    // Default config (max_int_pow=16).
    group.bench_function("compile_pow_sum_default", |b| {
        b.iter_batched(
            JitCompiler::new,
            |mut compiler| {
                compiler
                    .compile_with_opts(&ast, &OptConfig::default())
                    .unwrap()
            },
            BatchSize::SmallInput,
        )
    });

    // Conservative config (max_int_pow=4, no CSE, no guard elision).
    let conservative = OptConfig {
        max_int_pow: 4,
        expand_sqrt: false,
        allow_reciprocal_math: false,
        elide_nan_guard: false,
        enable_cse: false,
    };
    group.bench_function("compile_pow_sum_conservative", |b| {
        b.iter_batched(
            JitCompiler::new,
            |mut compiler| compiler.compile_with_opts(&ast, &conservative).unwrap(),
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

#[cfg(not(feature = "jit"))]
fn bench_opt_config_comparison(_c: &mut Criterion) {}

// =========================================================================
// Criterion group registration
// =========================================================================

criterion_group!(
    benches,
    bench_dag_construction,
    bench_simd_batch,
    bench_jit_compile,
    bench_jit_exec,
    bench_jit_batch,
    bench_heuristic_simplify,
    bench_opt_config_comparison,
);
criterion_main!(benches);
