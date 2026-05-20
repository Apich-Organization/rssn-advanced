//! High-performance Criterion benchmark suite for RSSN-Advanced.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rssn_advanced::dag::builder::DagBuilder;
use rssn_advanced::parser::parse_expression;
use rssn_advanced::simd::{batch_add, batch_mul};

#[cfg(feature = "jit")]
use rssn_advanced::ast::convert::dag_to_ast;

fn bench_parser_and_dag_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("dag_construction");
    
    let expr_str = "x + y * 2.0 + x * y + 3.5";

    group.bench_function("parse_and_build", |b| {
        b.iter(|| {
            let mut builder = DagBuilder::new();
            let _root = parse_expression(expr_str, &mut builder).unwrap();
        })
    });

    group.finish();
}

fn bench_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vectorization");

    for size in [1024, 4096, 16384].iter() {
        let lhs = vec![1.5f64; *size];
        let rhs = vec![2.5f64; *size];
        let mut result = vec![0.0f64; *size];

        group.bench_with_input(BenchmarkId::new("batch_add", size), size, |b, _| {
            b.iter(|| {
                let _ = batch_add(&lhs, &rhs, &mut result);
            })
        });

        group.bench_with_input(BenchmarkId::new("batch_mul", size), size, |b, _| {
            b.iter(|| {
                let _ = batch_mul(&lhs, &rhs, &mut result);
            })
        });
    }

    group.finish();
}

#[cfg(feature = "jit")]
fn bench_jit_compilation_and_exec(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_compiler");

    let mut builder = DagBuilder::new();
    let root = parse_expression("x0 + x1 + x2 + x3 + x4 + x5", &mut builder).unwrap();
    let ast = dag_to_ast(builder.arena(), root);

    group.bench_function("compile_ast", |b| {
        b.iter(|| {
            let mut compiler = rssn_advanced::jit::compiler::JitCompiler::new();
            let _compiled = compiler.compile(&ast).unwrap();
        })
    });

    // Compile once for execution benchmark
    let mut compiler = rssn_advanced::jit::compiler::JitCompiler::new();
    let compiled_fn = compiler.compile(&ast).unwrap();
    let vars = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    group.bench_function("execute_jit", |b| {
        b.iter(|| {
            let _val = compiled_fn(vars.as_ptr());
        })
    });

    group.finish();
}

#[cfg(not(feature = "jit"))]
fn bench_jit_compilation_and_exec(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_parser_and_dag_construction,
    bench_simd_vs_scalar,
    bench_jit_compilation_and_exec
);
criterion_main!(benches);
