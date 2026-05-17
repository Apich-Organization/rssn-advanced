//! Example 02: AST Projection, JIT Compilation, and Execution
//!
//! This example shows how to take a symbolic expression DAG, project it into
//! the high-locality cache-efficient `AstProjection` memory layout, compile it
//! to native binary machine instructions using Cranelift JIT, and execute the native
//! function at hardware speed.
//!
//! Run with: `cargo run --example 02_jit_compiler_and_evaluation --features cranelift-jit,cranelift-frontend,cranelift-native,cranelift-codegen,cranelift-module`

use rssn_advanced::dag::builder::DagBuilder;
use rssn_advanced::parser::parse_expression;
use rssn_advanced::ast::convert::dag_to_ast;

#[cfg(feature = "cranelift-jit")]
use rssn_advanced::jit::compiler::JitCompiler;

fn main() {
    println!("=== RSSN-Advanced Example 02: JIT Compiler & Evaluation ===\n");

    // 1. Construct the symbolic math expression
    let mut builder = DagBuilder::new();
    let expr_str = "x * 2.5 + y ^ 2.0";
    
    println!("Parsing formula: \"{}\"", expr_str);
    let root_id = parse_expression(expr_str, &mut builder)
        .expect("Failed to parse expression");

    // 2. Project the DAG subgraph into a cache-efficient AstProjection
    println!("Projecting DAG subgraph to flat AST layout...");
    let ast = dag_to_ast(builder.arena(), root_id);
    println!("  AST node count in flat layout: {}\n", ast.nodes.len());

    // 3. Dynamic JIT Compilation
    #[cfg(feature = "cranelift-jit")]
    {
        println!("Compiling AST to optimized machine instructions via Cranelift JIT...");
        let mut compiler = JitCompiler::new();
        
        let start_compile = std::time::Instant::now();
        let compiled_fn = compiler.compile(&ast)
            .expect("JIT compilation failed");
        let compile_duration = start_compile.elapsed();
        
        println!("Compilation succeeded in {:?}!", compile_duration);
        println!("Dynamic function bound to native machine pointer.\n");

        // 4. Bare-Metal Execution
        // Inputs are passed as a raw float slice matching the variable order in builder (x = 0, y = 1)
        // Evaluation: x * 2.5 + y ^ 2.0  =>  4.0 * 2.5 + 3.0 ^ 2.0 = 10.0 + 9.0 = 19.0
        let variables = vec![4.0, 3.0];
        println!("Executing native function with inputs x = 4.0, y = 3.0...");
        
        let start_exec = std::time::Instant::now();
        let result = compiled_fn(variables.as_ptr());
        let exec_duration = start_exec.elapsed();

        println!("Execution completed!");
        println!("  Result         : {:.6}", result);
        println!("  Time Taken     : {:?}", exec_duration);
        println!("  Calculated expected value: 19.0\n");
    }

    #[cfg(not(feature = "cranelift-jit"))]
    {
        println!("[!] Cranelift JIT features are not enabled. Compilation skipped.");
        println!("    Please run with the following command to test dynamic JIT:");
        println!("    cargo run --example 02_jit_compiler_and_evaluation --features cranelift-jit,cranelift-frontend,cranelift-native,cranelift-codegen,cranelift-module\n");
    }

    println!("===========================================================");
}
