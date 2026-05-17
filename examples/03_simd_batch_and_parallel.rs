//! Example 03: SIMD Vectorization, Batch Processing, and Parallelization
//!
//! This example shows how to perform portable CPU SIMD vectorization detection,
//! execute high-speed vectorized batch floating-point math operations, and
//! split and evaluate symbolic expressions in parallel across multiple worker threads.
//!
//! Run with: `cargo run --example 03_simd_batch_and_parallel`

use rssn_advanced::simd::{batch_add, batch_mul, has_avx2};
use rssn_advanced::parallel::permission::SymbolPermissions;
use rssn_advanced::parallel::splitter::split_commutative_tree;
use rssn_advanced::parallel::solver::parallel_evaluate;
use rssn_advanced::parallel::simplify::ThreadLocalState;
use rssn_advanced::dag::builder::DagBuilder;
use rssn_advanced::dag::symbol::SymbolId;
use rssn_advanced::parser::parse_expression;

fn main() {
    println!("=== RSSN-Advanced Example 03: SIMD & Parallelism ===\n");

    // 1. Portable CPU SIMD Feature Detection
    println!("Detecting host processor capabilities...");
    let avx2_supported = has_avx2();
    println!("  AVX2 Vector Extension Supported : {}\n", avx2_supported);

    // 2. High-Speed Vectorized Batch Arithmetic
    println!("Running SIMD-Vectorized batch calculations on 10,000 float pairs...");
    let size = 10000;
    let lhs = vec![2.5f64; size];
    let rhs = vec![4.0f64; size];
    let mut results_add = vec![0.0f64; size];
    let mut results_mul = vec![0.0f64; size];

    let start_simd = std::time::Instant::now();
    
    // Bounds-checks are fully eliminated, enabling compiler loop auto-vectorization
    batch_add(&lhs, &rhs, &mut results_add);
    batch_mul(&lhs, &rhs, &mut results_mul);
    
    let duration_simd = start_simd.elapsed();
    println!("Batch arithmetic completed in {:?}!", duration_simd);
    println!("  Verification: Add Result[0] = {:.2} (Expected: 6.50)", results_add[0]);
    println!("  Verification: Mul Result[0] = {:.2} (Expected: 10.00)\n", results_mul[0]);

    // 3. Parallel Tree Splitting & Commutativity Permissions
    println!("Initializing commutativity permissions for symbolic variables...");
    let mut builder = DagBuilder::new();
    
    // Build addition tree: x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
    let root = parse_expression("x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7", &mut builder)
        .expect("Failed to parse expression");

    let permissions = SymbolPermissions::new();
    // Mark variables (SymbolId 0 to 7) as commutative to allow safe tree partition splitting
    for i in 0..8 {
        permissions.set_commutative(SymbolId::new(i), true);
    }

    println!("Splitting addition tree into 4 independent parallel chunks...");
    let chunks = split_commutative_tree(builder.arena(), root, &permissions, 4);
    println!("  Partitioned into {} chunks successfully.", chunks.len());

    // 4. Parallel Task Solver Evaluation
    // Define inputs: x0 = 1.0, x1 = 2.0, x2 = 3.0, ..., x7 = 8.0
    let variables = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    println!("Evaluating partitions in parallel...");
    
    let start_parallel = std::time::Instant::now();
    let total_sum = parallel_evaluate(builder.arena(), chunks, &variables);
    let duration_parallel = start_parallel.elapsed();

    println!("Parallel evaluation completed in {:?}!", duration_parallel);
    println!("  Total sum calculated           : {:.6}", total_sum);
    println!("  Expected mathematical sum      : 36.000000\n");

    // 5. Thread-Local Scratch Isolation
    println!("Demonstrating Thread-Local isolation state padding...");
    let state = ThreadLocalState::new();
    state.increment();
    state.increment();
    println!("  Thread-local counter state     : {}\n", state.get_count());

    println!("====================================================");
}
