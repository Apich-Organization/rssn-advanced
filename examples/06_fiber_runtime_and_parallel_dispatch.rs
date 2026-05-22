//! Example 06: Fiber Runtime and Parallel Task Dispatch
//!
//! This example demonstrates how rssn-advanced manages concurrency through its
//! `dtact`-backed fiber runtime rather than heavyweight OS threads. It shows
//! one-shot runtime initialization, fire-and-forget task spawning with
//! explicit join, and high-throughput fan-out/fan-in via `parallel_for_each`.
//!
//! Run with: `cargo run --example 06_fiber_runtime_and_parallel_dispatch`

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rssn_advanced::runtime::{ensure_runtime, parallel_for_each, spawn_task, join};
use rssn_advanced::dag::builder::DagBuilder;
use rssn_advanced::parser::parse_expression;
use rssn_advanced::heuristic::{HeuristicEngine, HeuristicConfig, SearchStrategy};

fn main() {
    println!("=== RSSN-Advanced Example 06: Fiber Runtime & Parallel Dispatch ===\n");

    // 1. Initialize the global fiber pool (idempotent — safe to call many times)
    println!("Initializing dtact fiber pool...");
    let gate = ensure_runtime();
    println!("  Runtime gate acquired. Worker fibers ready.\n");

    // 2. Fire-and-forget task with explicit join
    println!("Spawning a single task onto the fiber pool...");
    let counter = Arc::new(AtomicU64::new(0));
    let c = Arc::clone(&counter);

    let handle = spawn_task(gate, move || {
        // Fibonacci(30) to burn a non-trivial amount of CPU work
        let result = fibonacci(30);
        c.fetch_add(result, Ordering::Release);
    });

    // Block until the fiber completes
    join(handle);
    println!("  Fiber finished. fibonacci(30) = {}\n", counter.load(Ordering::Acquire));

    // 3. Fan-out: simplify N independent symbolic expressions in parallel
    //    Each closure builds and simplifies its own sub-expression in isolation.
    let formulas = [
        "a * 1.0 + 0.0",          // simplifies to: a
        "b ^ 1.0",                 // simplifies to: b
        "c * 0.0 + d * 1.0",      // simplifies to: d
        "e + 0.0",                 // simplifies to: e
        "2.0 + 3.0",               // constant fold to: 5.0
        "f * 1.0 * 1.0",          // simplifies to: f
        "0.0 + g + 0.0",          // simplifies to: g
        "4.0 * 2.0 + h * 0.0",   // simplifies to: 8.0
    ];

    println!("Dispatching {} simplification tasks in parallel...", formulas.len());
    let start = std::time::Instant::now();

    let results = parallel_for_each(
        gate,
        formulas.iter().enumerate().map(|(i, &formula)| {
            move || {
                let mut builder = DagBuilder::new();
                let root = parse_expression(formula, &mut builder)
                    .unwrap_or_else(|_| panic!("parse failed for formula {i}"));
                let engine = HeuristicEngine::new(
                    HeuristicConfig::default(),
                    SearchStrategy::Greedy,
                );
                let simplified = engine.simplify(&mut builder, root);
                let node_count = builder.arena().len();
                (i, formula, simplified, node_count)
            }
        }),
    );

    let elapsed = start.elapsed();
    println!("All {} tasks completed in {:?}!\n", results.len(), elapsed);

    println!("  {:<5} {:<35} {:<15} {}", "Task", "Formula", "Simplified ID", "Arena nodes");
    println!("  {}", "-".repeat(75));
    for (i, formula, simplified_id, node_count) in &results {
        let id_str = format!("{simplified_id:?}");
        println!("  {i:<5} {formula:<35} {id_str:<15} {node_count}");
    }

    // 4. Aggregate: sum node counts across all parallel results
    let total_nodes: usize = results.iter().map(|(_, _, _, n)| n).sum();
    println!("\n  Total arena nodes across all parallel tasks: {}", total_nodes);

    println!("\n===================================================================");
}

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => {
            let mut a = 0u64;
            let mut b = 1u64;
            for _ in 2..=n {
                let next = a.saturating_add(b);
                a = b;
                b = next;
            }
            b
        }
    }
}
