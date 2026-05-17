//! Example 01: DAG Construction, Parser, and Algebraic Simplification
//!
//! This example demonstrates how to parse a text-based mathematical expression,
//! construct a deduplicated Directed Acyclic Graph (DAG) in memory, verify structural
//! node sharing, and apply heuristic algebraic simplification.
//!
//! Run with: `cargo run --example 01_dag_builder_and_simplification`

use rssn_advanced::dag::builder::DagBuilder;
use rssn_advanced::parser::parse_expression;
use rssn_advanced::heuristic::{HeuristicEngine, HeuristicConfig, SearchStrategy};

fn main() {
    println!("=== RSSN-Advanced Example 01: DAG & Simplification ===\n");

    // 1. Initialize a thread-safe DAG context builder
    let mut builder = DagBuilder::new();

    // 2. Parse a symbolic formula using the Nom parser
    // Identical sub-expressions "x * y" are repeated in the string representation
    let expr_str = "x * y + (x * y) + 5.0 + 3.0";
    println!("Parsing symbolic formula: \"{}\"", expr_str);
    
    let root_id = parse_expression(expr_str, &mut builder)
        .expect("Failed to parse expression string");
    
    println!("Parsing completed. Root node allocated at index: {:?}\n", root_id);

    // 3. Demonstrate Structural Sharing / Deduplication
    // Check that both instances of "x * y" were deduplicated to the exact same node ID!
    let x_id = builder.variable("x");
    let y_id = builder.variable("y");
    
    // Explicitly build "x * y" from the builder, which looks up or inserts
    let xy_id = builder.mul(x_id, y_id);
    println!("Structural Sharing Verification:");
    println!("  Variable 'x' node index        : {:?}", x_id);
    println!("  Variable 'y' node index        : {:?}", y_id);
    println!("  Sub-expression 'x * y' index   : {:?}", xy_id);
    
    // Total nodes currently in the arena
    let total_nodes = builder.arena().len();
    println!("  Total unique nodes in DagArena : {}\n", total_nodes);

    // 4. Perform Heuristic Algebraic Simplification
    // Constant folding (5.0 + 3.0 = 8.0) and term aggregation (xy + xy = 2 * xy)
    println!("Applying Heuristic Algebraic Simplification...");
    let config = HeuristicConfig::default();
    let engine = HeuristicEngine::new(config, SearchStrategy::Greedy);
    
    let simplified_id = engine.simplify(builder.arena_mut(), root_id);
    println!("Simplification completed.");
    println!("  Original root node index       : {:?}", root_id);
    println!("  Simplified root node index     : {:?}", simplified_id);
    println!("  New unique nodes in DagArena   : {}\n", builder.arena().len());

    println!("=====================================================");
}
