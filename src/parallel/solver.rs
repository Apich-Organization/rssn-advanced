//! Async parallel solver with work distribution.
//!
//! Spawns tasks/threads per chunk — each task gets its own cloned or read-only
//! view of the expression trees, allowing 100% lock-free parallel execution.

use std::thread;
use crate::dag::arena::DagArena;
use crate::dag::node::DagNodeId;
use crate::dag::symbol::{OpKind, SymbolKind};

/// Solves and evaluates a set of expression leaf chunks in parallel.
///
/// For each chunk (sub-list of leaf nodes), a thread is spawned to evaluate
/// the sum of the leaf sub-expressions using the provided variable values.
/// The threads run independently without shared mutable state.
pub fn parallel_evaluate(
    arena: &DagArena,
    chunks: Vec<Vec<DagNodeId>>,
    variables: &[f64],
) -> f64 {
    if chunks.is_empty() {
        return 0.0;
    }

    let mut handles = Vec::new();

    for chunk in chunks {
        let vars = variables.to_vec();
        let arena_clone = arena.clone();
        
        let handle = thread::spawn(move || {
            let mut sum = 0.0;
            for id in chunk {
                sum += evaluate_node(&arena_clone, id, &vars);
            }
            sum
        });
        handles.push(handle);
    }

    let mut total = 0.0;
    for handle in handles {
        if let Ok(val) = handle.join() {
            total += val;
        }
    }
    total
}

/// Evaluates a single node recursively.
#[must_use]
pub fn evaluate_node(arena: &DagArena, id: DagNodeId, vars: &[f64]) -> f64 {
    if let Some(node) = arena.get(id) {
        match node.kind {
            SymbolKind::Constant => node.value.unwrap_or(0.0),
            SymbolKind::Variable(sym_id) => {
                vars.get(sym_id.0 as usize).copied().unwrap_or(0.0)
            }
            SymbolKind::Operator(op) => {
                let mut child_vals = Vec::new();
                for child_id in node.children.iter() {
                    child_vals.push(evaluate_node(arena, child_id, vars));
                }
                match op {
                    OpKind::Add => child_vals.iter().sum(),
                    OpKind::Sub => {
                        let lhs = child_vals.first().copied().unwrap_or(0.0);
                        let rhs = child_vals.get(1).copied().unwrap_or(0.0);
                        lhs - rhs
                    }
                    OpKind::Mul => child_vals.iter().product(),
                    OpKind::Div => {
                        let lhs = child_vals.first().copied().unwrap_or(0.0);
                        let rhs = child_vals.get(1).copied().unwrap_or(1.0);
                        if rhs.abs() < f64::EPSILON {
                            0.0 // Handled gracefully
                        } else {
                            lhs / rhs
                        }
                    }
                    OpKind::Pow => {
                        let base = child_vals.first().copied().unwrap_or(0.0);
                        let exp = child_vals.get(1).copied().unwrap_or(0.0);
                        base.powf(exp)
                    }
                    OpKind::Neg => -child_vals.first().copied().unwrap_or(0.0),
                }
            }
            SymbolKind::Function(_) => 0.0,
        }
    } else {
        0.0
    }
}
