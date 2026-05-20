//! Async parallel solver with work distribution.
//!
//! Per `parallel_review §2` / `summary_review §2`, the previous
//! implementation cloned the entire `DagArena` once per worker chunk
//! and spun an OS thread per chunk via `std::thread::spawn`. For a
//! million-node DAG split into 16 chunks that's 16 × ~88 B × 1 M ≈
//! 1.4 GB of redundant allocations and 16 native thread spawns.
//!
//! The rewrite:
//!
//! 1. Arena is shared read-only via `Arc<DagArena>` — **one** Arc bump
//!    per fan-out instead of N clones.
//! 2. Tasks dispatch through [`crate::runtime::parallel_for_each`],
//!    which uses `dtact` fibers (`plan.md §4.3`).
//! 3. [`evaluate_node`] is now iterative (worklist + value stack)
//!    instead of recursive — no more stack-overflow risk on deep
//!    expressions.

use std::sync::Arc;

use crate::dag::arena::DagArena;
use crate::dag::node::DagNodeId;
use crate::dag::symbol::{OpKind, SymbolKind};
use crate::runtime::{ensure_runtime, parallel_for_each};

/// Solves and evaluates a set of expression-leaf chunks in parallel.
///
/// Convenience wrapper: if you already hold an `Arc<DagArena>`, call
/// [`parallel_evaluate_shared`] directly to avoid even the one-shot
/// `arena.clone()` here. For callers that only have a borrow, this
/// wraps once and dispatches.
#[must_use]
pub fn parallel_evaluate(
    arena: &DagArena,
    chunks: Vec<Vec<DagNodeId>>,
    variables: &[f64],
) -> f64 {
    let arc = Arc::new(arena.clone());
    parallel_evaluate_shared(&arc, chunks, variables)
}

/// Zero-clone parallel evaluator. The arena is shared read-only across
/// all worker fibers via `Arc`; no per-chunk duplication.
///
/// Takes `&Arc<DagArena>` so callers that already hold a long-lived
/// handle don't pay an `Arc::clone` for the outer call (the inner
/// task spawn still bumps the refcount once per chunk).
#[must_use]
pub fn parallel_evaluate_shared(
    arena: &Arc<DagArena>,
    chunks: Vec<Vec<DagNodeId>>,
    variables: &[f64],
) -> f64 {
    if chunks.is_empty() {
        return 0.0;
    }

    let gate = ensure_runtime();
    let vars_arc: Arc<Vec<f64>> = Arc::new(variables.to_vec());

    let tasks: Vec<_> = chunks
        .into_iter()
        .map(|chunk| {
            let arena_local = Arc::clone(arena);
            let vars_local = Arc::clone(&vars_arc);
            move || -> f64 {
                let mut sum = 0.0;
                for id in chunk {
                    sum += evaluate_node(&arena_local, id, &vars_local);
                }
                sum
            }
        })
        .collect();

    let partials = parallel_for_each(gate, tasks);
    partials.into_iter().sum()
}

// =========================================================================
// Iterative single-node evaluator
// =========================================================================

/// Evaluates a single DAG node against the variable bindings, iteratively.
///
/// Uses an explicit worklist + value-stack pattern (mirroring the JIT
/// iterative codegen in Phase 2). Depth is limited by heap, not by OS
/// stack, so a million-deep expression no longer overflows.
///
/// # Panics
///
/// Panics if the internal stack/value-stack invariant is broken — this
/// would indicate either arena corruption or a logic error inside this
/// function.
#[must_use]
pub fn evaluate_node(arena: &DagArena, id: DagNodeId, vars: &[f64]) -> f64 {
    if id.is_none() {
        return 0.0;
    }

    let mut stack: Vec<Frame> = Vec::with_capacity(64);
    let mut values: Vec<f64> = Vec::with_capacity(64);

    let root_arity = arena.get(id).map_or(0, |n| n.children.len());
    stack.push(Frame {
        id,
        arity: root_arity,
        cursor: 0,
    });

    while let Some(top) = stack.last_mut() {
        // Pull the next child id, if any, before any &mut self mutation.
        let next_child: Option<DagNodeId> = arena.get(top.id).and_then(|node| {
            let kids = node.children.as_slice();
            kids.get(top.cursor).copied()
        });

        if let Some(child_id) = next_child {
            top.cursor += 1;
            let child_arity = arena.get(child_id).map_or(0, |c| c.children.len());
            stack.push(Frame {
                id: child_id,
                arity: child_arity,
                cursor: 0,
            });
        } else {
            // All children evaluated → reduce this frame.
            let frame = stack.pop().expect("non-empty stack");
            let v = reduce_frame(arena, frame.id, frame.arity, &mut values, vars);
            values.push(v);
        }
    }

    values.pop().unwrap_or(0.0)
}

struct Frame {
    id: DagNodeId,
    arity: usize,
    cursor: usize,
}

fn reduce_frame(
    arena: &DagArena,
    id: DagNodeId,
    arity: usize,
    values: &mut Vec<f64>,
    vars: &[f64],
) -> f64 {
    let Some(node) = arena.get(id) else {
        // Drop any spurious children we might have pushed.
        values.truncate(values.len().saturating_sub(arity));
        return 0.0;
    };

    match node.kind {
        SymbolKind::Constant => node.value.unwrap_or(0.0),
        SymbolKind::Variable(sym_id) => vars.get(sym_id.0 as usize).copied().unwrap_or(0.0),
        SymbolKind::Function(_) => {
            // Custom functions aren't lowered here; consume their args.
            values.truncate(values.len().saturating_sub(arity));
            0.0
        }
        SymbolKind::Operator(op) => {
            let split_at = values.len().saturating_sub(arity);
            let child_vals: Vec<f64> = values.drain(split_at..).collect();
            apply_op(op, &child_vals)
        }
    }
}

fn apply_op(op: OpKind, child_vals: &[f64]) -> f64 {
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
            if rhs.abs() < f64::EPSILON { 0.0 } else { lhs / rhs }
        }
        OpKind::Pow => {
            let base = child_vals.first().copied().unwrap_or(0.0);
            let exp = child_vals.get(1).copied().unwrap_or(0.0);
            base.powf(exp)
        }
        OpKind::Neg => -child_vals.first().copied().unwrap_or(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::builder::DagBuilder;

    #[test]
    fn iterative_eval_matches_simple_expr() {
        // 3*x + y with x=2, y=4 should be 10.
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let y = b.variable("y");
        let three = b.constant(3.0);
        let mul = b.mul(three, x);
        let add = b.add(mul, y);
        let val = evaluate_node(b.arena(), add, &[2.0, 4.0]);
        assert!((val - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn iterative_eval_handles_deep_chain() {
        // ((x+x)+x)+x... 5000 deep should never blow the stack.
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let mut acc = x;
        for _ in 0..5000 {
            acc = b.add(acc, x);
        }
        let val = evaluate_node(b.arena(), acc, &[1.0]);
        // Each `add` contributes one extra +x.
        assert!((val - 5001.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parallel_evaluate_sums_chunks() {
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let chunks = vec![vec![x, x], vec![x], vec![x, x]];
        // 5 references to x; with x=2.0 each chunk sums = 4 + 2 + 4 = 10.
        let total = parallel_evaluate(b.arena(), chunks, &[2.0]);
        assert!((total - 10.0).abs() < f64::EPSILON);
    }
}
