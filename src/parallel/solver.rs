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

/// Registry of function-pointer callbacks for parallel evaluation of
/// `SymbolKind::Function` nodes.
///
/// Keys are [`crate::dag::symbol::FnId`] values. The registered function
/// receives a pointer to the variable array (same layout as JIT functions)
/// and returns an `f64`.
///
/// Pass a `&FnEvalRegistry` to [`evaluate_node_with_fns`] to enable
/// function-node evaluation in the parallel path.
#[derive(Default)]
pub struct FnEvalRegistry {
    fns: std::collections::HashMap<u32, fn(*const f64) -> f64>,
}

impl FnEvalRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fns: std::collections::HashMap::new(),
        }
    }

    /// Registers a function callback for `fn_id`.
    pub fn register(&mut self, fn_id: crate::dag::symbol::FnId, f: fn(*const f64) -> f64) {
        self.fns.insert(fn_id.0, f);
    }

    /// Looks up the callback for `fn_id`.
    #[must_use]
    pub fn get(&self, fn_id: crate::dag::symbol::FnId) -> Option<fn(*const f64) -> f64> {
        self.fns.get(&fn_id.0).copied()
    }
}

/// Registry of operator-evaluation overrides for the parallel evaluator.
///
/// Overrides are keyed by `OpKind` discriminant. When a registered override
/// exists for an op, it is called instead of `apply_op`.
#[derive(Default)]
pub struct OpEvalRegistry {
    overrides: std::collections::HashMap<u8, fn(&[f64]) -> f64>,
}

impl OpEvalRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            overrides: std::collections::HashMap::new(),
        }
    }

    /// Registers a custom evaluation function for `op`.
    pub fn register(&mut self, op: crate::dag::symbol::OpKind, f: fn(&[f64]) -> f64) {
        self.overrides.insert(op as u8, f);
    }

    /// Looks up an override for `op`, if any.
    #[must_use]
    pub fn get(&self, op: crate::dag::symbol::OpKind) -> Option<fn(&[f64]) -> f64> {
        self.overrides.get(&(op as u8)).copied()
    }
}

/// Variant of [`evaluate_node_with_fns`] that also accepts an [`OpEvalRegistry`]
/// for overriding built-in operator evaluation.
///
/// When an operator has a registered override, it is called instead of
/// `apply_op`. Unregistered operators use the default `apply_op` logic.
#[must_use]
pub fn evaluate_node_with_overrides(
    arena: &DagArena,
    id: DagNodeId,
    vars: &[f64],
    fn_registry: &FnEvalRegistry,
    op_registry: &OpEvalRegistry,
) -> f64 {
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
            let Some(frame) = stack.pop() else { break };
            let v = reduce_frame_with_overrides(
                arena,
                frame.id,
                frame.arity,
                &mut values,
                vars,
                fn_registry,
                op_registry,
            );
            values.push(v);
        }
    }

    values.pop().unwrap_or(0.0)
}

fn reduce_frame_with_overrides(
    arena: &DagArena,
    id: DagNodeId,
    arity: usize,
    values: &mut Vec<f64>,
    vars: &[f64],
    fn_registry: &FnEvalRegistry,
    op_registry: &OpEvalRegistry,
) -> f64 {
    let Some(node) = arena.get(id) else {
        values.truncate(values.len().saturating_sub(arity));
        return 0.0;
    };

    match node.kind {
        SymbolKind::Constant(v) => v,
        SymbolKind::Variable(sym_id) => vars.get(sym_id.0 as usize).copied().unwrap_or(0.0),
        SymbolKind::Function(fn_id) => {
            values.truncate(values.len().saturating_sub(arity));
            fn_registry.get(fn_id).map_or(0.0, |f| f(vars.as_ptr()))
        }
        SymbolKind::Operator(op) => {
            let split_at = values.len().saturating_sub(arity);
            let result = op_registry
                .get(op)
                .map_or_else(|| apply_op(op, &values[split_at..]), |override_fn| override_fn(&values[split_at..]));
            values.truncate(split_at);
            result
        }
    }
}

/// Solves and evaluates a set of expression-leaf chunks in parallel.
///
/// Convenience wrapper: if you already hold an `Arc<DagArena>`, call
/// [`parallel_evaluate_shared`] directly to avoid even the one-shot
/// `arena.clone()` here. For callers that only have a borrow, this
/// wraps once and dispatches.
#[must_use]
pub fn parallel_evaluate(arena: &DagArena, chunks: Vec<Vec<DagNodeId>>, variables: &[f64]) -> f64 {
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
    partials.into_iter().flatten().sum()
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
/// Panic-free: if the arena is corrupt the function returns whatever
/// partial result it has built so far (typically `0.0`).
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
            // All children evaluated → reduce this frame. The `Some`
            // here is guaranteed by the outer `while let Some(top)`
            // we just exited; if it somehow isn't, we treat it as
            // "no more work" and return whatever we have on the
            // value stack.
            let Some(frame) = stack.pop() else { break };
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
        SymbolKind::Constant(v) => v,
        SymbolKind::Variable(sym_id) => vars.get(sym_id.0 as usize).copied().unwrap_or(0.0),
        SymbolKind::Function(_) => {
            // Custom functions aren't lowered here; consume their args.
            values.truncate(values.len().saturating_sub(arity));
            0.0
        }
        SymbolKind::Operator(op) => {
            let split_at = values.len().saturating_sub(arity);
            // Borrow as slice to avoid a per-node Vec allocation. NLL
            // guarantees the immutable borrow ends before `truncate`.
            let result = apply_op(op, &values[split_at..]);
            values.truncate(split_at);
            result
        }
    }
}

/// Variant of [`evaluate_node`] that supports `SymbolKind::Function` nodes
/// via a user-supplied [`FnEvalRegistry`].
///
/// When a `Function` node is encountered the registry is consulted; if a
/// matching callback is found it is called with the variable pointer.
/// Unregistered functions fall back to `0.0` (consistent with the JIT
/// behaviour for unresolved symbols).
#[must_use]
pub fn evaluate_node_with_fns(
    arena: &DagArena,
    id: DagNodeId,
    vars: &[f64],
    registry: &FnEvalRegistry,
) -> f64 {
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
            let Some(frame) = stack.pop() else { break };
            let v =
                reduce_frame_with_fns(arena, frame.id, frame.arity, &mut values, vars, registry);
            values.push(v);
        }
    }

    values.pop().unwrap_or(0.0)
}

fn reduce_frame_with_fns(
    arena: &DagArena,
    id: DagNodeId,
    arity: usize,
    values: &mut Vec<f64>,
    vars: &[f64],
    registry: &FnEvalRegistry,
) -> f64 {
    let Some(node) = arena.get(id) else {
        values.truncate(values.len().saturating_sub(arity));
        return 0.0;
    };

    match node.kind {
        SymbolKind::Constant(v) => v,
        SymbolKind::Variable(sym_id) => vars.get(sym_id.0 as usize).copied().unwrap_or(0.0),
        SymbolKind::Function(fn_id) => {
            values.truncate(values.len().saturating_sub(arity));
            // Invoke the registered callback with the variable pointer.
            registry.get(fn_id).map_or(0.0, |f| f(vars.as_ptr()))
        }
        SymbolKind::Operator(op) => {
            let split_at = values.len().saturating_sub(arity);
            let result = apply_op(op, &values[split_at..]);
            values.truncate(split_at);
            result
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
            // Return NaN on exact zero — consistent with IEEE-754 and the
            // JIT path. Returning 0.0 or using EPSILON threshold masks bugs.
            if rhs == 0.0 { f64::NAN } else { lhs / rhs }
        }
        OpKind::Mod => {
            let lhs = child_vals.first().copied().unwrap_or(0.0);
            let rhs = child_vals.get(1).copied().unwrap_or(1.0);
            if rhs == 0.0 { f64::NAN } else { lhs % rhs }
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
