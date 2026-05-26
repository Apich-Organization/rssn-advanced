//! Iterative tree/DAG traversal helpers.
//!
//! Every recursive traversal in `dag`, `ast`, `jit`, `parallel`, `heuristic`
//! and `parser` is a stack-overflow hazard for industrial-scale expressions
//! (millions of nodes, depths in the tens of thousands). The helpers here
//! replace those recursions with explicit work stacks living on the heap.
//!
//! ## Post-order semantics
//!
//! [`post_order`] visits every reachable node exactly once and guarantees the
//! child-before-parent ordering required by code-gen and evaluator passes.
//! For DAGs (where a node can be reached via multiple parents) it dedups
//! through a caller-provided `visited` predicate so that shared subgraphs
//! aren't visited repeatedly — preserving the hash-consing advantage.
//!
//! ## Pre-order semantics
//!
//! [`pre_order`] visits parents before children. Useful for rewrite passes
//! that need to short-circuit a subtree.

extern crate alloc;

use alloc::vec::Vec;

/// A frame on the explicit traversal stack.
///
/// `expanded = false` means "first encounter — push children, then revisit".
/// `expanded = true`  means "children already processed — emit this node".
struct Frame<Id> {
    id: Id,
    expanded: bool,
}

/// Iteratively visits `root` and every node reachable through `children` in
/// post-order (children before parent).
///
/// `is_visited` should return `true` if the id has already been emitted; it
/// is the caller's responsibility to mark ids inside `visit` so the helper
/// can skip shared subgraphs in a DAG without re-traversing them.
///
/// The work stack is `Vec`-backed so the only depth limit is heap memory.
pub fn post_order<Id, FC, FV, FVis>(
    root: Id,
    mut children_of: FC,
    mut is_visited: FV,
    mut visit: FVis,
) where
    Id: Copy,
    FC: FnMut(Id, &mut dyn FnMut(Id)),
    FV: FnMut(Id) -> bool,
    FVis: FnMut(Id),
{
    let mut stack: Vec<Frame<Id>> = Vec::with_capacity(64);
    stack.push(Frame {
        id: root,
        expanded: false,
    });

    while let Some(top) = stack.last_mut() {
        if top.expanded {
            let id = top.id;
            stack.pop();
            if !is_visited(id) {
                visit(id);
            }
            continue;
        }

        if is_visited(top.id) {
            stack.pop();
            continue;
        }

        top.expanded = true;
        let parent_id = top.id;

        // Collect children first, then push — borrow of `top` ends before
        // we touch `stack` again.
        let mut buf: Vec<Id> = Vec::new();
        children_of(parent_id, &mut |c| buf.push(c));

        // Push in reverse so the leftmost child is processed first.
        for child in buf.into_iter().rev() {
            stack.push(Frame {
                id: child,
                expanded: false,
            });
        }
    }
}

/// Iteratively visits `root` and every reachable node in pre-order
/// (parent before children).
///
/// `visit` may return `false` to prune the subtree rooted at the visited
/// node — children are not enqueued in that case.
pub fn pre_order<Id, FC, FV, FVis>(
    root: Id,
    mut children_of: FC,
    mut is_visited: FV,
    mut visit: FVis,
) where
    Id: Copy,
    FC: FnMut(Id, &mut dyn FnMut(Id)),
    FV: FnMut(Id) -> bool,
    FVis: FnMut(Id) -> bool,
{
    let mut stack: Vec<Id> = Vec::with_capacity(64);
    stack.push(root);

    while let Some(id) = stack.pop() {
        if is_visited(id) {
            continue;
        }
        if !visit(id) {
            continue;
        }
        let mut buf: Vec<Id> = Vec::new();
        children_of(id, &mut |c| buf.push(c));
        for child in buf.into_iter().rev() {
            stack.push(child);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;
    use core::cell::RefCell;

    // Tree:
    //         0
    //        / \
    //       1   2
    //      / \
    //     3   4
    fn tree() -> Vec<Vec<u32>> {
        alloc::vec![
            alloc::vec![1, 2],
            alloc::vec![3, 4],
            alloc::vec![],
            alloc::vec![],
            alloc::vec![],
        ]
    }

    // DAG: 0 -> {1, 2}; 1 -> {3}; 2 -> {3}.  Node 3 is shared.
    fn dag() -> Vec<Vec<u32>> {
        alloc::vec![
            alloc::vec![1, 2],
            alloc::vec![3],
            alloc::vec![3],
            alloc::vec![],
        ]
    }

    #[test]
    fn post_order_tree() {
        let edges = tree();
        let mut out: Vec<u32> = Vec::new();
        post_order(
            0u32,
            |id, push| {
                if let Some(cs) = edges.get(id as usize) {
                    for &c in cs {
                        push(c);
                    }
                }
            },
            |_| false,
            |id| out.push(id),
        );
        assert_eq!(out, alloc::vec![3, 4, 1, 2, 0]);
    }

    #[test]
    fn pre_order_tree() {
        let edges = tree();
        let mut out: Vec<u32> = Vec::new();
        pre_order(
            0u32,
            |id, push| {
                if let Some(cs) = edges.get(id as usize) {
                    for &c in cs {
                        push(c);
                    }
                }
            },
            |_| false,
            |id| {
                out.push(id);
                true
            },
        );
        assert_eq!(out, alloc::vec![0, 1, 3, 4, 2]);
    }

    #[test]
    fn post_order_dag_dedups_shared_child() {
        let edges = dag();
        let out: RefCell<Vec<u32>> = RefCell::new(Vec::new());
        let seen: RefCell<BTreeSet<u32>> = RefCell::new(BTreeSet::new());
        post_order(
            0u32,
            |id, push| {
                if let Some(cs) = edges.get(id as usize) {
                    for &c in cs {
                        push(c);
                    }
                }
            },
            |id| seen.borrow().contains(&id),
            |id| {
                seen.borrow_mut().insert(id);
                out.borrow_mut().push(id);
            },
        );
        let out = out.into_inner();
        // Node 3 must appear exactly once even though it has two parents.
        assert_eq!(out.iter().filter(|&&x| x == 3).count(), 1);
        let pos = |v: u32| out.iter().position(|&x| x == v).expect("node missing");
        // Children before parents: 3 < 1, 1 < 0.
        assert!(pos(3) < pos(1));
        assert!(pos(1) < pos(0));
    }

    #[test]
    fn pre_order_prune() {
        let edges = tree();
        let mut out: Vec<u32> = Vec::new();
        pre_order(
            0u32,
            |id, push| {
                if let Some(cs) = edges.get(id as usize) {
                    for &c in cs {
                        push(c);
                    }
                }
            },
            |_| false,
            |id| {
                out.push(id);
                // Skip below node 1 (so 3 and 4 are not visited).
                id != 1
            },
        );
        assert_eq!(out, alloc::vec![0, 1, 2]);
    }

    // 100_000-deep chain proves no recursion.
    #[test]
    fn deep_chain_no_overflow() {
        const N: u32 = 100_000;
        let mut count = 0usize;
        post_order(
            0u32,
            |id, push| {
                if id < N {
                    push(id + 1);
                }
            },
            |_| false,
            |_| {
                count += 1;
            },
        );
        assert_eq!(count, (N + 1) as usize);
    }
}
