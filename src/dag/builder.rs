//! High-level DAG construction API.
//!
//! `DagBuilder` provides a unified, thread-safe or single-threaded context
//! that holds the `SymbolRegistry`, the `DagArena`, and the `DedupMap`
//! to construct structurally deduplicated symbolic expressions.

use super::arena::DagArena;
use super::dedup::DedupMap;
use super::metadata::NodeFlags;
use super::node::{ChildList, DagNodeId};
use super::symbol::{FnId, OpKind, SymbolKind, SymbolRegistry};

/// The primary context for building symbolic expression DAGs.
///
/// It coordinates the symbol registry, arena storage, and deduplication map
/// to construct perfectly-shared Directed Acyclic Graphs.
#[derive(Debug, Clone, Default)]
pub struct DagBuilder {
    /// Opaque registry mapping variable names to `SymbolId`.
    registry: SymbolRegistry,
    /// Registry mapping function names to `FnId`. Stored separately so
    /// function IDs and variable IDs occupy independent namespaces.
    fn_registry: SymbolRegistry,
    /// Vector-backed contiguous storage for nodes.
    arena: DagArena,
    /// Fast structural deduplication lookup.
    dedup: DedupMap,
}

impl DagBuilder {
    /// Creates a new, empty `DagBuilder` context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: SymbolRegistry::new(),
            fn_registry: SymbolRegistry::new(),
            arena: DagArena::new(),
            dedup: DedupMap::new(),
        }
    }

    /// Accesses the underlying arena.
    #[must_use]
    pub const fn arena(&self) -> &DagArena {
        &self.arena
    }

    /// Accesses the underlying arena mutably.
    pub const fn arena_mut(&mut self) -> &mut DagArena {
        &mut self.arena
    }

    /// Accesses the underlying symbol registry.
    #[must_use]
    pub const fn registry(&self) -> &SymbolRegistry {
        &self.registry
    }

    /// Accesses the underlying function symbol registry.
    #[must_use]
    pub const fn fn_registry(&self) -> &SymbolRegistry {
        &self.fn_registry
    }

    /// Accesses the underlying deduplication map.
    #[must_use]
    pub const fn dedup(&self) -> &DedupMap {
        &self.dedup
    }

    /// Interns or retrieves a variable name, producing a unique leaf node.
    pub fn variable(&mut self, name: &str) -> DagNodeId {
        let sym_id = self.registry.intern(name);
        self.variable_with_sym_id(sym_id)
    }

    /// Like [`Self::variable`] but accepts a raw byte slice — used by
    /// the FFI surface to skip the `to_string_lossy` allocation.
    ///
    /// Returns `None` if `name_bytes` is not valid UTF-8.
    pub fn variable_bytes(&mut self, name_bytes: &[u8]) -> Option<DagNodeId> {
        let sym_id = self.registry.intern_bytes(name_bytes)?;
        Some(self.variable_with_sym_id(sym_id))
    }

    fn variable_with_sym_id(&mut self, sym_id: crate::dag::symbol::SymbolId) -> DagNodeId {
        let kind = SymbolKind::Variable(sym_id);
        let hash = DedupMap::hash_variable(&kind);
        self.dedup.get_or_insert(
            &mut self.arena,
            kind,
            hash,
            ChildList::Empty,
            1.0,
            NodeFlags::EMPTY,
        )
    }

    /// Constructs a unique numeric constant node.
    pub fn constant(&mut self, val: f64) -> DagNodeId {
        let kind = SymbolKind::Constant(val);
        let hash = DedupMap::hash_constant(&kind);

        self.dedup.get_or_insert(
            &mut self.arena,
            kind,
            hash,
            ChildList::Empty,
            val, // coefficient matches the constant value for leaf constants
            NodeFlags::EMPTY,
        )
    }

    /// Constructs an addition node: `left + right`.
    pub fn add(&mut self, left: DagNodeId, right: DagNodeId) -> DagNodeId {
        let kind = SymbolKind::Operator(OpKind::Add);
        let children = ChildList::from_slice(&[left, right]);
        let hash = DedupMap::hash_operator(&kind, &children);
        let flags = NodeFlags::commutative_associative();

        self.dedup
            .get_or_insert(&mut self.arena, kind, hash, children, 1.0, flags)
    }

    /// Constructs a subtraction node: `left - right`.
    pub fn sub(&mut self, left: DagNodeId, right: DagNodeId) -> DagNodeId {
        let kind = SymbolKind::Operator(OpKind::Sub);
        let children = ChildList::from_slice(&[left, right]);
        let hash = DedupMap::hash_operator(&kind, &children);

        self.dedup
            .get_or_insert(&mut self.arena, kind, hash, children, 1.0, NodeFlags::EMPTY)
    }

    /// Constructs a multiplication node: `left * right`.
    pub fn mul(&mut self, left: DagNodeId, right: DagNodeId) -> DagNodeId {
        let kind = SymbolKind::Operator(OpKind::Mul);
        let children = ChildList::from_slice(&[left, right]);
        let hash = DedupMap::hash_operator(&kind, &children);
        let flags = NodeFlags::commutative_associative();

        self.dedup
            .get_or_insert(&mut self.arena, kind, hash, children, 1.0, flags)
    }

    /// Constructs a division node: `left / right`.
    pub fn div(&mut self, left: DagNodeId, right: DagNodeId) -> DagNodeId {
        let kind = SymbolKind::Operator(OpKind::Div);
        let children = ChildList::from_slice(&[left, right]);
        let hash = DedupMap::hash_operator(&kind, &children);

        self.dedup
            .get_or_insert(&mut self.arena, kind, hash, children, 1.0, NodeFlags::EMPTY)
    }

    /// Constructs an exponentiation node: `left ^ right`.
    pub fn pow(&mut self, left: DagNodeId, right: DagNodeId) -> DagNodeId {
        let kind = SymbolKind::Operator(OpKind::Pow);
        let children = ChildList::from_slice(&[left, right]);
        let hash = DedupMap::hash_operator(&kind, &children);

        self.dedup
            .get_or_insert(&mut self.arena, kind, hash, children, 1.0, NodeFlags::EMPTY)
    }

    /// Constructs a floating-point remainder node: `left % right`.
    ///
    /// Semantics follow IEEE-754 `remainder` (same as Cranelift `frem` and Rust `f64::rem`):
    /// result has the same sign as the dividend, and `x % 0 → NaN`.
    pub fn modulo(&mut self, left: DagNodeId, right: DagNodeId) -> DagNodeId {
        let kind = SymbolKind::Operator(OpKind::Mod);
        let children = ChildList::from_slice(&[left, right]);
        let hash = DedupMap::hash_operator(&kind, &children);

        self.dedup
            .get_or_insert(&mut self.arena, kind, hash, children, 1.0, NodeFlags::EMPTY)
    }

    /// Constructs a unary negation node: `-operand`.
    pub fn neg(&mut self, operand: DagNodeId) -> DagNodeId {
        let kind = SymbolKind::Operator(OpKind::Neg);
        let children = ChildList::from_slice(&[operand]);
        let hash = DedupMap::hash_operator(&kind, &children);

        self.dedup
            .get_or_insert(&mut self.arena, kind, hash, children, 1.0, NodeFlags::EMPTY)
    }

    /// Interns a function name and returns its [`FnId`].
    ///
    /// The function namespace is independent of the variable namespace, so a
    /// function named `"x"` never collides with a variable named `"x"`.
    pub fn intern_function(&mut self, name: &str) -> FnId {
        let sym_id = self.fn_registry.intern(name);
        FnId(sym_id.0)
    }

    /// Returns the name of the function with the given [`FnId`], if interned.
    #[must_use]
    pub fn function_name(&self, fn_id: FnId) -> Option<&str> {
        use super::symbol::SymbolId;
        self.fn_registry.name(SymbolId(fn_id.0))
    }

    /// Constructs a function-call node: `name(args...)`.
    ///
    /// Equivalent to calling [`Self::operator`] with `SymbolKind::Function`,
    /// but looks up the function's name in the registry for you.
    pub fn function_call(&mut self, fn_id: FnId, args: &[DagNodeId]) -> DagNodeId {
        self.operator(SymbolKind::Function(fn_id), args, NodeFlags::EMPTY)
    }

    /// Constructs an arbitrary custom operator or function node.
    pub fn operator(
        &mut self,
        kind: SymbolKind,
        children: &[DagNodeId],
        flags: NodeFlags,
    ) -> DagNodeId {
        let children_list = ChildList::from_slice(children);
        let hash = DedupMap::hash_operator(&kind, &children_list);

        self.dedup
            .get_or_insert(&mut self.arena, kind, hash, children_list, 1.0, flags)
    }

    /// Constructs a branchless value selection node: `select(cond, then_val, else_val)`.
    ///
    /// Compiles to a single Cranelift `select` instruction — no branches, no penalties.
    /// The condition is an `f64`; any non-zero value (including NaN) is treated as `true`.
    ///
    /// This is ideal for masking operations inside SIMD kernels.
    pub fn select(
        &mut self,
        cond: DagNodeId,
        then_val: DagNodeId,
        else_val: DagNodeId,
    ) -> DagNodeId {
        use super::symbol::CtrlKind;
        let kind = SymbolKind::ControlFlow(CtrlKind::Select);
        self.operator(kind, &[cond, then_val, else_val], NodeFlags::EMPTY)
    }

    /// Constructs an if-else control flow node.
    ///
    /// Compiles to two basic blocks and a merge (phi-node) block in Cranelift IR.
    /// Unlike `select`, this correctly handles expressions with divergent branches
    /// (e.g. early return patterns).
    ///
    /// - `cond`: condition (f64; non-zero = true).
    /// - `then_expr`: value when condition is true.
    /// - `else_expr`: value when condition is false.
    pub fn if_else(
        &mut self,
        cond: DagNodeId,
        then_expr: DagNodeId,
        else_expr: DagNodeId,
    ) -> DagNodeId {
        use super::symbol::CtrlKind;
        let kind = SymbolKind::ControlFlow(CtrlKind::IfElse);
        self.operator(kind, &[cond, then_expr, else_expr], NodeFlags::EMPTY)
    }

    /// Constructs a counted for-loop accumulator node.
    ///
    /// Compiles to a loop header, loop body, and loop exit block in Cranelift IR
    /// using SSA block parameters to carry the accumulator across iterations.
    ///
    /// - `init`: initial accumulator value.
    /// - `limit`: loop runs while `loop_idx < limit` (exclusive upper bound).
    /// - `step`: loop index is incremented by `step` each iteration.
    /// - `body`: expression evaluated each iteration; may reference the current
    ///    loop index and accumulator via `LoopVar`/`Acc` special variables.
    pub fn for_loop(
        &mut self,
        init: DagNodeId,
        limit: DagNodeId,
        step: DagNodeId,
        body: DagNodeId,
    ) -> DagNodeId {
        use super::symbol::CtrlKind;
        let kind = SymbolKind::ControlFlow(CtrlKind::ForLoop);
        self.operator(kind, &[init, limit, step, body], NodeFlags::EMPTY)
    }

    /// Resets the builder to a completely fresh state.
    pub fn clear(&mut self) {
        self.arena.clear();
        self.dedup.clear();
        self.registry = SymbolRegistry::new();
        self.fn_registry = SymbolRegistry::new();
    }

    /// Returns the number of nodes currently in the arena.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.arena.len()
    }

    /// Returns `true` if the arena contains no nodes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    /// Returns a snapshot of the arena in the 32-byte packed format.
    ///
    /// This is the primary bridge to the packed wire representation:
    /// the returned [`super::packed::PackedArenaImage`] can be encoded
    /// for disk storage, shipped across an FFI boundary as a contiguous
    /// byte buffer, or iterated cache-efficiently for batch operations.
    ///
    /// The snapshot is a *copy* — subsequent mutations to the builder do
    /// not affect it. For a live zero-copy view, encode then decode via
    /// `BorrowedArenaView`.
    #[must_use]
    pub fn packed_snapshot(&self) -> super::packed::PackedArenaImage {
        super::packed::PackedArenaImage::from_arena(&self.arena)
    }

    /// Iterates all nodes in ID order, calling `f(id, &node)` for each.
    ///
    /// Uses the arena's contiguous storage for cache-friendly sequential
    /// access. Equivalent to iterating `0..node_count()` and calling
    /// `arena().get(DagNodeId(i))`, but avoids repeated bounds checks.
    pub fn for_each_node<F>(&self, mut f: F)
    where
        F: FnMut(DagNodeId, &super::node::DagNode),
    {
        for i in 0..self.arena.len() {
            let id = DagNodeId(i as u32);
            if let Some(node) = self.arena.get(id) {
                f(id, node);
            }
        }
    }

    /// Parses a textual expression and inserts it into this builder's DAG.
    ///
    /// This is a convenience wrapper around [`crate::parser::parse_expression`].
    /// It avoids the need to import the parser separately for simple use cases.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::parser::error::ParseError`] if the expression is
    /// syntactically invalid or if the paren depth is exceeded.
    pub fn parse(
        &mut self,
        expr: &str,
    ) -> Result<super::node::DagNodeId, crate::parser::error::ParseError> {
        crate::parser::expr::parse_expression(expr, self)
    }

    /// Constructs the sum of all nodes in `terms`.
    ///
    /// Returns the single node if `terms` has exactly one element, or
    /// constructs a left-associative addition tree otherwise.
    ///
    /// Returns `None` if `terms` is empty.
    #[must_use]
    pub fn add_many(&mut self, terms: &[super::node::DagNodeId]) -> Option<super::node::DagNodeId> {
        let mut iter = terms.iter().copied();
        let first = iter.next()?;
        Some(iter.fold(first, |acc, t| self.add(acc, t)))
    }

    /// Constructs the product of all nodes in `factors`.
    ///
    /// Returns the single node if `factors` has exactly one element, or
    /// constructs a left-associative multiplication tree otherwise.
    ///
    /// Returns `None` if `factors` is empty.
    #[must_use]
    pub fn mul_many(
        &mut self,
        factors: &[super::node::DagNodeId],
    ) -> Option<super::node::DagNodeId> {
        let mut iter = factors.iter().copied();
        let first = iter.next()?;
        Some(iter.fold(first, |acc, f| self.mul(acc, f)))
    }

    /// Constructs `base^2` (one multiplication, no `powf` call).
    ///
    /// Equivalent to `self.mul(base, base)` but communicates intent more
    /// clearly at the call site.
    pub fn square(&mut self, base: super::node::DagNodeId) -> super::node::DagNodeId {
        self.mul(base, base)
    }

    /// Constructs `-(lhs - rhs)` = `rhs - lhs` without an extra negation node.
    pub fn sub_rev(
        &mut self,
        lhs: super::node::DagNodeId,
        rhs: super::node::DagNodeId,
    ) -> super::node::DagNodeId {
        self.sub(rhs, lhs)
    }

    /// Returns the `SymbolId` of a variable, if it has already been interned.
    ///
    /// Unlike [`Self::variable`] this does NOT create a new node or intern the name.
    #[must_use]
    pub fn lookup_variable(&self, name: &str) -> Option<super::symbol::SymbolId> {
        self.registry.lookup(name)
    }

    /// Returns the `FnId` of a function, if it has already been interned.
    #[must_use]
    pub fn lookup_function(&self, name: &str) -> Option<super::symbol::FnId> {
        self.fn_registry.lookup(name).map(|sid| FnId(sid.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_simple_expressions() {
        let mut builder = DagBuilder::new();

        // Build: x + y
        let x = builder.variable("x");
        let y = builder.variable("y");
        let expr1 = builder.add(x, y);

        // Build: x + y again
        let expr2 = builder.add(x, y);

        assert_eq!(
            expr1, expr2,
            "Structural deduplication failed for operators"
        );

        // Build: x * 2.0
        let c = builder.constant(2.0);
        let expr3 = builder.mul(x, c);

        let node = builder.arena().get(expr3).unwrap();
        assert_eq!(node.children.len(), 2);
    }
}
