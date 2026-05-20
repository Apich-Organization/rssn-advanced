//! High-level DAG construction API.
//!
//! `DagBuilder` provides a unified, thread-safe or single-threaded context
//! that holds the `SymbolRegistry`, the `DagArena`, and the `DedupMap`
//! to construct structurally deduplicated symbolic expressions.

use super::arena::DagArena;
use super::dedup::DedupMap;
use super::metadata::NodeFlags;
use super::node::{ChildList, DagNodeId};
use super::symbol::{OpKind, SymbolKind, SymbolRegistry};

/// The primary context for building symbolic expression DAGs.
///
/// It coordinates the symbol registry, arena storage, and deduplication map
/// to construct perfectly-shared Directed Acyclic Graphs.
#[derive(Debug, Clone, Default)]
pub struct DagBuilder {
    /// Opaque registry mapping names to `SymbolId`.
    registry: SymbolRegistry,
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
    pub fn arena_mut(&mut self) -> &mut DagArena {
        &mut self.arena
    }

    /// Accesses the underlying symbol registry.
    #[must_use]
    pub const fn registry(&self) -> &SymbolRegistry {
        &self.registry
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
            None,
            1.0,
            NodeFlags::EMPTY,
        )
    }

    /// Constructs a unique numeric constant node.
    pub fn constant(&mut self, val: f64) -> DagNodeId {
        let kind = SymbolKind::Constant;
        let hash = DedupMap::hash_constant(val);

        self.dedup.get_or_insert(
            &mut self.arena,
            kind,
            hash,
            ChildList::Empty,
            Some(val),
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

        self.dedup.get_or_insert(
            &mut self.arena,
            kind,
            hash,
            children,
            None,
            1.0,
            flags,
        )
    }

    /// Constructs a subtraction node: `left - right`.
    pub fn sub(&mut self, left: DagNodeId, right: DagNodeId) -> DagNodeId {
        let kind = SymbolKind::Operator(OpKind::Sub);
        let children = ChildList::from_slice(&[left, right]);
        let hash = DedupMap::hash_operator(&kind, &children);

        self.dedup.get_or_insert(
            &mut self.arena,
            kind,
            hash,
            children,
            None,
            1.0,
            NodeFlags::EMPTY,
        )
    }

    /// Constructs a multiplication node: `left * right`.
    pub fn mul(&mut self, left: DagNodeId, right: DagNodeId) -> DagNodeId {
        let kind = SymbolKind::Operator(OpKind::Mul);
        let children = ChildList::from_slice(&[left, right]);
        let hash = DedupMap::hash_operator(&kind, &children);
        let flags = NodeFlags::commutative_associative();

        self.dedup.get_or_insert(
            &mut self.arena,
            kind,
            hash,
            children,
            None,
            1.0,
            flags,
        )
    }

    /// Constructs a division node: `left / right`.
    pub fn div(&mut self, left: DagNodeId, right: DagNodeId) -> DagNodeId {
        let kind = SymbolKind::Operator(OpKind::Div);
        let children = ChildList::from_slice(&[left, right]);
        let hash = DedupMap::hash_operator(&kind, &children);

        self.dedup.get_or_insert(
            &mut self.arena,
            kind,
            hash,
            children,
            None,
            1.0,
            NodeFlags::EMPTY,
        )
    }

    /// Constructs an exponentiation node: `left ^ right`.
    pub fn pow(&mut self, left: DagNodeId, right: DagNodeId) -> DagNodeId {
        let kind = SymbolKind::Operator(OpKind::Pow);
        let children = ChildList::from_slice(&[left, right]);
        let hash = DedupMap::hash_operator(&kind, &children);

        self.dedup.get_or_insert(
            &mut self.arena,
            kind,
            hash,
            children,
            None,
            1.0,
            NodeFlags::EMPTY,
        )
    }

    /// Constructs a unary negation node: `-operand`.
    pub fn neg(&mut self, operand: DagNodeId) -> DagNodeId {
        let kind = SymbolKind::Operator(OpKind::Neg);
        let children = ChildList::from_slice(&[operand]);
        let hash = DedupMap::hash_operator(&kind, &children);

        self.dedup.get_or_insert(
            &mut self.arena,
            kind,
            hash,
            children,
            None,
            1.0,
            NodeFlags::EMPTY,
        )
    }

    /// Constructs an arbitrary custom operator or function node.
    pub fn operator(&mut self, kind: SymbolKind, children: &[DagNodeId], flags: NodeFlags) -> DagNodeId {
        let children_list = ChildList::from_slice(children);
        let hash = DedupMap::hash_operator(&kind, &children_list);

        self.dedup.get_or_insert(
            &mut self.arena,
            kind,
            hash,
            children_list,
            None,
            1.0,
            flags,
        )
    }

    /// Clears the builder state while retaining allocated capacities.
    pub fn clear(&mut self) {
        self.arena.clear();
        self.dedup.clear();
        // The registry can be cleared too if needed, but retaining it is often useful.
        // Let's reset the entire builder state to completely fresh.
        self.registry = SymbolRegistry::new();
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

        assert_eq!(expr1, expr2, "Structural deduplication failed for operators");

        // Build: x * 2.0
        let c = builder.constant(2.0);
        let expr3 = builder.mul(x, c);

        let node = builder.arena().get(expr3).unwrap();
        assert_eq!(node.children.len(), 2);
    }
}
