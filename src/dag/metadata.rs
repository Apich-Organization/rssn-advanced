//! Per-node metadata for DAG entries.
//!
//! Each DAG node carries metadata that describes its algebraic properties:
//! structural hash (via `rapidhash`), numeric coefficient, arity, and
//! commutativity flags.

use bincode_next::{Decode, Encode};

/// Hash value computed by `rapidhash` for structural deduplication.
///
/// Two nodes with the same `NodeHash` are structurally identical
/// (modulo hash collisions, which are verified by full structural comparison).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Encode, Decode)]
pub struct NodeHash(pub(crate) u64);

impl NodeHash {
    /// The hash value for an empty / uninitialized node.
    pub const ZERO: Self = Self(0);
}

/// Flags that describe algebraic properties of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Encode, Decode)]
pub struct NodeFlags {
    bits: u8,
}

impl NodeFlags {
    /// No flags set.
    pub const EMPTY: Self = Self { bits: 0 };

    /// This node's operator is commutative (e.g. `+`, `*`).
    const COMMUTATIVE: u8 = 0b0000_0001;
    /// This node's operator is associative.
    const ASSOCIATIVE: u8 = 0b0000_0010;
    /// This node has been simplified / is in canonical form.
    const CANONICAL: u8 = 0b0000_0100;

    /// Creates flags with commutativity set.
    #[must_use]
    pub const fn commutative() -> Self {
        Self {
            bits: Self::COMMUTATIVE,
        }
    }

    /// Creates flags with both commutativity and associativity set.
    #[must_use]
    pub const fn commutative_associative() -> Self {
        Self {
            bits: Self::COMMUTATIVE | Self::ASSOCIATIVE,
        }
    }

    /// Returns `true` if the commutative flag is set.
    #[must_use]
    pub const fn is_commutative(self) -> bool {
        self.bits & Self::COMMUTATIVE != 0
    }

    /// Returns `true` if the associative flag is set.
    #[must_use]
    pub const fn is_associative(self) -> bool {
        self.bits & Self::ASSOCIATIVE != 0
    }

    /// Returns `true` if the canonical flag is set.
    #[must_use]
    pub const fn is_canonical(self) -> bool {
        self.bits & Self::CANONICAL != 0
    }

    /// Sets the canonical flag.
    #[must_use]
    pub const fn with_canonical(self) -> Self {
        Self {
            bits: self.bits | Self::CANONICAL,
        }
    }
}

/// Metadata attached to every DAG node.
///
/// This is stored inline in the `DagNode` and contains all the information
/// needed for hash-consing, algebraic classification, and simplification.
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct NodeMetadata {
    /// Structural hash for deduplication lookups.
    pub hash: NodeHash,
    /// Numeric coefficient (e.g. the `3` in `3*x`).
    /// Defaults to `1.0` for non-coefficient nodes.
    pub coefficient: f64,
    /// Number of children this node has.
    pub arity: u16,
    /// Algebraic property flags.
    pub flags: NodeFlags,
}

impl NodeMetadata {
    /// Creates metadata for a leaf node (arity 0) with coefficient 1.
    #[must_use]
    pub const fn leaf(hash: NodeHash) -> Self {
        Self {
            hash,
            coefficient: 1.0,
            arity: 0,
            flags: NodeFlags::EMPTY,
        }
    }

    /// Creates metadata for an operator node with the given arity and flags.
    #[must_use]
    pub const fn operator(hash: NodeHash, arity: u16, flags: NodeFlags) -> Self {
        Self {
            hash,
            coefficient: 1.0,
            arity,
            flags,
        }
    }

    /// Returns a copy with the coefficient set to the given value.
    #[must_use]
    pub const fn with_coefficient(mut self, coeff: f64) -> Self {
        self.coefficient = coeff;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_round_trip() {
        let f = NodeFlags::commutative_associative();
        assert!(f.is_commutative());
        assert!(f.is_associative());
        assert!(!f.is_canonical());

        let f2 = f.with_canonical();
        assert!(f2.is_canonical());
        assert!(f2.is_commutative());
    }

    #[test]
    fn leaf_metadata() {
        let m = NodeMetadata::leaf(NodeHash(42));
        assert_eq!(m.arity, 0);
        assert_eq!(m.coefficient, 1.0);
        assert_eq!(m.hash, NodeHash(42));
    }

    #[test]
    fn coefficient_override() {
        let m = NodeMetadata::leaf(NodeHash(1)).with_coefficient(3.5);
        assert_eq!(m.coefficient, 3.5);
    }
}
