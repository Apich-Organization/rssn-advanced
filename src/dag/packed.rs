//! Packed, `Pod`-compatible representation of DAG nodes.
//!
//! ## Why this lives alongside `DagNode`
//!
//! The "rich" `DagNode` carries owned types (`String` via `SymbolKind`'s
//! transitive deps, `Vec` via `ChildList::Many`, `Option<f64>` whose layout
//! is not stable). Those are convenient for in-memory construction but
//! prevent `bincode-next::BorrowDecode` — and prevent us from
//! `mmap`-ing an arena directly off disk.
//!
//! [`PackedDagNode`] is a 32-byte `#[repr(C)]` value that satisfies the
//! `Pod` contract from `crate::zerocopy`. It stores:
//!
//! ```text
//! offset  size  field
//!   0       8   hash             (NodeHash)
//!   8       8   coefficient      (f64; also doubles as constant value)
//!  16       4   child0           (inline child #0, or pool offset)
//!  20       4   child1           (inline child #1, or unused)
//!  24       4   kind_payload     (SymbolId / FnId; unused for op/const)
//!  28       1   arity            (0..=2 inline; >2 → use pool[child0..child0+arity])
//!  29       1   kind_tag         (0=Var, 1=Const, 2=Op, 3=Fn)
//!  30       1   op_tag           (only meaningful when kind_tag==Op)
//!  31       1   flags            (NodeFlags bits)
//! ```
//!
//! The arena image bundles the packed node array with two side pools:
//!
//! * `children_pool: BorrowedSlice<'_, u32>` — overflow children for nodes
//!   with arity > 2.
//! * No separate constants pool — constants store their value in
//!   `coefficient` directly (`Option::None` becomes `coefficient = 0.0`
//!   on non-constant nodes; the `kind_tag` disambiguates).
//!
//! ## Conversion
//!
//! [`PackedArenaImage::from_arena`] performs the rich → packed conversion
//! (allocates `Vec`s, owns the data). [`PackedArenaImage::encode`] writes
//! it to an [`AlignedBytes`]. [`BorrowedArenaView::decode`] reads it back
//! without copying any node data.
//!
//! Future task T1.2 will swap the rich `DagNode` out and have the arena
//! store `PackedDagNode` directly. Until then, callers that need the
//! zero-copy path use this side-by-side image.

#![allow(unsafe_code)]

use bincode_next::de::{BorrowDecode, BorrowDecoder};
use bincode_next::enc::{Encode, Encoder};
use bincode_next::error::{DecodeError, EncodeError};

use crate::dag::arena::DagArena;
use crate::dag::metadata::{NodeFlags, NodeHash, NodeMetadata};
use crate::dag::node::{ChildList, DagNode, DagNodeId};
use crate::dag::symbol::{FnId, OpKind, SymbolId, SymbolKind};
use bincode_next::enc::write::Writer as BincodeWriter;
use crate::zerocopy::{AlignedBytes, BorrowedSlice, Pod, decode_zerocopy, encode_zerocopy};

extern crate alloc;
use alloc::vec::Vec;

// =========================================================================
// PackedDagNode — the 32-byte canonical wire form
// =========================================================================

/// Discriminant tags used in [`PackedDagNode::kind_tag`].
pub mod kind_tag {
    /// Variable: `kind_payload` carries the `SymbolId`.
    pub const VARIABLE: u8 = 0;
    /// Constant: numeric value lives in the `coefficient` field.
    pub const CONSTANT: u8 = 1;
    /// Operator: `op_tag` carries the `OpKind`.
    pub const OPERATOR: u8 = 2;
    /// Function: `kind_payload` carries the `FnId`.
    pub const FUNCTION: u8 = 3;
}

/// Discriminant tags used in [`PackedDagNode::op_tag`] (only meaningful
/// when `kind_tag == kind_tag::OPERATOR`).
pub mod op_tag {
    /// `+`
    pub const ADD: u8 = 0;
    /// `-`
    pub const SUB: u8 = 1;
    /// `*`
    pub const MUL: u8 = 2;
    /// `/`
    pub const DIV: u8 = 3;
    /// `^`
    pub const POW: u8 = 4;
    /// unary `-`
    pub const NEG: u8 = 5;
    /// `%` (IEEE-754 remainder)
    pub const MOD: u8 = 6;
}

/// The 32-byte canonical wire-image of a DAG node.
///
/// `#[repr(C)]` plus no padding plus no references plus `Copy` are what
/// make this safe to reinterpret straight out of a borrowed byte slice
/// via [`crate::zerocopy::Pod`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PackedDagNode {
    /// Structural hash (matches `NodeMetadata::hash`).
    pub hash: u64,
    /// For constants: numeric value. For others: algebraic coefficient.
    pub coefficient: f64,
    /// Inline child #0 (`DagNodeId.0`), or pool offset if `arity > 2`.
    pub child0: u32,
    /// Inline child #1, or unused (`u32::MAX`) if `arity > 2` or `arity < 2`.
    pub child1: u32,
    /// For Variable: `SymbolId.0`. For Function: `FnId.0`. Else `u32::MAX`.
    pub kind_payload: u32,
    /// Number of children. `0..=2` inline; `>2` spills to `children_pool`.
    pub arity: u8,
    /// Discriminant for `SymbolKind`. See [`kind_tag`].
    pub kind_tag: u8,
    /// Operator discriminant (valid iff `kind_tag == OPERATOR`).
    pub op_tag: u8,
    /// Packed [`NodeFlags`] bits.
    pub flags: u8,
}

const _: () = assert!(core::mem::size_of::<PackedDagNode>() == 32);
const _: () = assert!(core::mem::align_of::<PackedDagNode>() == 8);

// SAFETY: `#[repr(C)]`, all-`Copy` fields, no padding (verified by the
// static asserts above), no references, no NonZero, alignment 8 ≤ 8.
unsafe impl Pod for PackedDagNode {}

impl PackedDagNode {
    /// Reconstructs the rich [`SymbolKind`] from the packed tags.
    ///
    /// Returns `None` if the tags are out of range — which can only
    /// happen when reading a corrupt buffer.
    #[must_use]
    pub const fn kind(&self) -> Option<SymbolKind> {
        match self.kind_tag {
            kind_tag::VARIABLE => Some(SymbolKind::Variable(SymbolId(self.kind_payload))),
            kind_tag::CONSTANT => Some(SymbolKind::Constant),
            kind_tag::OPERATOR => match self.op_tag {
                op_tag::ADD => Some(SymbolKind::Operator(OpKind::Add)),
                op_tag::SUB => Some(SymbolKind::Operator(OpKind::Sub)),
                op_tag::MUL => Some(SymbolKind::Operator(OpKind::Mul)),
                op_tag::DIV => Some(SymbolKind::Operator(OpKind::Div)),
                op_tag::POW => Some(SymbolKind::Operator(OpKind::Pow)),
                op_tag::NEG => Some(SymbolKind::Operator(OpKind::Neg)),
                op_tag::MOD => Some(SymbolKind::Operator(OpKind::Mod)),
                _ => None,
            },
            kind_tag::FUNCTION => Some(SymbolKind::Function(FnId(self.kind_payload))),
            _ => None,
        }
    }

    /// The numeric value for `Constant` nodes; `None` for everything else.
    #[must_use]
    pub const fn value(&self) -> Option<f64> {
        if self.kind_tag == kind_tag::CONSTANT {
            Some(self.coefficient)
        } else {
            None
        }
    }

    /// Reconstructs the metadata bundle.
    #[must_use]
    pub const fn meta(&self) -> NodeMetadata {
        NodeMetadata {
            hash: NodeHash(self.hash),
            coefficient: if self.kind_tag == kind_tag::CONSTANT {
                // For constant nodes the user-facing coefficient is 1.0;
                // the actual constant value is exposed via `value()`.
                1.0
            } else {
                self.coefficient
            },
            arity: self.arity as u16,
            flags: NodeFlags::from_bits(self.flags),
        }
    }
}

// =========================================================================
// Owned image: rich arena → packed bytes
// =========================================================================

/// Owned packed representation of a [`DagArena`].
///
/// Built by [`PackedArenaImage::from_arena`]. The owned `Vec`s here are
/// what get encoded; the decoded counterpart is [`BorrowedArenaView`].
#[derive(Debug, Clone, Default)]
pub struct PackedArenaImage {
    nodes: Vec<PackedDagNode>,
    children_pool: Vec<u32>,
}

impl PackedArenaImage {
    /// Packs an existing rich arena into the wire form.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_arena(arena: &DagArena) -> Self {
        let n = arena.len();
        let mut nodes: Vec<PackedDagNode> = Vec::with_capacity(n);
        let mut children_pool: Vec<u32> = Vec::new();

        for i in 0..n {
            let id = DagNodeId::new(i as u32);
            // Indexing through `get` keeps us robust if `len()` and the
            // underlying vec ever diverge (currently they don't).
            let node = arena
                .get(id)
                .map_or_else(empty_placeholder, pack_one_node_ref);

            let packed = match node {
                PackOne::Inline(p) => p,
                PackOne::WithPool(mut p, extra) => {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        p.child0 = children_pool.len() as u32;
                    }
                    // For large-arity nodes (arity > 254, stored as 255 in u8)
                    // we write the true count as the first pool entry so the
                    // decoder can reconstruct the exact slice without guessing.
                    if p.arity == 255 {
                        #[allow(clippy::cast_possible_truncation)]
                        children_pool.push(extra.len() as u32);
                    }
                    children_pool.extend(extra.iter().map(|id| id.value()));
                    p
                }
            };
            nodes.push(packed);
        }

        Self {
            nodes,
            children_pool,
        }
    }

    /// Number of packed nodes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the image holds zero nodes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Borrow the packed node array.
    #[must_use]
    pub const fn nodes(&self) -> &[PackedDagNode] {
        self.nodes.as_slice()
    }

    /// Borrow the children overflow pool.
    #[must_use]
    pub const fn children_pool(&self) -> &[u32] {
        self.children_pool.as_slice()
    }

    /// Encodes the image to an 8-byte aligned byte buffer using
    /// [`crate::zerocopy::zerocopy_config`].
    ///
    /// # Errors
    ///
    /// Propagates any `bincode_next` encode error.
    pub fn encode(&self) -> Result<AlignedBytes, EncodeError> {
        let view = SerializableView {
            nodes: BorrowedSlice::new(&self.nodes),
            children_pool: BorrowedSlice::new(&self.children_pool),
        };
        encode_zerocopy(view)
    }

    /// Streams the encoded image directly into `writer`, bypassing any
    /// intermediate `AlignedBytes` allocation.
    ///
    /// Use this for spilling to disk (`BufWriter<File>`) where the
    /// in-process alignment guarantee is irrelevant — the file will be
    /// mmap-decoded (page-aligned) on the next load.
    ///
    /// # Errors
    ///
    /// Propagates any `bincode_next` encode error or I/O error.
    pub fn write_to<W: std::io::Write>(&self, writer: W) -> Result<(), EncodeError> {
        let view = SerializableView {
            nodes: BorrowedSlice::new(&self.nodes),
            children_pool: BorrowedSlice::new(&self.children_pool),
        };
        struct IoWriter<W: std::io::Write>(W);
        impl<W: std::io::Write> BincodeWriter for IoWriter<W> {
            fn write(&mut self, bytes: &[u8]) -> Result<(), EncodeError> {
                self.0
                    .write_all(bytes)
                    .map_err(|e| EncodeError::OtherString(alloc::format!("I/O error: {e}")))
            }
        }
        bincode_next::encode_into_writer(view, &mut IoWriter(writer), crate::zerocopy::zerocopy_config())
    }
}

enum PackOne {
    Inline(PackedDagNode),
    WithPool(PackedDagNode, Vec<DagNodeId>),
}

fn pack_one_node_ref(node: &DagNode) -> PackOne {
    let mut packed = PackedDagNode {
        hash: node.meta.hash.0,
        coefficient: if matches!(node.kind, SymbolKind::Constant) {
            node.value.unwrap_or(0.0)
        } else {
            node.meta.coefficient
        },
        child0: u32::MAX,
        child1: u32::MAX,
        kind_payload: u32::MAX,
        arity: 0,
        kind_tag: 0,
        op_tag: 0,
        flags: node.meta.flags.bits(),
    };

    match node.kind {
        SymbolKind::Variable(sym) => {
            packed.kind_tag = kind_tag::VARIABLE;
            packed.kind_payload = sym.0;
        }
        SymbolKind::Constant => {
            packed.kind_tag = kind_tag::CONSTANT;
        }
        SymbolKind::Operator(op) => {
            packed.kind_tag = kind_tag::OPERATOR;
            packed.op_tag = match op {
                OpKind::Add => op_tag::ADD,
                OpKind::Sub => op_tag::SUB,
                OpKind::Mul => op_tag::MUL,
                OpKind::Div => op_tag::DIV,
                OpKind::Pow => op_tag::POW,
                OpKind::Neg => op_tag::NEG,
                OpKind::Mod => op_tag::MOD,
            };
        }
        SymbolKind::Function(fn_id) => {
            packed.kind_tag = kind_tag::FUNCTION;
            packed.kind_payload = fn_id.0;
        }
    }

    let children = node.children.as_slice();
    let arity = children.len();
    // `arity` is stored in a u8; saturate at 255 — any node with more
    // children is exceedingly rare and we mark its arity = 255 while
    // still spilling all children to the pool (length kept in pool).
    #[allow(clippy::cast_possible_truncation)]
    {
        packed.arity = if arity > 255 { 255 } else { arity as u8 };
    }

    match arity {
        0 => PackOne::Inline(packed),
        1 => {
            packed.child0 = children[0].value();
            PackOne::Inline(packed)
        }
        2 => {
            packed.child0 = children[0].value();
            packed.child1 = children[1].value();
            PackOne::Inline(packed)
        }
        _ => PackOne::WithPool(packed, children.to_vec()),
    }
}

const fn empty_placeholder() -> PackOne {
    PackOne::Inline(PackedDagNode {
        hash: 0,
        coefficient: 0.0,
        child0: u32::MAX,
        child1: u32::MAX,
        kind_payload: u32::MAX,
        arity: 0,
        kind_tag: kind_tag::CONSTANT,
        op_tag: 0,
        flags: 0,
    })
}

// =========================================================================
// Encode/Decode of the image header
// =========================================================================

/// Internal helper that serialises the two side arrays back-to-back.
struct SerializableView<'a> {
    nodes: BorrowedSlice<'a, PackedDagNode>,
    children_pool: BorrowedSlice<'a, u32>,
}

impl Encode for SerializableView<'_> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.nodes.encode(encoder)?;
        self.children_pool.encode(encoder)?;
        Ok(())
    }
}

// =========================================================================
// Borrowed view: bytes → arena
// =========================================================================

/// Zero-copy view of a previously-encoded [`PackedArenaImage`].
///
/// The byte buffer (typically an [`crate::zerocopy::MmapBuffer`]) owns the
/// data; `BorrowedArenaView` carries only borrowed slices.
#[derive(Debug, Clone, Copy)]
pub struct BorrowedArenaView<'a> {
    nodes: BorrowedSlice<'a, PackedDagNode>,
    children_pool: BorrowedSlice<'a, u32>,
}

impl<'a> BorrowedArenaView<'a> {
    /// Borrow-decode an arena image from `bytes`.
    ///
    /// # Errors
    ///
    /// Propagates any `bincode_next` decode error.
    pub fn decode(bytes: &'a AlignedBytes) -> Result<Self, DecodeError> {
        decode_zerocopy::<Self>(bytes)
    }

    /// All packed nodes in arena order.
    #[must_use]
    pub const fn nodes(&self) -> &'a [PackedDagNode] {
        self.nodes.as_slice()
    }

    /// Looks up a node by [`DagNodeId`].
    ///
    /// Returns `None` for `DagNodeId::NONE` or out-of-range ids.
    #[must_use]
    pub fn get(&self, id: DagNodeId) -> Option<&'a PackedDagNode> {
        if id.is_none() {
            return None;
        }
        self.nodes.as_slice().get(id.index())
    }

    /// The children of `node`, projected back to a slice of `DagNodeId`.
    ///
    /// For inline-arity nodes the returned slice borrows from a small
    /// stack array we reconstruct on demand; for pool-spilled nodes it
    /// borrows directly from the `children_pool`. Returns a small
    /// `[DagNodeId; 0..=2]` array reborrowed via [`Children`].
    #[must_use]
    pub fn children(&self, node: &PackedDagNode) -> Children<'a> {
        let arity = node.arity as usize;
        match arity {
            0 => Children::Inline {
                ids: [DagNodeId::NONE; 2],
                len: 0,
            },
            1 => Children::Inline {
                ids: [DagNodeId::new(node.child0), DagNodeId::NONE],
                len: 1,
            },
            2 => Children::Inline {
                ids: [DagNodeId::new(node.child0), DagNodeId::new(node.child1)],
                len: 2,
            },
            _ => {
                let pool = self.children_pool.as_slice();
                let start = node.child0 as usize;
                if arity == 255 {
                    // Large-arity node: the true count is stored as the
                    // first u32 in the pool at `start`, followed by that
                    // many child ids. This is the only correct way to
                    // support multiple large-arity nodes in one image —
                    // the previous "pool end" heuristic broke as soon as
                    // any other node followed in the pool.
                    let Some(&count) = pool.get(start) else {
                        return Children::Pool(&[]);
                    };
                    let data_start = start + 1;
                    let data_end = data_start + count as usize;
                    let slice = pool.get(data_start..data_end).unwrap_or(&[]);
                    Children::Pool(slice)
                } else {
                    let slice = pool.get(start..start + arity).unwrap_or(&[]);
                    Children::Pool(slice)
                }
            }
        }
    }

    /// Number of nodes in the view.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the view holds zero nodes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Materializes the borrowed view back into an owned [`DagArena`].
    ///
    /// Used by callers (the disk-cache restore path in particular)
    /// that need the rich in-memory representation rather than the
    /// packed wire form.
    #[must_use]
    pub fn to_owned_arena(&self) -> DagArena {
        let mut arena = DagArena::new();
        for packed in self.nodes() {
            let kind = packed.kind().unwrap_or(SymbolKind::Constant);
            let meta = NodeMetadata {
                hash: NodeHash(packed.hash),
                coefficient: if matches!(kind, SymbolKind::Constant) {
                    1.0
                } else {
                    packed.coefficient
                },
                arity: packed.arity as u16,
                flags: NodeFlags::from_bits(packed.flags),
            };
            let kids: Vec<DagNodeId> = self.children(packed).iter().collect();
            let children = ChildList::from_slice(&kids);
            let value = packed.value();
            let node = match kind {
                SymbolKind::Constant => {
                    DagNode::constant(value.unwrap_or(0.0), meta)
                }
                SymbolKind::Variable(_) => DagNode::variable(kind, meta),
                SymbolKind::Operator(_) | SymbolKind::Function(_) => {
                    DagNode::operator(kind, meta, children)
                }
            };
            arena.alloc(node);
        }
        arena
    }
}

impl<'de> BorrowDecode<'de, ()> for BorrowedArenaView<'de> {
    fn borrow_decode<D: BorrowDecoder<'de, Context = ()>>(
        decoder: &mut D,
    ) -> Result<Self, DecodeError> {
        let nodes = BorrowedSlice::<PackedDagNode>::borrow_decode(decoder)?;
        let children_pool = BorrowedSlice::<u32>::borrow_decode(decoder)?;
        Ok(Self {
            nodes,
            children_pool,
        })
    }
}

/// Result of [`BorrowedArenaView::children`].
///
/// `Inline` borrows from a tiny on-stack array built fresh per query;
/// `Pool` borrows directly into the arena's `children_pool`. Both
/// expose the same `as_slice()` shape so callers don't care which.
#[derive(Debug, Clone, Copy)]
pub enum Children<'a> {
    /// Up to two inline children. `ids[..len]` are valid.
    Inline {
        /// Backing array; only the first `len` entries are meaningful.
        ids: [DagNodeId; 2],
        /// Number of valid entries.
        len: usize,
    },
    /// Children spilled to the pool.
    Pool(&'a [u32]),
}

impl<'a> Children<'a> {
    /// Number of children.
    #[must_use]
    pub const fn len(&self) -> usize {
        match self {
            Self::Inline { len, .. } => *len,
            Self::Pool(s) => s.len(),
        }
    }

    /// Whether there are no children.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate children as [`DagNodeId`].
    #[must_use]
    pub fn iter(&self) -> ChildIter<'a> {
        match self {
            Self::Inline { ids, len } => ChildIter::Inline {
                ids: *ids,
                len: *len,
                idx: 0,
            },
            Self::Pool(slice) => ChildIter::Pool(slice.iter()),
        }
    }
}

impl<'a> IntoIterator for &Children<'a> {
    type Item = DagNodeId;
    type IntoIter = ChildIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator returned by [`Children::iter`].
pub enum ChildIter<'a> {
    /// Walks an inline pair.
    Inline {
        /// Underlying inline ids.
        ids: [DagNodeId; 2],
        /// Number of valid entries.
        len: usize,
        /// Cursor.
        idx: usize,
    },
    /// Walks the pool slice.
    Pool(core::slice::Iter<'a, u32>),
}

impl Iterator for ChildIter<'_> {
    type Item = DagNodeId;

    fn next(&mut self) -> Option<DagNodeId> {
        match self {
            Self::Inline { ids, len, idx } => {
                if *idx >= *len {
                    return None;
                }
                let id = ids[*idx];
                *idx += 1;
                Some(id)
            }
            Self::Pool(it) => it.next().copied().map(DagNodeId::new),
        }
    }
}

// =========================================================================
// Helper: rebuild a `ChildList` from a `Children`
// =========================================================================

/// Materializes a [`ChildList`] from the borrowed children — useful when
/// the rich path needs the legacy enum.
#[must_use]
pub fn children_to_child_list(children: Children<'_>) -> ChildList {
    let ids: Vec<DagNodeId> = children.iter().collect();
    ChildList::from_slice(&ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::builder::DagBuilder;

    #[test]
    fn packed_node_is_32_bytes() {
        assert_eq!(core::mem::size_of::<PackedDagNode>(), 32);
        assert_eq!(core::mem::align_of::<PackedDagNode>(), 8);
    }

    #[test]
    fn pack_then_view_roundtrip() {
        // Build: (a + b) * 2.5
        let mut builder = DagBuilder::new();
        let a = builder.variable("a");
        let b = builder.variable("b");
        let sum = builder.add(a, b);
        let coeff = builder.constant(2.5);
        let _root = builder.mul(sum, coeff);

        let image = PackedArenaImage::from_arena(builder.arena());
        assert_eq!(image.len(), builder.arena().len());

        let bytes = image.encode().expect("encode");
        let view = BorrowedArenaView::decode(&bytes).expect("decode");

        assert_eq!(view.len(), image.len());

        // Walk every node and confirm the high-level reconstructions match.
        for i in 0..view.len() {
            let id = DagNodeId::new(i as u32);
            let original = builder.arena().get(id).expect("rich node");
            let packed = view.get(id).expect("packed node");
            assert_eq!(packed.kind(), Some(original.kind), "kind mismatch at {i}");
            assert_eq!(packed.value(), original.value, "value mismatch at {i}");
            assert_eq!(packed.meta().hash, original.meta.hash);
            assert_eq!(packed.meta().arity, original.meta.arity);
            let kids: Vec<DagNodeId> = view.children(packed).iter().collect();
            assert_eq!(kids.as_slice(), original.children.as_slice());
        }
    }

    #[test]
    fn variadic_children_spill_to_pool() {
        // Build a function-like operator with 5 children to force the
        // pool path.
        let mut builder = DagBuilder::new();
        let a = builder.variable("a");
        let b = builder.variable("b");
        let c = builder.variable("c");
        let d = builder.variable("d");
        let e = builder.variable("e");
        let big = builder.operator(
            SymbolKind::Function(FnId(0)),
            &[a, b, c, d, e],
            NodeFlags::EMPTY,
        );

        let image = PackedArenaImage::from_arena(builder.arena());
        let bytes = image.encode().expect("encode");
        let view = BorrowedArenaView::decode(&bytes).expect("decode");

        let node = view.get(big).expect("get big");
        assert_eq!(node.arity, 5);
        let kids: Vec<DagNodeId> = view.children(node).iter().collect();
        assert_eq!(kids, alloc::vec![a, b, c, d, e]);
        // child0 must point into the pool (not be u32::MAX).
        assert_ne!(node.child0, u32::MAX);
    }

    #[test]
    fn decoded_nodes_borrow_from_buffer() {
        let mut builder = DagBuilder::new();
        let x = builder.variable("x");
        let _ = builder.mul(x, x);
        let image = PackedArenaImage::from_arena(builder.arena());
        let bytes = image.encode().expect("encode");
        let view = BorrowedArenaView::decode(&bytes).expect("decode");

        let buf_start = bytes.as_bytes().as_ptr() as usize;
        let buf_end = buf_start + bytes.as_bytes().len();
        let nodes_start = view.nodes().as_ptr() as usize;
        assert!(
            nodes_start >= buf_start && nodes_start < buf_end,
            "BorrowedArenaView.nodes() does not point into the input \
             buffer — zero-copy invariant violated"
        );
    }
}
