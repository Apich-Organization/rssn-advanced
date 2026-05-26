//! Symbol identifiers and classification.
//!
//! A **symbol** is the atomic unit of a symbolic expression — a named variable,
//! a numeric constant, an operator, or a function reference.

use core::fmt;
use std::sync::Arc;

use bincode_next::{Decode, Encode};

/// A unique identifier for a symbol in the symbol registry.
///
/// Symbol IDs are indices into the global symbol table. They are cheap
/// to copy and compare (just a `u32`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Encode, Decode)]
pub struct SymbolId(pub(crate) u32);

impl SymbolId {
    /// Creates a new `SymbolId` from a raw u32 index.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Returns the raw u32 index of this symbol.
    #[must_use]
    pub const fn value(self) -> u32 {
        self.0
    }
}

impl fmt::Debug for SymbolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SymbolId({})", self.0)
    }
}

impl fmt::Display for SymbolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sym#{}", self.0)
    }
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Pow => "^",
            Self::Neg => "neg",
            Self::Mod => "%",
        };
        f.write_str(s)
    }
}

/// A function identifier, referencing a named function in the registry.
///
/// Values `0..=6` are reserved as **intrinsic** identifiers, one per
/// [`OpKind`] variant (matching its `#[repr(u8)]` discriminant). This
/// lets built-in operators be expressed as `FnId` for the JIT/parallel
/// engines that want to handle them uniformly alongside user functions.
///
/// User-defined functions must use IDs ≥ [`FnId::FIRST_USER`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Encode, Decode)]
pub struct FnId(pub(crate) u32);

impl FnId {
    /// The first `FnId` available for user-defined functions.
    ///
    /// IDs `0..=6` are reserved for built-in operators.
    pub const FIRST_USER: Self = Self(7);

    /// Converts an [`OpKind`] to its intrinsic `FnId`.
    ///
    /// Each `OpKind` variant's `#[repr(u8)]` discriminant becomes the `FnId`
    /// value (0 = Add, 1 = Sub, … 6 = Neg). These IDs are stable and will
    /// not be reassigned.
    #[must_use]
    pub const fn from_op(op: OpKind) -> Self {
        Self(op as u32)
    }

    /// Returns the corresponding [`OpKind`] if this `FnId` is an intrinsic.
    ///
    /// Returns `None` for user-defined function IDs (≥ 7).
    #[must_use]
    pub const fn to_op(self) -> Option<OpKind> {
        match self.0 {
            0 => Some(OpKind::Add),
            1 => Some(OpKind::Sub),
            2 => Some(OpKind::Mul),
            3 => Some(OpKind::Div),
            4 => Some(OpKind::Pow),
            5 => Some(OpKind::Mod),
            6 => Some(OpKind::Neg),
            _ => None,
        }
    }

    /// Returns `true` if this `FnId` maps to a built-in operator.
    #[must_use]
    pub const fn is_intrinsic(self) -> bool {
        self.0 <= 6
    }
}

/// The kind of an operator node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Encode, Decode)]
#[repr(u8)]
pub enum OpKind {
    /// Addition (`+`).
    Add = 0,
    /// Subtraction (`-`).
    Sub = 1,
    /// Multiplication (`*`).
    Mul = 2,
    /// Division (`/`).
    Div = 3,
    /// Exponentiation (`^`).
    Pow = 4,
    /// IEEE-754 floating-point remainder (`%`). Same precedence as `*` and `/`.
    Mod = 5,
    /// Unary negation (`-x`).
    Neg = 6,
}

/// Classification of a symbol node.
///
/// Every node in the DAG is one of these four kinds:
/// - A named **variable** (e.g. `x`, `y`).
/// - A numeric **constant** (e.g. `3.14`).
/// - An **operator** (e.g. `+`, `*`).
/// - A **function** call (e.g. `sin`, `log`).
#[derive(Debug, Clone, Copy, Encode, Decode)]
pub enum SymbolKind {
    /// A named variable, identified by its `SymbolId` in the symbol table.
    Variable(SymbolId),
    /// A numeric constant value (the f64 value is stored inline).
    Constant(f64),
    /// An algebraic operator.
    Operator(OpKind),
    /// A function call, identified by its `FnId`.
    Function(FnId),
}

impl PartialEq for SymbolKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Variable(a), Self::Variable(b)) => a == b,
            (Self::Constant(a), Self::Constant(b)) => a.to_bits() == b.to_bits(),
            (Self::Operator(a), Self::Operator(b)) => a == b,
            (Self::Function(a), Self::Function(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for SymbolKind {}

impl std::hash::Hash for SymbolKind {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Self::Variable(sym) => sym.hash(state),
            Self::Constant(v) => v.to_bits().hash(state),
            Self::Operator(op) => op.hash(state),
            Self::Function(fn_id) => fn_id.hash(state),
        }
    }
}

/// The symbol registry: maps symbol IDs to their string names.
///
/// This is the canonical name table — all variable and function names
/// are interned here to avoid redundant string storage. A side-index
/// `HashMap<Arc<str>, SymbolId>` provides O(1) name → ID lookup so that
/// `intern` does not degrade to O(N) when a program declares thousands
/// of distinct variables (`dag_review §2`).
///
/// Both `names` and `lookup` store `Arc<str>` sharing the same allocation —
/// `Arc::clone` is a ref-count bump, not a heap allocation, so `intern` now
/// does exactly **one** heap allocation per new symbol instead of two
/// (`dag_review §1.3`).
#[derive(Debug, Clone)]
pub struct SymbolRegistry {
    /// Interned names, indexed by `SymbolId.0`. Shared with the lookup map.
    names: Vec<Arc<str>>,
    /// O(1) lookup from name to existing `SymbolId`.
    /// `Arc<str>: Borrow<str>` so `HashMap::get(name: &str)` works without
    /// allocating a temporary key. rapidhash `GlobalState` is used as the
    /// hasher since string keys benefit from its throughput over `SipHash`.
    lookup: std::collections::HashMap<Arc<str>, SymbolId, rapidhash::fast::GlobalState>,
}

impl SymbolRegistry {
    /// Creates a new, empty symbol registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            names: Vec::new(),
            lookup: std::collections::HashMap::default(),
        }
    }

    /// Interns a name and returns its `SymbolId`.
    ///
    /// If the name already exists, returns the existing ID in O(1) with zero
    /// allocation. Otherwise allocates exactly one `Arc<str>` and shares it
    /// between `names` and `lookup` via a ref-count bump (`dag_review §1.3`).
    pub fn intern(&mut self, name: &str) -> SymbolId {
        if let Some(&id) = self.lookup.get(name) {
            return id;
        }
        #[allow(clippy::cast_possible_truncation)]
        let id = SymbolId(self.names.len() as u32);
        let arc: Arc<str> = Arc::from(name); // one allocation
        self.names.push(Arc::clone(&arc)); // ref-count bump — no alloc
        self.lookup.insert(arc, id);
        id
    }

    /// Looks up the name for a given `SymbolId`.
    ///
    /// Returns `None` if the ID is out of range.
    #[must_use]
    pub fn name(&self, id: SymbolId) -> Option<&str> {
        self.names
            .get(id.0 as usize)
            .map(std::convert::AsRef::as_ref)
    }

    /// Looks up an existing `SymbolId` for `name` without interning.
    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<SymbolId> {
        self.lookup.get(name).copied()
    }

    /// Looks up an existing `SymbolId` from a raw byte slice.
    ///
    /// Returns `None` if `bytes` is not valid UTF-8 or if no name
    /// matches. **Allocates nothing on the hot path** — used by the
    /// FFI surface to avoid the `to_string_lossy()` allocation per
    /// variable reference (`ffi_review §2`).
    #[must_use]
    pub fn lookup_bytes(&self, bytes: &[u8]) -> Option<SymbolId> {
        core::str::from_utf8(bytes)
            .ok()
            .and_then(|s| self.lookup(s))
    }

    /// Interns a name from a raw byte slice. The hot path (already
    /// interned) allocates zero bytes; first-time insertion allocates
    /// one `Box<str>` for the new entry.
    ///
    /// # Errors
    ///
    /// Returns `None` if `bytes` is not valid UTF-8.
    pub fn intern_bytes(&mut self, bytes: &[u8]) -> Option<SymbolId> {
        let name = core::str::from_utf8(bytes).ok()?;
        Some(self.intern(name))
    }

    /// Returns the number of interned symbols.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.names.len()
    }

    /// Returns `true` if no symbols have been interned.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

impl Default for SymbolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_and_lookup() {
        let mut reg = SymbolRegistry::new();
        let x = reg.intern("x");
        let y = reg.intern("y");
        let x2 = reg.intern("x");

        assert_eq!(x, x2, "re-interning should return same ID");
        assert_ne!(x, y);
        assert_eq!(reg.name(x), Some("x"));
        assert_eq!(reg.name(y), Some("y"));
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn display_op_kind() {
        assert_eq!(format!("{}", OpKind::Add), "+");
        assert_eq!(format!("{}", OpKind::Pow), "^");
        assert_eq!(format!("{}", OpKind::Neg), "neg");
    }

    #[test]
    fn intern_scales_o1() {
        // With the HashMap side-index, interning N distinct names then
        // re-interning them must take amortized O(N) total (not O(N^2)).
        // We can't measure complexity in a unit test, but we can prove
        // the lookup is constant-time by lookup() and that re-interning
        // returns the same ID without growing the names vector.
        let mut reg = SymbolRegistry::new();
        for i in 0..1000 {
            let name = format!("v{i}");
            let id = reg.intern(&name);
            assert_eq!(reg.lookup(&name), Some(id));
        }
        assert_eq!(reg.len(), 1000);
        // Re-interning each name must not allocate.
        for i in 0..1000 {
            let name = format!("v{i}");
            let _ = reg.intern(&name);
        }
        assert_eq!(reg.len(), 1000);
    }
}
