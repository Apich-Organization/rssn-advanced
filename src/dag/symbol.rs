//! Symbol identifiers and classification.
//!
//! A **symbol** is the atomic unit of a symbolic expression — a named variable,
//! a numeric constant, an operator, or a function reference.

use core::fmt;

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

/// The kind of an operator node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Encode, Decode)]
pub enum OpKind {
    /// Addition (`+`).
    Add,
    /// Subtraction (`-`).
    Sub,
    /// Multiplication (`*`).
    Mul,
    /// Division (`/`).
    Div,
    /// Exponentiation (`^`).
    Pow,
    /// Unary negation (`-x`).
    Neg,
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
        };
        f.write_str(s)
    }
}

/// A function identifier, referencing a named function in the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Encode, Decode)]
pub struct FnId(pub(crate) u32);

/// Classification of a symbol node.
///
/// Every node in the DAG is one of these four kinds:
/// - A named **variable** (e.g. `x`, `y`).
/// - A numeric **constant** (e.g. `3.14`).
/// - An **operator** (e.g. `+`, `*`).
/// - A **function** call (e.g. `sin`, `log`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Encode, Decode)]
pub enum SymbolKind {
    /// A named variable, identified by its `SymbolId` in the symbol table.
    Variable(SymbolId),
    /// A numeric constant value.
    Constant,
    /// An algebraic operator.
    Operator(OpKind),
    /// A function call, identified by its `FnId`.
    Function(FnId),
}

/// The symbol registry: maps symbol IDs to their string names.
///
/// This is the canonical name table — all variable and function names
/// are interned here to avoid redundant string storage.
#[derive(Debug, Clone)]
pub struct SymbolRegistry {
    /// Interned names, indexed by `SymbolId.0`.
    names: Vec<String>,
}

impl SymbolRegistry {
    /// Creates a new, empty symbol registry.
    #[must_use]
    pub fn new() -> Self {
        Self { names: Vec::new() }
    }

    /// Interns a name and returns its `SymbolId`.
    ///
    /// If the name already exists, returns the existing ID.
    /// Otherwise, allocates a new slot.
    pub fn intern(&mut self, name: &str) -> SymbolId {
        // Linear scan is fine for typical expression sizes (< 1000 symbols).
        // For larger workloads, consider a HashMap<String, SymbolId> side-index.
        for (i, existing) in self.names.iter().enumerate() {
            if existing == name {
                #[allow(clippy::cast_possible_truncation)]
                return SymbolId(i as u32);
            }
        }
        let id = self.names.len();
        self.names.push(name.to_owned());
        #[allow(clippy::cast_possible_truncation)]
        SymbolId(id as u32)
    }

    /// Looks up the name for a given `SymbolId`.
    ///
    /// Returns `None` if the ID is out of range.
    #[must_use]
    pub fn name(&self, id: SymbolId) -> Option<&str> {
        self.names.get(id.0 as usize).map(String::as_str)
    }

    /// Returns the number of interned symbols.
    #[must_use]
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Returns `true` if no symbols have been interned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
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
}
