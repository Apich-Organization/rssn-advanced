//! Per-symbol commutativity permission flags.
//!
//! Provides explicit control over which symbols support commutativity,
//! allowing the parallel splitter to safely partition expressions.

use std::collections::HashSet;
use std::sync::RwLock;
use crate::dag::symbol::SymbolId;

/// Manages explicit control over which symbols support commutativity.
#[derive(Debug, Default)]
pub struct SymbolPermissions {
    // Thread-safe set of SymbolIds that have commutativity enabled.
    commutative_symbols: RwLock<HashSet<SymbolId>>,
}

impl SymbolPermissions {
    /// Creates a new `SymbolPermissions` manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            commutative_symbols: RwLock::new(HashSet::new()),
        }
    }

    /// Sets whether a symbol supports the commutative property.
    ///
    /// # Panics
    /// Panics if the internal lock is poisoned.
    pub fn set_commutative(&self, sym: SymbolId, commutative: bool) {
        let mut guard = self.commutative_symbols.write().expect("Permissions lock poisoned");
        if commutative {
            guard.insert(sym);
        } else {
            guard.remove(&sym);
        }
    }

    /// Checks if a symbol supports the commutative property.
    ///
    /// # Panics
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn is_commutative(&self, sym: SymbolId) -> bool {
        let guard = self.commutative_symbols.read().expect("Permissions lock poisoned");
        guard.contains(&sym)
    }
}
