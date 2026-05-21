//! Per-symbol commutativity permission flags.
//!
//! Provides explicit control over which symbols support commutativity,
//! allowing the parallel splitter to safely partition expressions.
//!
//! Lock recovery: poisoned locks pass through transparently via
//! `PoisonError::into_inner` — we don't maintain any cross-thread
//! invariant that the panic would invalidate (`storage_review §2`,
//! Phase 7 cleanup).

use std::collections::HashSet;
use std::sync::{PoisonError, RwLock};

use crate::dag::symbol::SymbolId;

/// Manages explicit control over which symbols support commutativity.
#[derive(Debug, Default)]
pub struct SymbolPermissions {
    /// Thread-safe set of `SymbolId`s that have commutativity enabled.
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
    pub fn set_commutative(&self, sym: SymbolId, commutative: bool) {
        let mut guard = self
            .commutative_symbols
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        if commutative {
            guard.insert(sym);
        } else {
            guard.remove(&sym);
        }
    }

    /// Checks if a symbol supports the commutative property.
    #[must_use]
    pub fn is_commutative(&self, sym: SymbolId) -> bool {
        let guard = self
            .commutative_symbols
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        guard.contains(&sym)
    }
}
