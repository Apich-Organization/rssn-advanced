//! Compiled JIT function cache.
//!
//! Caches JIT-compiled native functions keyed by expression string, avoiding
//! redundant recompilation of identical derivation or evaluation paths.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use super::compiler::CompiledExprFn;

/// A thread-safe, dynamic cache for storing and reusing JIT-compiled native functions.
#[derive(Debug, Clone, Default)]
pub struct JitCache {
    // Stores compiled function pointers cast as usize for thread-safe storage.
    cache: Arc<RwLock<HashMap<String, usize>>>,
}

impl JitCache {
    /// Creates a new, empty JIT function cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Attempts to retrieve a compiled function from the cache.
    #[allow(unsafe_code)]
    pub fn get(&self, key: &str) -> Option<CompiledExprFn> {
        let read_guard = self.cache.read().ok()?;
        let addr = *read_guard.get(key)?;
        #[allow(clippy::type_complexity)]
        let func: CompiledExprFn = unsafe { std::mem::transmute(addr) };
        Some(func)
    }

    /// Inserts a compiled function into the cache.
    ///
    /// # Panics
    /// Panics if the internal lock is poisoned.
    pub fn insert(&self, key: String, func: CompiledExprFn) {
        let mut write_guard = self.cache.write().expect("JIT cache lock poisoned");
        let addr = func as usize;
        write_guard.insert(key, addr);
    }

    /// Clears all compiled functions from the cache.
    ///
    /// # Panics
    /// Panics if the internal lock is poisoned.
    pub fn clear(&self) {
        let mut write_guard = self.cache.write().expect("JIT cache lock poisoned");
        write_guard.clear();
    }
}
