//! User-defined pattern-rewrite derivation rules.
//!
//! Allows advanced users to define custom `(pattern, replacement)` rewrite
//! rules that are compiled into native pattern-match + rewrite code via Cranelift.

/// A custom user-defined symbolic rewrite rule representing `(pattern, replacement)`.
#[derive(Debug, Clone)]
pub struct CustomRule {
    /// The target pattern string to search for (e.g. `"x + x"`).
    pub pattern: String,
    /// The replacement pattern string to insert (e.g. `"2 * x"`).
    pub replacement: String,
}

impl CustomRule {
    /// Creates a new custom rewrite rule.
    #[must_use]
    pub fn new(pattern: &str, replacement: &str) -> Self {
        Self {
            pattern: pattern.to_owned(),
            replacement: replacement.to_owned(),
        }
    }
}
