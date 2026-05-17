//! Parse error types with span information.

use core::fmt;

/// An error that occurred during symbolic expression parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// A human-readable description of the error.
    pub message: String,
    /// The input substring where the error occurred.
    pub span: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Parse error: {} at '{}'", self.message, self.span)
    }
}

impl std::error::Error for ParseError {}
