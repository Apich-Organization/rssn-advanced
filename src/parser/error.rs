//! Parse error types with line/column span information.
//!
//! Per `parser_review §3`, the previous `ParseError::span: String`
//! captured the raw "remaining input" suffix, which is opaque for
//! large inputs. This rewrite tracks the exact byte offset, line, and
//! column relative to the original source buffer.

use core::fmt;

/// Location of a parser error in the original source buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Byte offset from the start of the input (0-based).
    pub offset: usize,
    /// Length of the offending substring, in bytes.
    pub len: usize,
    /// 1-based line number where the error starts.
    pub line: u32,
    /// 1-based column number where the error starts.
    pub col: u32,
}

impl Span {
    /// Computes a [`Span`] from an absolute byte `offset` into `source`,
    /// counting newlines for line/col attribution.
    #[must_use]
    pub fn from_offset(source: &str, offset: usize, len: usize) -> Self {
        let clamped = offset.min(source.len());
        let prefix = &source.as_bytes()[..clamped];
        let mut line: u32 = 1;
        let mut last_newline: usize = 0;
        for (i, &b) in prefix.iter().enumerate() {
            if b == b'\n' {
                line = line.saturating_add(1);
                last_newline = i + 1;
            }
        }
        // Column = bytes since the most recent '\n', plus 1.
        #[allow(clippy::cast_possible_truncation)]
        let col = (clamped - last_newline) as u32 + 1;
        Self {
            offset,
            len,
            line,
            col,
        }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

/// An error that occurred during symbolic expression parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// A human-readable description of the error.
    pub message: String,
    /// Location of the error in the original source buffer.
    pub span: Span,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Parse error at {}: {}", self.span, self.message)
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_at_start_of_input() {
        let s = Span::from_offset("hello", 0, 1);
        assert_eq!(s.line, 1);
        assert_eq!(s.col, 1);
    }

    #[test]
    fn span_after_newline() {
        let src = "a + b\nc + d\nbad";
        // offset of the 'b' in "bad" — past two newlines.
        let off = src.find("bad").expect("find");
        let s = Span::from_offset(src, off, 3);
        assert_eq!(s.line, 3);
        assert_eq!(s.col, 1);
    }

    #[test]
    fn span_offset_past_end_clamps() {
        let s = Span::from_offset("xy", 999, 1);
        assert_eq!(s.line, 1);
        assert_eq!(s.col, 3); // After the last byte.
    }
}
