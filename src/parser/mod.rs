//! Symbolic expression parser.
//!
//! Parses textual mathematical expressions into the global DAG. Uses
//! `nom` combinators with precedence climbing for correct operator
//! binding.
//!
//! - `lexer` — Tokenizer for numbers, identifiers, operators, parens.
//! - `expr` — Precedence-climbing expression parser.
//! - `error` — Span-annotated parse error types.

pub mod error;
pub mod expr;
pub mod lexer;

pub use expr::{PrecedenceTable, parse_expression, parse_with_table};
