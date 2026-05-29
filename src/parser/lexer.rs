//! Lexer helper functions for expression parsing.

use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::{alpha1, alphanumeric1, char as nom_char, multispace0};
use nom::combinator::recognize;
use nom::error::ParseError as NomParseError;
use nom::multi::many0_count;
use nom::number::complete::double;
use nom::sequence::{pair, preceded};
use nom::{IResult, Parser};

/// Strips leading whitespace before the inner parser.
///
/// Only leading whitespace is consumed; trailing whitespace is left for
/// the next token's `ws` call. Stripping both ends (via `delimited`) caused
/// the lexer to consume whitespace that belongs to the following token,
/// masking "unexpected token" errors — `parser_review §1.2`.
pub fn ws<'a, F, O, E>(mut inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: FnMut(&'a str) -> IResult<&'a str, O, E>,
    E: NomParseError<&'a str>,
{
    move |input| preceded(multispace0, &mut inner).parse(input)
}

/// Parses a numeric constant as an `f64`.
///
/// # Errors
///
/// Returns `nom::Err::Error` if the input does not start with a valid
/// floating-point literal.
pub fn parse_constant(input: &str) -> IResult<&str, f64, nom::error::Error<&str>> {
    double.parse(input)
}

/// Parses a character operator with a concrete error type to assist compiler inference.
pub fn parse_char(c: char) -> impl Fn(&str) -> IResult<&str, char, nom::error::Error<&str>> {
    move |input| nom_char(c).parse(input)
}

/// Parses an identifier (starts with an alphabetic character or `_`, followed by alphanumeric or `_`).
///
/// # Errors
///
/// Returns `nom::Err::Error` if the input does not start with a valid
/// identifier character (`[a-zA-Z_][a-zA-Z0-9_]*`).
pub fn parse_identifier(input: &str) -> IResult<&str, &str, nom::error::Error<&str>> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0_count(alt((alphanumeric1, tag("_")))),
    ))
    .parse(input)
}

/// Parses a variable node.
///
/// # Errors
///
/// Returns `nom::Err::Error` if the input does not start with a valid
/// identifier (delegates to [`parse_identifier`]).
pub fn parse_variable(input: &str) -> IResult<&str, &str, nom::error::Error<&str>> {
    parse_identifier(input)
}
