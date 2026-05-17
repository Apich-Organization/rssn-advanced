//! Precedence-climbing expression parser.

use super::error::ParseError;
use super::lexer::{parse_char, parse_constant, parse_variable, ws};
use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use nom::IResult;

// Returns the precedence of an operator. Higher number means higher precedence.
fn op_precedence(op: char) -> Option<u8> {
    match op {
        '+' | '-' => Some(1),
        '*' | '/' => Some(2),
        '^' => Some(3),
        _ => None,
    }
}

// Returns true if the operator is right-associative (e.g., exponentiation `^`).
fn op_right_associative(op: char) -> bool {
    op == '^'
}

fn parse_atom<'a>(input: &'a str, builder: &mut DagBuilder) -> IResult<&'a str, DagNodeId, nom::error::Error<&'a str>> {
    // 1. Parenthesized expression
    if let Ok((rem, _)) = ws(parse_char('('))(input) {
        let (rem, expr) = parse_expr_climbing(rem, builder, 0)?;
        let (rem, _) = ws(parse_char(')'))(rem)?;
        return Ok((rem, expr));
    }

    // 2. Unary minus
    if let Ok((rem, _)) = ws(parse_char('-'))(input) {
        let (rem, atom) = parse_expr_climbing(rem, builder, 4)?;
        let neg = builder.neg(atom);
        return Ok((rem, neg));
    }

    // 3. Constant
    if let Ok((rem, val)) = ws(parse_constant)(input) {
        let node_id = builder.constant(val);
        return Ok((rem, node_id));
    }

    // 4. Variable
    if let Ok((rem, name)) = ws(parse_variable)(input) {
        let node_id = builder.variable(name);
        return Ok((rem, node_id));
    }

    Err(nom::Err::Error(nom::error::Error::new(
        input,
        nom::error::ErrorKind::Fail,
    )))
}

fn parse_expr_climbing<'a>(
    input: &'a str,
    builder: &mut DagBuilder,
    min_prec: u8,
) -> IResult<&'a str, DagNodeId, nom::error::Error<&'a str>> {
    let (mut rem, mut lhs) = parse_atom(input, builder)?;

    loop {
        // Look ahead for infix operator
        let next_input = rem;
        let mut chars = next_input.trim_start().chars();
        let Some(op_char) = chars.next() else {
            break;
        };

        let Some(op_prec) = op_precedence(op_char) else {
            break;
        };

        if op_prec < min_prec {
            break;
        }

        // Consume the operator
        let (rem_after_op, _) = ws(parse_char(op_char))(rem)?;
        rem = rem_after_op;

        // Next minimum precedence is op_prec + 1 for left-associative, or op_prec for right-associative
        let next_min_prec = if op_right_associative(op_char) {
            op_prec
        } else {
            op_prec + 1
        };

        let (rem_after_rhs, rhs) = parse_expr_climbing(rem, builder, next_min_prec)?;
        rem = rem_after_rhs;

        // Combine lhs and rhs using builder
        lhs = match op_char {
            '+' => builder.add(lhs, rhs),
            '-' => builder.sub(lhs, rhs),
            '*' => builder.mul(lhs, rhs),
            '/' => builder.div(lhs, rhs),
            '^' => builder.pow(lhs, rhs),
            _ => unreachable!(),
        };
    }

    Ok((rem, lhs))
}

/// Parses a mathematical string expression (e.g. `"x^2 - 2*x + 1"`) into the global DAG.
///
/// # Errors
/// Returns a [`ParseError`] if there are syntax errors or unexpected trailing tokens.
pub fn parse_expression(input: &str, builder: &mut DagBuilder) -> Result<DagNodeId, ParseError> {
    match parse_expr_climbing(input, builder, 0) {
        Ok((remaining, id)) => {
            if !remaining.trim().is_empty() {
                return Err(ParseError {
                    message: "Unexpected trailing tokens".to_owned(),
                    span: remaining.to_owned(),
                });
            }
            Ok(id)
        }
        Err(e) => Err(ParseError {
            message: format!("Parser failed: {:?}", e),
            span: input.to_owned(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::symbol::SymbolKind;

    #[test]
    fn test_parse_expression_precedence() {
        let mut builder = DagBuilder::new();

        // Build: x + y * z
        let id = parse_expression("x + y * z", &mut builder).unwrap();
        let node = builder.arena().get(id).unwrap();
        assert_eq!(node.meta.arity, 2);
        assert_eq!(node.kind, SymbolKind::Operator(crate::dag::symbol::OpKind::Add));

        // Build: (x + y) * z
        let id2 = parse_expression("(x + y) * z", &mut builder).unwrap();
        let node2 = builder.arena().get(id2).unwrap();
        assert_eq!(node2.kind, SymbolKind::Operator(crate::dag::symbol::OpKind::Mul));
    }

    #[test]
    fn test_parse_exponentiation_right_associative() {
        let mut builder = DagBuilder::new();

        // Build: x ^ y ^ z (which is x ^ (y ^ z))
        let id = parse_expression("x ^ y ^ z", &mut builder).unwrap();
        let node = builder.arena().get(id).unwrap();
        assert_eq!(node.kind, SymbolKind::Operator(crate::dag::symbol::OpKind::Pow));
    }
}
