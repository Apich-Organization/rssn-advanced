//! Precedence-climbing expression parser.
//!
//! Per `parser_review §2` / `§3`:
//!
//! * Recursion is depth-capped at [`MAX_PAREN_DEPTH`] so `(((...)))`
//!   inputs cannot blow the OS stack.
//! * Errors carry a `Span` with line/column information, computed
//!   against the original source buffer (not the remaining suffix).

use nom::IResult;

use super::error::{ParseError, Span, cold_parse_error_unexpected_eof, cold_parse_error_unexpected_token};
use super::lexer::{parse_char, parse_constant, parse_identifier, ws};
use crate::dag::builder::DagBuilder;
use crate::dag::node::DagNodeId;
use crate::dag::symbol::SymbolKind;

/// Maximum allowed depth of parenthesis / operator recursion.
///
/// Each parse level requires two stack frames (`parse_expr_climbing` +
/// `parse_atom`). In debug builds these frames are large enough that
/// 1024 levels can overflow the default 8 MB thread stack. 200 keeps
/// the recursive depth ≈ 400 frames, which is safe on every target.
pub const MAX_PAREN_DEPTH: u16 = 200;

/// A runtime-extensible operator precedence table for the expression parser.
///
/// The built-in table handles `+`, `-`, `*`, `/`, `%`, and `^`. Additional
/// infix operators with custom precedence levels can be registered at runtime
/// without modifying the parser source.
///
/// Named operators (multi-character strings such as `"and"`, `"or"`, `"mod"`,
/// `"xor"`) are supported alongside single-character operators. Named operators
/// are matched as identifiers during infix parsing — after the left operand is
/// parsed, the parser checks whether the next token matches any registered
/// named operator before falling back to single-character matching.
///
/// Higher precedence numbers bind more tightly (e.g. `*` before `+`).
/// Right-associative operators (currently only `^`) are marked separately.
///
/// The `parse_with_table` function uses a `PrecedenceTable` instead of the
/// hardcoded `op_precedence` / `op_right_associative` functions.
#[derive(Debug, Clone)]
pub struct PrecedenceTable {
    /// Maps operator string → (precedence, is_right_associative).
    /// Single-character operators are stored as single-char strings.
    entries: std::collections::HashMap<String, (u8, bool)>,
    /// Maps prefix unary operator string → DAG `SymbolKind` for the node to build.
    unary: std::collections::HashMap<String, SymbolKind>,
}

impl PrecedenceTable {
    /// Creates the default table matching the built-in parser behaviour.
    #[must_use]
    pub fn default_table() -> Self {
        let mut t = Self {
            entries: std::collections::HashMap::new(),
            unary: std::collections::HashMap::new(),
        };
        t.entries.insert("+".into(), (1, false));
        t.entries.insert("-".into(), (1, false));
        t.entries.insert("*".into(), (2, false));
        t.entries.insert("/".into(), (2, false));
        t.entries.insert("%".into(), (2, false));
        t.entries.insert("^".into(), (3, true));
        t
    }

    /// Creates an empty table (no operators registered).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            unary: std::collections::HashMap::new(),
        }
    }

    /// Registers a new infix operator by name (single- or multi-character).
    ///
    /// - `op`: any string key; single chars such as `'+'` are converted to a
    ///   one-character string. Multi-char keys like `"and"` or `"mod"` are
    ///   matched as identifiers during infix parsing.
    /// - `precedence`: binding strength; higher binds tighter.
    /// - `right_associative`: `true` for right-to-left evaluation (like `^`).
    pub fn register_op(&mut self, op: impl Into<String>, precedence: u8, right_associative: bool) {
        self.entries.insert(op.into(), (precedence, right_associative));
    }

    /// Registers a single-character infix operator.
    ///
    /// Convenience alias for [`register_op`](Self::register_op) with a `char`
    /// argument; preserves backward compatibility with the previous API.
    pub fn register(&mut self, op: char, precedence: u8, right_associative: bool) {
        self.register_op(op.to_string(), precedence, right_associative);
    }

    /// Returns the precedence of a single-char operator, or `None` if not registered.
    #[must_use]
    pub fn precedence(&self, op: char) -> Option<u8> {
        self.entries.get(&op.to_string()).map(|&(prec, _)| prec)
    }

    /// Returns the precedence of any operator (single- or multi-char).
    #[must_use]
    pub fn precedence_str(&self, op: &str) -> Option<u8> {
        self.entries.get(op).map(|&(prec, _)| prec)
    }

    /// Returns `true` if a single-char operator is right-associative.
    #[must_use]
    pub fn is_right_associative(&self, op: char) -> bool {
        self.entries.get(&op.to_string()).is_some_and(|&(_, ra)| ra)
    }

    /// Returns `true` if any operator (single- or multi-char) is right-associative.
    #[must_use]
    pub fn is_right_associative_str(&self, op: &str) -> bool {
        self.entries.get(op).is_some_and(|&(_, ra)| ra)
    }

    /// Returns all registered multi-character operator names (length > 1).
    ///
    /// Used by the infix parser to attempt named-operator matching before
    /// falling back to single-character operators.
    #[must_use]
    pub fn named_ops(&self) -> impl Iterator<Item = &str> {
        self.entries.keys()
            .filter(|k| k.len() > 1)
            .map(String::as_str)
    }

    /// Registers a prefix unary operator (e.g. `"!"`, `"~"`, `"not"`).
    ///
    /// When the parser encounters this prefix in atom position, it consumes
    /// it, recursively parses the operand, and wraps it in a DAG node whose
    /// `SymbolKind` is `kind`. The prefix is matched literally from the
    /// current position (after whitespace).
    pub fn register_unary_op(&mut self, prefix: impl Into<String>, kind: SymbolKind) {
        self.unary.insert(prefix.into(), kind);
    }

    /// Iterator over all registered prefix unary operators.
    ///
    /// Yields `(prefix, kind)` pairs. Use this to inspect what custom
    /// unary operators are active without modifying the table.
    pub fn unary_ops(&self) -> impl Iterator<Item = (&str, &SymbolKind)> {
        self.unary.iter().map(|(k, v)| (k.as_str(), v))
    }
}

/// Returns the precedence of an operator. Higher number means higher precedence.
const fn op_precedence(op: char) -> Option<u8> {
    match op {
        '+' | '-' => Some(1),
        '*' | '/' | '%' => Some(2),
        '^' => Some(3),
        _ => None,
    }
}

/// Returns true if the operator is right-associative (e.g., `^`).
const fn op_right_associative(op: char) -> bool {
    op == '^'
}

/// Internal recursion-capped error sentinel. We return it via an
/// `ErrorKind::TooLarge` so `nom`'s Err path knows to propagate it.
#[doc(hidden)]
#[cold]
#[track_caller]
#[inline(never)]
fn too_deep(input: &str) -> nom::Err<nom::error::Error<&str>> {
    nom::Err::Failure(nom::error::Error::new(input, nom::error::ErrorKind::TooLarge))
}

fn parse_atom<'a>(
    input: &'a str,
    builder: &mut DagBuilder,
    depth: u16,
) -> IResult<&'a str, DagNodeId, nom::error::Error<&'a str>> {
    // 1. Parenthesized expression — bounded recursion.
    if let Ok((rem, _)) = ws(parse_char('('))(input) {
        if depth >= MAX_PAREN_DEPTH {
            return Err(too_deep(input));
        }
        let (rem, expr) = parse_expr_climbing(rem, builder, 0, depth + 1)?;
        let (rem, _) = ws(parse_char(')'))(rem)?;
        return Ok((rem, expr));
    }

    // 2. Unary minus.
    if let Ok((rem, _)) = ws(parse_char('-'))(input) {
        let (rem, atom) = parse_expr_climbing(rem, builder, 3, depth)?;
        let neg = builder.neg(atom);
        return Ok((rem, neg));
    }

    // 3. Numeric constant.
    if let Ok((rem, val)) = ws(parse_constant)(input) {
        let node_id = builder.constant(val);
        return Ok((rem, node_id));
    }

    // 4. Function call `identifier '(' args ')'` or plain variable.
    if let Ok((rem, name)) = ws(parse_identifier)(input) {
        // Peek for '(' to distinguish function call from variable.
        if let Ok((mut cur, _)) = ws(parse_char('('))(rem) {
            if depth >= MAX_PAREN_DEPTH {
                return Err(too_deep(input));
            }
            let mut args: Vec<DagNodeId> = Vec::new();
            // Handle zero-arg call: `f()`.
            if let Ok((after_close, _)) = ws(parse_char(')'))(cur) {
                cur = after_close;
            } else {
                loop {
                    let (r, arg) = parse_expr_climbing(cur, builder, 0, depth + 1)?;
                    args.push(arg);
                    cur = r;
                    if let Ok((r2, _)) = ws(parse_char(','))(cur) {
                        cur = r2;
                    } else if let Ok((r2, _)) = ws(parse_char(')'))(cur) {
                        cur = r2;
                        break;
                    } else {
                        return Err(nom::Err::Error(nom::error::Error::new(
                            cur,
                            nom::error::ErrorKind::Tag, // "expected ',' or ')'"
                        )));
                    }
                }
            }
            let fn_id = builder.intern_function(name);
            let node_id = builder.function_call(fn_id, &args);
            return Ok((cur, node_id));
        }
        // Plain variable.
        let node_id = builder.variable(name);
        return Ok((rem, node_id));
    }

    Err(nom::Err::Error(nom::error::Error::new(
        input,
        nom::error::ErrorKind::Char, // "unexpected character"
    )))
}

fn parse_expr_climbing<'a>(
    input: &'a str,
    builder: &mut DagBuilder,
    min_prec: u8,
    depth: u16,
) -> IResult<&'a str, DagNodeId, nom::error::Error<&'a str>> {
    // Guard against deep right-associative chains (e.g. `x^y^z^...`)
    // which recurse here for each `^` — independent of paren depth.
    if depth >= MAX_PAREN_DEPTH {
        return Err(too_deep(input));
    }

    let (mut rem, mut lhs) = parse_atom(input, builder, depth)?;

    loop {
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

        // Consume the operator.
        let (rem_after_op, _) = ws(parse_char(op_char))(rem)?;
        rem = rem_after_op;

        // For right-associative operators increment depth to cap chains
        // like `x^y^z^...` which otherwise bypass the paren depth check.
        let next_min_prec = if op_right_associative(op_char) {
            op_prec
        } else {
            op_prec + 1
        };
        let next_depth = if op_right_associative(op_char) {
            depth.saturating_add(1)
        } else {
            depth
        };

        let (rem_after_rhs, rhs) = parse_expr_climbing(rem, builder, next_min_prec, next_depth)?;
        rem = rem_after_rhs;

        // Combine lhs and rhs using the builder (every constructor
        // hash-conses through DedupMap, so the precedence climber
        // can't accidentally produce duplicates).
        lhs = match op_char {
            '+' => builder.add(lhs, rhs),
            '-' => builder.sub(lhs, rhs),
            '*' => builder.mul(lhs, rhs),
            '/' => builder.div(lhs, rhs),
            '%' => builder.modulo(lhs, rhs),
            '^' => builder.pow(lhs, rhs),
            _ => return Err(too_deep(input)),
        };
    }

    Ok((rem, lhs))
}

// ---------------------------------------------------------------------------
// Table-driven parsing (public API)
// ---------------------------------------------------------------------------

fn parse_atom_with_table<'a>(
    input: &'a str,
    builder: &mut DagBuilder,
    depth: u16,
    table: &PrecedenceTable,
) -> IResult<&'a str, DagNodeId, nom::error::Error<&'a str>> {
    // 1. Parenthesized expression — bounded recursion.
    if let Ok((rem, _)) = ws(parse_char('('))(input) {
        if depth >= MAX_PAREN_DEPTH {
            return Err(too_deep(input));
        }
        let (rem, expr) = parse_expr_climbing_with_table(rem, builder, 0, depth + 1, table)?;
        let (rem, _) = ws(parse_char(')'))(rem)?;
        return Ok((rem, expr));
    }

    // 2. Unary minus.
    if let Ok((rem, _)) = ws(parse_char('-'))(input) {
        let (rem, atom) = parse_expr_climbing_with_table(rem, builder, 3, depth, table)?;
        let neg = builder.neg(atom);
        return Ok((rem, neg));
    }

    // 2b. User-registered prefix unary operators (e.g. "!", "~", "not").
    // Sort by prefix length descending so longer prefixes take priority.
    {
        let trimmed = input.trim_start();
        let mut candidates: Vec<(&str, &SymbolKind)> = table.unary_ops().collect();
        candidates.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        for (prefix, kind) in candidates {
            if trimmed.starts_with(prefix) {
                let after = &trimmed[prefix.len()..];
                // For single-char prefixes accept any position; for
                // multi-char word prefixes ensure a word boundary.
                let ok = prefix.chars().next().is_some_and(|c| !c.is_alphanumeric() && c != '_')
                    || after.is_empty()
                    || !after.chars().next().is_some_and(|c| c.is_alphanumeric() || c == '_');
                if ok {
                    let (rem, atom) =
                        parse_expr_climbing_with_table(after, builder, 3, depth, table)?;
                    use crate::dag::metadata::NodeFlags;
                    let node_id = builder.operator(*kind, &[atom], NodeFlags::EMPTY);
                    return Ok((rem, node_id));
                }
            }
        }
    }

    // 3. Numeric constant.
    if let Ok((rem, val)) = ws(parse_constant)(input) {
        let node_id = builder.constant(val);
        return Ok((rem, node_id));
    }

    // 4. Function call `identifier '(' args ')'` or plain variable.
    if let Ok((rem, name)) = ws(parse_identifier)(input) {
        if let Ok((mut cur, _)) = ws(parse_char('('))(rem) {
            if depth >= MAX_PAREN_DEPTH {
                return Err(too_deep(input));
            }
            let mut args: Vec<DagNodeId> = Vec::new();
            if let Ok((after_close, _)) = ws(parse_char(')'))(cur) {
                cur = after_close;
            } else {
                loop {
                    let (r, arg) =
                        parse_expr_climbing_with_table(cur, builder, 0, depth + 1, table)?;
                    args.push(arg);
                    cur = r;
                    if let Ok((r2, _)) = ws(parse_char(','))(cur) {
                        cur = r2;
                    } else if let Ok((r2, _)) = ws(parse_char(')'))(cur) {
                        cur = r2;
                        break;
                    } else {
                        return Err(nom::Err::Error(nom::error::Error::new(
                            cur,
                            nom::error::ErrorKind::Tag,
                        )));
                    }
                }
            }
            let fn_id = builder.intern_function(name);
            let node_id = builder.function_call(fn_id, &args);
            return Ok((cur, node_id));
        }
        let node_id = builder.variable(name);
        return Ok((rem, node_id));
    }

    Err(nom::Err::Error(nom::error::Error::new(
        input,
        nom::error::ErrorKind::Char,
    )))
}

fn parse_expr_climbing_with_table<'a>(
    input: &'a str,
    builder: &mut DagBuilder,
    min_prec: u8,
    depth: u16,
    table: &PrecedenceTable,
) -> IResult<&'a str, DagNodeId, nom::error::Error<&'a str>> {
    if depth >= MAX_PAREN_DEPTH {
        return Err(too_deep(input));
    }

    let (mut rem, mut lhs) = parse_atom_with_table(input, builder, depth, table)?;

    loop {
        let trimmed = rem.trim_start();

        // Try named operators (multi-char, matched as identifiers) first.
        // We sort by length descending to prefer longer matches ("and" over "a").
        let mut named_match: Option<(&str, u8, bool, &str)> = None; // (op_name, prec, ra, rem_after)
        {
            let mut candidates: Vec<&str> = table.named_ops().collect();
            candidates.sort_by(|a, b| b.len().cmp(&a.len()));
            for named_op in candidates {
                if trimmed.starts_with(named_op) {
                    // Ensure the match is a complete identifier token (not a prefix of a longer word).
                    let after = &trimmed[named_op.len()..];
                    let is_word_boundary = after.is_empty()
                        || !after.chars().next().is_some_and(|c| c.is_alphanumeric() || c == '_');
                    if is_word_boundary {
                        if let Some(prec) = table.precedence_str(named_op) {
                            let ra = table.is_right_associative_str(named_op);
                            named_match = Some((named_op, prec, ra, after.trim_start()));
                            break;
                        }
                    }
                }
            }
        }

        if let Some((op_name, op_prec, ra, rem_after_op)) = named_match {
            if op_prec < min_prec {
                break;
            }
            rem = rem_after_op;
            let next_min_prec = if ra { op_prec } else { op_prec + 1 };
            let next_depth = if ra { depth.saturating_add(1) } else { depth };
            let (rem_after_rhs, rhs) =
                parse_expr_climbing_with_table(rem, builder, next_min_prec, next_depth, table)?;
            rem = rem_after_rhs;
            // Named operators beyond the built-ins become FunctionCall nodes with two children.
            let fn_id = builder.intern_function(op_name);
            lhs = builder.function_call(fn_id, &[lhs, rhs]);
            continue;
        }

        // Single-character operator path.
        let mut chars = trimmed.chars();
        let Some(op_char) = chars.next() else {
            break;
        };

        let Some(op_prec) = table.precedence(op_char) else {
            break;
        };

        if op_prec < min_prec {
            break;
        }

        let (rem_after_op, _) = ws(parse_char(op_char))(rem)?;
        rem = rem_after_op;

        let ra = table.is_right_associative(op_char);
        let next_min_prec = if ra { op_prec } else { op_prec + 1 };
        let next_depth = if ra { depth.saturating_add(1) } else { depth };

        let (rem_after_rhs, rhs) =
            parse_expr_climbing_with_table(rem, builder, next_min_prec, next_depth, table)?;
        rem = rem_after_rhs;

        lhs = match op_char {
            '+' => builder.add(lhs, rhs),
            '-' => builder.sub(lhs, rhs),
            '*' => builder.mul(lhs, rhs),
            '/' => builder.div(lhs, rhs),
            '%' => builder.modulo(lhs, rhs),
            '^' => builder.pow(lhs, rhs),
            _ => return Err(too_deep(input)),
        };
    }

    Ok((rem, lhs))
}

/// Parses `input` using a caller-supplied [`PrecedenceTable`].
///
/// This is the runtime-extensible counterpart to [`parse_expression`].  Where
/// [`parse_expression`] always uses the six built-in operators (`+`, `-`, `*`,
/// `/`, `%`, `^`), this function consults `table` for every infix token it
/// encounters.  Custom operators registered via [`PrecedenceTable::register`]
/// are fully supported.
///
/// # Errors
///
/// Returns a [`ParseError`] on syntax errors, unexpected trailing tokens, or
/// paren-depth overflow — identical semantics to [`parse_expression`].
pub fn parse_with_table(
    input: &str,
    builder: &mut DagBuilder,
    table: &PrecedenceTable,
) -> Result<DagNodeId, super::error::ParseError> {
    match parse_expr_climbing_with_table(input, builder, 0, 0, table) {
        Ok((remaining, id)) => {
            let trimmed = remaining.trim_start();
            if !trimmed.is_empty() {
                let offset = offset_in(input, trimmed).unwrap_or(input.len());
                return Err(super::error::ParseError {
                    message: "Unexpected trailing tokens".to_owned(),
                    span: super::error::Span::from_offset(input, offset, trimmed.len()),
                });
            }
            Ok(id)
        }
        Err(nom::Err::Error(e) | nom::Err::Failure(e)) => {
            let offset = offset_in(input, e.input).unwrap_or(input.len());
            let len = e.input.len().min(input.len().saturating_sub(offset));
            let span = super::error::Span::from_offset(input, offset, len);
            match e.code {
                nom::error::ErrorKind::TooLarge => Err(super::error::ParseError {
                    message: format!("Parenthesis depth exceeded {MAX_PAREN_DEPTH}"),
                    span,
                }),
                nom::error::ErrorKind::Char => {
                    if e.input.trim_start().is_empty() {
                        Err(cold_parse_error_unexpected_eof(span))
                    } else {
                        let bad = e.input.trim_start().chars().next().unwrap_or('?');
                        Err(cold_parse_error_unexpected_token(span, bad))
                    }
                }
                nom::error::ErrorKind::Tag => Err(super::error::ParseError {
                    message: "Expected ',' or ')' to close function argument list".to_owned(),
                    span,
                }),
                nom::error::ErrorKind::Eof => Err(super::error::ParseError {
                    message: "Unexpected end of input; expression is incomplete".to_owned(),
                    span,
                }),
                _ => Err(super::error::ParseError {
                    message: format!("Syntax error near {:?}", &e.input[..e.input.len().min(8)]),
                    span,
                }),
            }
        }
        Err(nom::Err::Incomplete(_)) => Err(super::error::ParseError {
            message: "Incomplete input".to_owned(),
            span: super::error::Span::from_offset(input, input.len(), 0),
        }),
    }
}

/// Byte offset of `slice` within `whole`, or `None` if `slice` isn't
/// a substring view of `whole` (different allocation).
fn offset_in(whole: &str, slice: &str) -> Option<usize> {
    let whole_ptr = whole.as_ptr() as usize;
    let slice_ptr = slice.as_ptr() as usize;
    let whole_end = whole_ptr.checked_add(whole.len())?;
    if slice_ptr < whole_ptr || slice_ptr > whole_end {
        return None;
    }
    Some(slice_ptr - whole_ptr)
}

/// Parses a mathematical string expression into the global DAG.
///
/// # Errors
///
/// Returns a [`ParseError`] (with line/column [`Span`]) on syntax
/// errors, unexpected trailing tokens, or paren-depth overflow.
pub fn parse_expression(input: &str, builder: &mut DagBuilder) -> Result<DagNodeId, ParseError> {
    match parse_expr_climbing(input, builder, 0, 0) {
        Ok((remaining, id)) => {
            let trimmed = remaining.trim_start();
            if !trimmed.is_empty() {
                let offset = offset_in(input, trimmed).unwrap_or(input.len());
                return Err(ParseError {
                    message: "Unexpected trailing tokens".to_owned(),
                    span: Span::from_offset(input, offset, trimmed.len()),
                });
            }
            Ok(id)
        }
        Err(nom::Err::Error(e) | nom::Err::Failure(e)) => {
            let offset = offset_in(input, e.input).unwrap_or(input.len());
            let len = e.input.len().min(input.len().saturating_sub(offset));
            let span = Span::from_offset(input, offset, len);
            match e.code {
                nom::error::ErrorKind::TooLarge => Err(ParseError {
                    message: format!("Parenthesis depth exceeded {MAX_PAREN_DEPTH}"),
                    span,
                }),
                nom::error::ErrorKind::Char => {
                    // `nom_char(c)` emits Char on mismatch. When input is
                    // exhausted it means a required character (e.g. `)`) was
                    // never found; when input has a character it is unexpected.
                    if e.input.trim_start().is_empty() {
                        Err(cold_parse_error_unexpected_eof(span))
                    } else {
                        let bad = e.input.trim_start().chars().next().unwrap_or('?');
                        Err(cold_parse_error_unexpected_token(span, bad))
                    }
                }
                nom::error::ErrorKind::Tag => Err(ParseError {
                    message: "Expected ',' or ')' to close function argument list".to_owned(),
                    span,
                }),
                nom::error::ErrorKind::Eof => Err(ParseError {
                    message: "Unexpected end of input; expression is incomplete".to_owned(),
                    span,
                }),
                _ => Err(ParseError {
                    message: format!("Syntax error near {:?}", &e.input[..e.input.len().min(8)]),
                    span,
                }),
            }
        }
        Err(nom::Err::Incomplete(_)) => Err(ParseError {
            message: "Incomplete input".to_owned(),
            span: Span::from_offset(input, input.len(), 0),
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

        let id = parse_expression("x + y * z", &mut builder).expect("ok");
        let node = builder.arena().get(id).expect("root");
        assert_eq!(node.children.len(), 2);
        assert_eq!(
            node.kind,
            SymbolKind::Operator(crate::dag::symbol::OpKind::Add)
        );

        let id2 = parse_expression("(x + y) * z", &mut builder).expect("ok");
        let node2 = builder.arena().get(id2).expect("root");
        assert_eq!(
            node2.kind,
            SymbolKind::Operator(crate::dag::symbol::OpKind::Mul)
        );
    }

    #[test]
    fn test_parse_exponentiation_right_associative() {
        let mut builder = DagBuilder::new();
        let id = parse_expression("x ^ y ^ z", &mut builder).expect("ok");
        let node = builder.arena().get(id).expect("root");
        assert_eq!(
            node.kind,
            SymbolKind::Operator(crate::dag::symbol::OpKind::Pow)
        );
    }

    #[test]
    fn paren_depth_overflow_is_a_clean_error() {
        // Use a depth well above MAX_PAREN_DEPTH (200) but small enough
        // that even in debug builds we never actually recurse that deep
        // (the error fires at depth 200, long before 400).
        let n: usize = 400;
        let mut input = String::with_capacity(n * 2 + 1);
        for _ in 0..n {
            input.push('(');
        }
        input.push('1');
        for _ in 0..n {
            input.push(')');
        }
        let mut b = DagBuilder::new();
        let err = parse_expression(&input, &mut b).expect_err("must error");
        assert!(err.message.contains("depth"));
    }

    #[test]
    fn function_call_parses_correctly() {
        let mut b = DagBuilder::new();
        let id = parse_expression("sin(x)", &mut b).expect("ok");
        let node = b.arena().get(id).expect("root");
        assert!(matches!(node.kind, crate::dag::symbol::SymbolKind::Function(_)));
        assert_eq!(node.children.len(), 1);
    }

    #[test]
    fn function_call_with_multiple_args() {
        let mut b = DagBuilder::new();
        let id = parse_expression("pow(x, y)", &mut b).expect("ok");
        let node = b.arena().get(id).expect("root");
        assert!(matches!(node.kind, crate::dag::symbol::SymbolKind::Function(_)));
        assert_eq!(node.children.len(), 2);
    }

    #[test]
    fn function_call_nested_in_expression() {
        let mut b = DagBuilder::new();
        let id = parse_expression("sin(x) + cos(y)", &mut b).expect("ok");
        let node = b.arena().get(id).expect("root");
        assert!(matches!(
            node.kind,
            crate::dag::symbol::SymbolKind::Operator(crate::dag::symbol::OpKind::Add)
        ));
    }

    #[test]
    fn unary_minus_has_lower_precedence_than_power() {
        // Standard math: `-x^y` = `-(x^y)`, not `(-x)^y`.
        // With x=2, y=3: -(2^3) = -8, not (-2)^3 = -8 (same here but different for even exponents).
        // Use y=2 to distinguish: -(2^2) = -4, but (-2)^2 = +4.
        let mut b = DagBuilder::new();
        let id = parse_expression("-x^y", &mut b).expect("ok");
        // Root should be Neg, not Pow.
        let node = b.arena().get(id).expect("root");
        assert_eq!(
            node.kind,
            crate::dag::symbol::SymbolKind::Operator(crate::dag::symbol::OpKind::Neg),
            "-x^y must parse as -(x^y), root should be Neg"
        );
        // The child of Neg must be Pow.
        let child_id = node.children.as_slice()[0];
        let child = b.arena().get(child_id).expect("child");
        assert_eq!(
            child.kind,
            crate::dag::symbol::SymbolKind::Operator(crate::dag::symbol::OpKind::Pow)
        );
    }

    #[test]
    fn span_carries_line_col_on_error() {
        let src = "x +\n+ ";
        let mut b = DagBuilder::new();
        let err = parse_expression(src, &mut b).expect_err("trailing junk");
        // The error region begins past the first newline.
        assert!(err.span.line >= 1);
        assert!(err.span.col >= 1);
    }
}
