//! Integration tests for the expression parser.

#[cfg(test)]
mod parser_tests {
    use rssn_advanced::dag::builder::DagBuilder;
    use rssn_advanced::parser::expr::parse_expression;

    #[test]
    fn test_parse_complex_formula() {
        let mut builder = DagBuilder::new();

        // Parse: -((a * b) ^ 2) / c
        let parsed_id = parse_expression("-((a * b) ^ 2.0) / c", &mut builder).unwrap();

        // Build manually to verify identical structural sharing
        let mut manual_builder = DagBuilder::new();
        let a = manual_builder.variable("a");
        let b = manual_builder.variable("b");
        let c = manual_builder.variable("c");
        let two = manual_builder.constant(2.0);

        let mul = manual_builder.mul(a, b);
        let power = manual_builder.pow(mul, two);
        let negation = manual_builder.neg(power);
        let manual_root = manual_builder.div(negation, c);

        // Verify identical structure reconverted and same node properties
        let parsed_node = builder.arena().get(parsed_id).unwrap();
        let manual_node = manual_builder.arena().get(manual_root).unwrap();

        assert_eq!(parsed_node.kind, manual_node.kind);
        assert_eq!(parsed_node.children.len(), manual_node.children.len());
        assert_eq!(
            builder.arena().len(),
            8,
            "Expected exactly 8 nodes in the parsed DAG"
        );
    }

    #[test]
    fn test_parse_error_handling() {
        let mut builder = DagBuilder::new();

        // 1. Missing closing parenthesis — should mention end of input or ')'
        let err1 = parse_expression("(x + y", &mut builder).unwrap_err();
        assert!(
            err1.message.contains("end of input") || err1.message.contains("')'"),
            "missing ')' error should mention end-of-input or ')': {:?}",
            err1.message
        );

        // 2. Invalid character / syntax — trailing junk after valid expression
        let err2 = parse_expression("x @ y", &mut builder).unwrap_err();
        assert!(
            err2.message.contains("trailing") || err2.message.contains("Unexpected"),
            "invalid char error should mention trailing or unexpected: {:?}",
            err2.message
        );

        // 3. Consecutive operators — unexpected character where operand expected
        let err3 = parse_expression("x ++ y", &mut builder).unwrap_err();
        assert!(
            err3.message.contains("Unexpected") || err3.message.contains("Syntax"),
            "consecutive ops error should mention unexpected or syntax: {:?}",
            err3.message
        );
    }

    #[test]
    fn test_whitespace_and_formatting() {
        let mut builder = DagBuilder::new();

        // Parse with excessive whitespace, newlines, and tabs
        let parsed_id = parse_expression("  \n\t x   + \t\n  y  ", &mut builder).unwrap();

        let x = builder.variable("x");
        let y = builder.variable("y");
        let expected_add = builder.add(x, y);

        assert_eq!(
            parsed_id, expected_add,
            "Whitespace stripping failed to parse equivalence"
        );
    }

    #[test]
    fn test_operator_precedence_complex() {
        let mut builder = DagBuilder::new();

        // x + y * z ^ w should be parsed as x + (y * (z ^ w))
        let parsed_id = parse_expression("x + y * z ^ w", &mut builder).unwrap();

        let x = builder.variable("x");
        let y = builder.variable("y");
        let z = builder.variable("z");
        let w = builder.variable("w");

        let power = builder.pow(z, w);
        let product = builder.mul(y, power);
        let expected = builder.add(x, product);

        assert_eq!(
            parsed_id, expected,
            "Operator precedence climbing failed for power and product combinations"
        );
    }
}
