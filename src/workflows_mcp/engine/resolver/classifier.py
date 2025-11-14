"""
Expression classification for optimal routing.

This module classifies expressions to determine the most efficient evaluation
strategy. Different expression types are routed to different Jinja2 methods:
    - Pure variables: compile_expression (preserves types)
    - Templates: from_string (always returns string)
    - Complex expressions: compile_expression (full evaluation)

Expression Types:
    LITERAL: No template markers ({{...}})
    PURE_VARIABLE: Single variable reference
    FILTER_EXPRESSION: Variable with filter(s)
    BOOLEAN_EXPRESSION: Boolean logic (and, or, not)
    MATH_EXPRESSION: Mathematical operations
    TEMPLATE: Mixed text and variables
    COMPLEX_EXPRESSION: Function calls, parentheses, etc.
"""

import re
from enum import Enum


class ExpressionType(Enum):
    """Types of expressions for optimal routing."""

    LITERAL = "literal"  # No {{}} markers
    PURE_VARIABLE = "pure_variable"  # {{blocks.foo}}
    FILTER_EXPRESSION = "filter"  # {{value | default('x')}}
    BOOLEAN_EXPRESSION = "boolean"  # {{a > 10 and b < 5}}
    MATH_EXPRESSION = "math"  # {{(a * 2) + b}}
    TEMPLATE = "template"  # Mixed text and {{}}
    COMPLEX_EXPRESSION = "complex"  # Combination of above


class ExpressionClassifier:
    """
    Classify expressions to route to appropriate handlers.

    Classification Process:
    1. Check for template markers ({{...}})
    2. If single expression, analyze inner content
    3. Route based on expression type

    Example:
        classifier = ExpressionClassifier()
        expr_type = classifier.classify("{{blocks.foo.succeeded}}")
        # Returns: ExpressionType.PURE_VARIABLE
    """

    # Regex patterns for classification
    VARIABLE_PATTERN = re.compile(r"^[a-zA-Z_][\w\.\[\]\'\"]*$")
    FILTER_PATTERN = re.compile(r"\s*\|")
    BOOLEAN_OPS = {"and", "or", "not", "==", "!=", ">", "<", ">=", "<=", "is", "in"}
    MATH_OPS = {"+", "-", "*", "/", "//", "%", "**"}

    def classify(self, expression: str) -> ExpressionType:
        """
        Classify expression type for optimal handling.

        Args:
            expression: Expression string to classify

        Returns:
            ExpressionType enum value
        """
        # No template markers
        if "{{" not in expression or "}}" not in expression:
            return ExpressionType.LITERAL

        # Check if pure expression (single {{...}})
        stripped = expression.strip()
        if stripped.startswith("{{") and stripped.endswith("}}"):
            if stripped.count("{{") == 1 and stripped.count("}}") == 1:
                # Extract inner expression
                inner = stripped[2:-2].strip()
                return self._classify_inner(inner)

        # Multiple {{}} or mixed with text
        return ExpressionType.TEMPLATE

    def _classify_inner(self, inner: str) -> ExpressionType:
        """
        Classify the inner content of {{...}}.

        Args:
            inner: Inner expression content (without {{...}})

        Returns:
            ExpressionType enum value
        """
        # Has filter pipe
        if "|" in inner:
            return ExpressionType.FILTER_EXPRESSION

        # Has boolean operators
        tokens = self._tokenize(inner)
        if any(op in tokens for op in self.BOOLEAN_OPS):
            return ExpressionType.BOOLEAN_EXPRESSION

        # Has math operators (but not inside strings)
        if any(op in inner for op in self.MATH_OPS):
            # Check if operators are inside string literals
            if not self._is_inside_string(inner):
                return ExpressionType.MATH_EXPRESSION

        # Has parentheses (function call or grouping)
        if "(" in inner and ")" in inner:
            return ExpressionType.COMPLEX_EXPRESSION

        # Simple variable reference
        if self.VARIABLE_PATTERN.match(inner):
            return ExpressionType.PURE_VARIABLE

        # Default to complex
        return ExpressionType.COMPLEX_EXPRESSION

    def _tokenize(self, expr: str) -> list[str]:
        """
        Simple tokenization for operator detection.

        Args:
            expr: Expression to tokenize

        Returns:
            List of tokens
        """
        # Split on word boundaries while preserving operators
        tokens = re.findall(r"\w+|[<>=!]+|\S", expr)
        return tokens

    def _is_inside_string(self, expr: str) -> bool:
        """
        Check if operators appear inside string literals.

        This is a simplified check. Real implementation would need
        proper string parsing with escape sequence handling.

        Args:
            expr: Expression to check

        Returns:
            True if expression appears to be inside strings
        """
        # Count quotes to detect if we're inside strings
        # This is simplified - just checks if there are balanced quotes
        single_quotes = expr.count("'") - expr.count("\\'")
        double_quotes = expr.count('"') - expr.count('\\"')

        # If we have balanced quotes, operators might be inside
        return (single_quotes % 2 == 0 and single_quotes > 0) or (
            double_quotes % 2 == 0 and double_quotes > 0
        )
