"""
Syntax transformation rules for variable resolution.

These rules transform expression syntax to be compatible with Jinja2 while
maintaining workflow-specific semantics.

Rules:
    - BracketNotationRule: Convert double-quote brackets to single-quote
    - DotNotationNormalizationRule: Normalize special characters in paths
"""

import re

from .rules import RuleContext, RuleType, TransformRule


class BracketNotationRule(TransformRule):
    """
    Convert double-quote brackets to single-quote for Jinja2 compatibility.

    Transforms: blocks["foo-bar"] → blocks['foo-bar']
    Reason: Jinja2 prefers single quotes for bracket notation
    """

    rule_type = RuleType.SYNTAX
    priority = 10

    def applies_to(self, context: RuleContext) -> bool:
        return '["' in context.expression and '"]' in context.expression

    def transform(self, context: RuleContext) -> RuleContext:
        # Convert ["foo"] to ['foo']
        context.expression = re.sub(
            r'\["([^"]+)"\]',
            r"['\1']",
            context.expression,
        )
        return context

    @property
    def description(self) -> str:
        return "Convert double-quote bracket notation to Jinja2-compatible single quotes"


class DotNotationNormalizationRule(TransformRule):
    """
    Normalize special characters in dot notation paths.

    Transforms: blocks.foo-bar.outputs → blocks['foo-bar'].outputs
    Reason: Identifiers with hyphens/special chars need bracket notation
    """

    rule_type = RuleType.SYNTAX
    priority = 20

    def applies_to(self, context: RuleContext) -> bool:
        # Check for paths with hyphens or other special chars
        pattern = r"blocks\.[^\s\.\[\]]+[\-][^\s\.\[\]]*"
        return bool(re.search(pattern, context.expression))

    def transform(self, context: RuleContext) -> RuleContext:
        # Convert blocks.foo-bar.outputs to blocks['foo-bar'].outputs
        def replace_special_keys(match: re.Match[str]) -> str:
            key = match.group(1)
            # Check if key needs bracket notation (contains hyphen or not a valid identifier)
            if "-" in key or not key.replace("_", "a").isidentifier():
                return f"blocks['{key}']"
            return match.group(0)

        context.expression = re.sub(
            r"blocks\.([a-zA-Z0-9_\-]+)",
            replace_special_keys,
            context.expression,
        )
        return context

    @property
    def description(self) -> str:
        return "Convert special characters in dot notation to bracket notation"
