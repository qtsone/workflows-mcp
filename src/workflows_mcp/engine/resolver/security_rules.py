"""
Security transformation rules for variable resolution.

These rules enforce security boundaries and track sensitive data access.

Rules:
    - ForbiddenNamespaceRule: Block access to forbidden namespaces
    - SecretRedactionRule: Track secret access for audit logging
"""

import re

from .rules import RuleContext, RuleType, TransformRule


class SecurityError(Exception):
    """Raised when a security rule is violated."""

    pass


class ForbiddenNamespaceRule(TransformRule):
    """
    Block access to forbidden namespaces.

    Prevents access to:
    - __internal__: Internal system state
    - __builtins__: Python built-ins
    - __import__: Import system
    - Dangerous functions: exec, eval, compile, open, file
    """

    rule_type = RuleType.SECURITY
    priority = 1  # Security rules run first

    FORBIDDEN_PATTERNS = [
        "__internal__",
        "__builtins__",
        "__import__",
        "__subclasses__",
        "exec(",
        "eval(",
        "compile(",
        "open(",
        "file(",
    ]

    def applies_to(self, context: RuleContext) -> bool:
        return any(pattern in context.expression for pattern in self.FORBIDDEN_PATTERNS)

    def transform(self, context: RuleContext) -> RuleContext:
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern in context.expression:
                raise SecurityError(f"Access to '{pattern}' is forbidden in expressions")
        return context

    @property
    def description(self) -> str:
        return "Prevent access to forbidden namespaces and functions"


class SecretRedactionRule(TransformRule):
    """
    Track secret access for audit logging.

    Marks expressions containing {{secrets.*}} references so that:
    1. Audit logs can track secret access
    2. Output redaction can sanitize secret values
    3. Execution context knows when secrets are involved
    """

    rule_type = RuleType.SECURITY
    priority = 2

    def applies_to(self, context: RuleContext) -> bool:
        return "secrets." in context.expression

    def transform(self, context: RuleContext) -> RuleContext:
        # Mark expression as containing secrets for audit logging
        if not context.metadata:
            context.metadata = {}
        context.metadata["contains_secrets"] = True
        context.metadata["secret_keys"] = self._extract_secret_keys(context.expression)
        return context

    def _extract_secret_keys(self, expression: str) -> list[str]:
        """Extract secret key names for audit logging."""
        # Match secrets.KEY_NAME patterns
        pattern = r"\bsecrets\.([a-zA-Z_][a-zA-Z0-9_]*)"
        return re.findall(pattern, expression)

    @property
    def description(self) -> str:
        return "Track and protect secret access in expressions"
