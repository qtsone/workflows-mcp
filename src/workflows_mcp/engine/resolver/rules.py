"""
Rule system foundation for variable resolution transformations.

This module provides the base classes and infrastructure for rule-based
transformations in the variable resolution pipeline. Rules are applied in
priority order to transform expressions before evaluation.

Rule Types:
    - SYNTAX: Expression syntax transformations (e.g., bracket notation)
    - SECURITY: Security validations and restrictions
    - NAMESPACE: Namespace enhancements and shortcuts
    - SHORTCUT: Attribute shortcuts (e.g., blocks.id.succeeded)

Example:
    class MyRule(TransformRule):
        rule_type = RuleType.SYNTAX
        priority = 10

        def applies_to(self, context: RuleContext) -> bool:
            return 'special_pattern' in context.expression

        def transform(self, context: RuleContext) -> RuleContext:
            context.expression = context.expression.replace('old', 'new')
            return context

        @property
        def description(self) -> str:
            return "Transforms special patterns"
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RuleType(Enum):
    """Types of transformation rules."""

    SYNTAX = "syntax"  # Expression syntax transformations
    SECURITY = "security"  # Security validations
    NAMESPACE = "namespace"  # Namespace enhancements
    SHORTCUT = "shortcut"  # Attribute shortcuts


@dataclass
class RuleContext:
    """
    Context passed to rules for processing.

    Attributes:
        expression: Expression string to transform
        context: Workflow execution context (inputs, blocks, metadata, etc.)
        metadata: Rule-specific metadata (e.g., contains_secrets flag)
    """

    expression: str
    context: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class TransformRule(ABC):
    """
    Base class for transformation rules.

    Rules are applied in priority order (lower = higher priority) to transform
    expressions before evaluation. Each rule can:
    1. Check if it applies to the current context
    2. Transform the expression
    3. Add metadata for downstream processing

    Security rules run first (priority 1-9), followed by syntax rules (10-49),
    then namespace/shortcut rules (50+).
    """

    rule_type: RuleType
    priority: int = 0  # Lower = higher priority

    @abstractmethod
    def applies_to(self, context: RuleContext) -> bool:
        """
        Check if rule applies to this context.

        Args:
            context: Current rule context with expression and workflow context

        Returns:
            True if rule should be applied
        """
        pass

    @abstractmethod
    def transform(self, context: RuleContext) -> RuleContext:
        """
        Apply transformation to context.

        Args:
            context: Current rule context

        Returns:
            Transformed rule context (may modify expression or metadata)
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this rule does."""
        pass
