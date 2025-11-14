"""
Unified variable resolver package.

This package implements a unified variable resolution system that combines
workflow-specific domain rules with Jinja2's powerful expression evaluation.

The architecture uses a clean overlay pattern with rule-based transformations:
1. Expression classification for optimal routing
2. Rule-based transformation pipeline
3. Context enhancement with proxies
4. Jinja2 evaluation with appropriate method
5. Post-processing and validation

Public API:
    - UnifiedVariableResolver: Main resolver class for all variable resolution
    - TransformRule: Base class for custom transformation rules
    - BlockProxy: Proxy for block attribute shortcuts
    - ExpressionClassifier: Expression type detection for routing
"""

from .classifier import ExpressionClassifier, ExpressionType
from .proxies import BlockProxy, ProxyBase, SecretProxy
from .rules import RuleContext, RuleType, TransformRule
from .security_rules import SecurityError
from .unified_resolver import UnifiedVariableResolver

__all__ = [
    "UnifiedVariableResolver",
    "TransformRule",
    "RuleType",
    "RuleContext",
    "BlockProxy",
    "SecretProxy",
    "ProxyBase",
    "ExpressionClassifier",
    "ExpressionType",
    "SecurityError",
]
