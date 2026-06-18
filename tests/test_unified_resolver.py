"""Unit tests for the variable resolution subsystem.

The resolver sits on the critical interpolation path: the orchestrator runs it
to substitute ``{{...}}`` expressions into plain values before strict-typed
input validators coerce them (the deliberate load-time/execute-time two-phase
design). These tests exercise the pipeline through its public entry point
``resolve_async`` — classification, the surviving transformation rules, and
Jinja2 evaluation — plus the rule and classifier units that back it.
"""

import pytest

from workflows_mcp.engine.resolver import (
    RuleType,
    UnifiedVariableResolver,
)
from workflows_mcp.engine.resolver.classifier import ExpressionClassifier, ExpressionType
from workflows_mcp.engine.resolver.rules import RuleContext
from workflows_mcp.engine.resolver.security_rules import (
    ForbiddenNamespaceRule,
    SecretRedactionRule,
)
from workflows_mcp.engine.resolver.syntax_rules import DotNotationNormalizationRule
from workflows_mcp.engine.secrets.exceptions import SecretNotFoundError
from workflows_mcp.engine.secrets.provider import SecretProvider


class _DictSecretProvider(SecretProvider):
    """In-memory secret provider for exercising secret resolution."""

    def __init__(self, secrets: dict[str, str]):
        self._secrets = secrets

    async def get_secret(self, key: str) -> str:
        try:
            return self._secrets[key]
        except KeyError as exc:
            raise SecretNotFoundError(key) from exc

    async def list_secret_keys(self) -> list[str]:
        return list(self._secrets)


class TestExpressionClassifier:
    """Classification routes each expression to the right Jinja2 strategy."""

    def setup_method(self) -> None:
        self.classifier = ExpressionClassifier()

    def test_literal_has_no_markers(self) -> None:
        assert self.classifier.classify("plain text") == ExpressionType.LITERAL

    def test_pure_variable(self) -> None:
        assert self.classifier.classify("{{inputs.name}}") == ExpressionType.PURE_VARIABLE

    def test_filter_expression(self) -> None:
        assert (
            self.classifier.classify("{{value | default('x')}}") == ExpressionType.FILTER_EXPRESSION
        )

    def test_boolean_expression(self) -> None:
        assert self.classifier.classify("{{a > 10 and b < 5}}") == ExpressionType.BOOLEAN_EXPRESSION

    def test_math_expression(self) -> None:
        assert self.classifier.classify("{{count + 1}}") == ExpressionType.MATH_EXPRESSION

    def test_template_with_surrounding_text(self) -> None:
        assert self.classifier.classify("Hello {{name}}!") == ExpressionType.TEMPLATE

    def test_template_with_multiple_expressions(self) -> None:
        assert self.classifier.classify("{{a}}-{{b}}") == ExpressionType.TEMPLATE


class TestTransformRules:
    """The two surviving rule types and their transforms in isolation."""

    def test_rule_type_enum_has_only_surviving_members(self) -> None:
        assert {member.name for member in RuleType} == {"SYNTAX", "SECURITY"}

    def test_forbidden_namespace_rule_is_security(self) -> None:
        rule = ForbiddenNamespaceRule()
        assert rule.rule_type is RuleType.SECURITY
        ctx = RuleContext(expression="{{__internal__.token}}", context={})
        assert rule.applies_to(ctx)
        with pytest.raises(Exception, match="forbidden"):
            rule.transform(ctx)

    def test_forbidden_namespace_rule_ignores_safe_expression(self) -> None:
        rule = ForbiddenNamespaceRule()
        assert not rule.applies_to(RuleContext(expression="{{inputs.value}}", context={}))

    def test_secret_redaction_rule_marks_metadata(self) -> None:
        rule = SecretRedactionRule()
        assert rule.rule_type is RuleType.SECURITY
        ctx = RuleContext(expression="{{secrets.API_KEY}}-{{secrets.TOKEN}}", context={})
        assert rule.applies_to(ctx)
        transformed = rule.transform(ctx)
        assert transformed.metadata["contains_secrets"] is True
        assert transformed.metadata["secret_keys"] == ["API_KEY", "TOKEN"]

    def test_dot_notation_rule_is_syntax_and_brackets_hyphens(self) -> None:
        rule = DotNotationNormalizationRule()
        assert rule.rule_type is RuleType.SYNTAX
        ctx = RuleContext(expression="{{blocks.foo-bar.outputs.value}}", context={})
        assert rule.applies_to(ctx)
        transformed = rule.transform(ctx)
        assert "blocks['foo-bar']" in transformed.expression

    def test_dot_notation_rule_ignores_plain_identifiers(self) -> None:
        rule = DotNotationNormalizationRule()
        assert not rule.applies_to(RuleContext(expression="{{blocks.foo.outputs}}", context={}))


class TestResolveAsync:
    """End-to-end resolution through the public entry point."""

    async def test_non_string_values_pass_through(self) -> None:
        resolver = UnifiedVariableResolver({})
        assert await resolver.resolve_async(42) == 42
        assert await resolver.resolve_async(True) is True
        assert await resolver.resolve_async(None) is None

    async def test_literal_returned_unchanged(self) -> None:
        resolver = UnifiedVariableResolver({})
        assert await resolver.resolve_async("no markers here") == "no markers here"

    async def test_pure_variable_preserves_type(self) -> None:
        resolver = UnifiedVariableResolver({"inputs": {"count": 7}})
        result = await resolver.resolve_async("{{inputs.count}}")
        assert result == 7
        assert isinstance(result, int)

    async def test_pure_variable_preserves_dict(self) -> None:
        features = {"a": True, "b": False}
        resolver = UnifiedVariableResolver({"inputs": {"features": features}})
        assert await resolver.resolve_async("{{inputs.features}}") == features

    async def test_boolean_expression_evaluates_to_bool(self) -> None:
        resolver = UnifiedVariableResolver({"inputs": {"count": 12}})
        assert await resolver.resolve_async("{{inputs.count > 10}}") is True

    async def test_template_renders_to_string(self) -> None:
        resolver = UnifiedVariableResolver({"inputs": {"name": "world"}})
        assert await resolver.resolve_async("Hello {{inputs.name}}!") == "Hello world!"

    async def test_block_status_shortcut(self) -> None:
        context = {"blocks": {"test": {"metadata": {"succeeded": True}, "outputs": {}}}}
        resolver = UnifiedVariableResolver(context)
        assert await resolver.resolve_async("{{blocks.test.succeeded}}") is True

    async def test_control_structure_template(self) -> None:
        resolver = UnifiedVariableResolver({"inputs": {"nums": [1, 2, 3]}})
        result = await resolver.resolve_async("{% for i in inputs.nums %}{{i}}{% endfor %}")
        assert result == "123"

    async def test_nested_dict_and_list_recursion(self) -> None:
        resolver = UnifiedVariableResolver({"inputs": {"flag": True, "n": 5}})
        resolved = await resolver.resolve_async(
            {"a": "{{inputs.flag}}", "b": ["{{inputs.n}}", "literal"]}
        )
        assert resolved == {"a": True, "b": [5, "literal"]}

    async def test_dot_notation_rule_applied_during_resolution(self) -> None:
        context = {"blocks": {"foo-bar": {"outputs": {"value": "ok"}, "metadata": {}}}}
        resolver = UnifiedVariableResolver(context)
        assert await resolver.resolve_async("{{blocks.foo-bar.outputs.value}}") == "ok"

    async def test_forbidden_namespace_raises(self) -> None:
        resolver = UnifiedVariableResolver({})
        with pytest.raises(ValueError, match="Security violation"):
            await resolver.resolve_async("{{__internal__.secret}}")

    async def test_secret_resolution_with_provider(self) -> None:
        provider = _DictSecretProvider({"API_KEY": "sk-test"})
        resolver = UnifiedVariableResolver({}, secret_provider=provider)
        assert await resolver.resolve_async("{{secrets.API_KEY}}") == "sk-test"

    async def test_missing_secret_defaults_to_empty_string(self) -> None:
        provider = _DictSecretProvider({})
        resolver = UnifiedVariableResolver({}, secret_provider=provider)
        assert await resolver.resolve_async("{{secrets.ABSENT}}") == ""
