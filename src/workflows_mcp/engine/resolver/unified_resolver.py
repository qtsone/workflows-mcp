"""
Unified variable resolver with rule-based transformation pipeline.

This module implements the main resolver that orchestrates:
1. Expression classification for optimal routing
2. Rule-based transformation pipeline
3. Context enhancement with proxies
4. Jinja2 evaluation with appropriate method
5. Post-processing and validation

Architecture:
    User Expression
          ↓
    Expression Classifier
          ↓
    Transform Pipeline (Rules)
          ↓
    Context Enhancement (Proxies)
          ↓
    Jinja2 Engine
          ↓
    Post-Processing

Example:
    resolver = UnifiedVariableResolver(context)
    result = await resolver.resolve_async("{{blocks.foo.succeeded}}")
"""

import asyncio
import base64
import hashlib
import json
import shlex
from datetime import datetime
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

from .classifier import ExpressionClassifier, ExpressionType
from .proxies import BlockProxy, SecretProxy
from .rules import RuleContext, TransformRule
from .security_rules import ForbiddenNamespaceRule, SecretRedactionRule, SecurityError
from .syntax_rules import BracketNotationRule, DotNotationNormalizationRule

if TYPE_CHECKING:
    from ..secrets import SecretAuditLog, SecretProvider


class UnifiedVariableResolver:
    """
    Unified variable resolver with rule-based transformation pipeline.

    Architecture:
    1. Expression classification for optimal routing
    2. Rule-based transformation pipeline
    3. Context enhancement with proxies
    4. Jinja2 evaluation with appropriate method
    5. Post-processing and validation

    Example:
        resolver = UnifiedVariableResolver(context)

        # Synchronous (no secrets)
        result = resolver.resolve("{{blocks.foo.exit_code}}")

        # Asynchronous (with secrets)
        result = await resolver.resolve_async("{{secrets.API_KEY}}")
    """

    def __init__(
        self,
        context: dict[str, Any],
        rules: list[TransformRule] | None = None,
        secret_provider: "SecretProvider | None" = None,
        audit_log: "SecretAuditLog | None" = None,
        safe_mode: bool = True,
        workflow_name: str = "",
        block_id: str = "",
    ):
        """
        Initialize unified variable resolver.

        Args:
            context: Workflow context (inputs, blocks, metadata, etc.)
            rules: Optional custom transformation rules
            secret_provider: Optional secret provider for {{secrets.*}}
            audit_log: Optional audit log for secret access
            safe_mode: Use sandboxed Jinja2 environment (default: True)
            workflow_name: Workflow name for audit logging
            block_id: Block ID for audit logging
        """
        self.raw_context = context
        self.secret_provider = secret_provider
        self.audit_log = audit_log
        self.safe_mode = safe_mode
        self.workflow_name = workflow_name
        self.block_id = block_id

        # Initialize components
        self.classifier = ExpressionClassifier()
        self.rules = self._initialize_rules(rules)

        # Set up Jinja2 environment
        self.env: Environment
        if safe_mode:
            self.env = SandboxedEnvironment(
                undefined=StrictUndefined,
                autoescape=False,
                trim_blocks=False,
                lstrip_blocks=False,
            )
        else:
            self.env = Environment(
                undefined=StrictUndefined,
                autoescape=False,
            )

        # Register custom filters and functions
        self._register_extensions()

        # Prepare enhanced context
        self.jinja_context = self._enhance_context(context)

    def _initialize_rules(self, custom_rules: list[TransformRule] | None) -> list[TransformRule]:
        """
        Initialize transformation rules in priority order.

        Args:
            custom_rules: Optional custom rules to add

        Returns:
            Sorted list of rules (by priority)
        """
        # Default rules
        default_rules = [
            ForbiddenNamespaceRule(),  # Security first
            SecretRedactionRule(),
            BracketNotationRule(),  # Syntax transformation
            DotNotationNormalizationRule(),
        ]

        # Merge with custom rules
        all_rules = default_rules + (custom_rules or [])

        # Sort by priority (lower number = higher priority)
        return sorted(all_rules, key=lambda r: r.priority)

    def _register_extensions(self) -> None:
        """Register custom filters and functions in Jinja2 environment."""
        # Filters
        self.env.filters.update(
            {
                "quote": shlex.quote,
                "prettyjson": lambda x: json.dumps(x, indent=2),
                "tojson": json.dumps,
                "b64encode": lambda x: base64.b64encode(x.encode()).decode(),
                "b64decode": lambda x: base64.b64decode(x).decode(),
                "hash": lambda x, algo="sha256": hashlib.new(algo, x.encode()).hexdigest(),
                "keys": lambda x: list(x.keys()) if isinstance(x, dict) else [],
                "values": lambda x: list(x.values()) if isinstance(x, dict) else [],
            }
        )

        # Global functions
        self.env.globals.update(
            {
                "now": lambda: datetime.now().isoformat(),
                "timestamp": lambda: int(datetime.now().timestamp()),
                "len": len,
                "range": range,
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "get": self._get,
                "render": self._render,
            }
        )

    @staticmethod
    def _get(obj: Any, key: int | str, default: Any = None) -> Any:
        """
        Unified safe accessor for dicts, lists, and attributes.

        Similar to dict.get() but works for:
        - Dict key access: get(mydict, 'key', default)
        - List index access: get(mylist, 1, default)
        - Attribute access: get(myobj, 'attr', default)

        Returns default if key/index/attribute doesn't exist or is out of bounds.

        Examples:
            {{get(files, 1, {})}}                      # List index
            {{get(files, 1, {}).content | default('')}} # Chained access
            {{get(block, 'outputs', {})}}              # Dict key
            {{get(obj, 'missing_attr', None)}}         # Attribute
        """
        try:
            # List/tuple index access
            if isinstance(key, int) and isinstance(obj, (list, tuple)):
                if -len(obj) <= key < len(obj):
                    return obj[key]
                return default

            # Dict key access
            if isinstance(obj, dict):
                return obj.get(key, default)

            # Attribute access
            if isinstance(key, str) and hasattr(obj, key):
                return getattr(obj, key, default)

            return default
        except (TypeError, KeyError, IndexError, AttributeError):
            return default

    @staticmethod
    def _render(template_str: str, variables: dict[str, Any]) -> str:
        """
        Render Jinja2 template with isolated environment (nested rendering).

        Enables multi-level template rendering by creating a new sandboxed
        environment for the template. This allows templates to be stored in
        files or variables and rendered with dynamic context.

        Args:
            template_str: Jinja2 template string to render
            variables: Dictionary of variables to pass to template

        Returns:
            Rendered template as string

        Examples:
            # Simple rendering
            {{render('Hello {{ name }}', {'name': 'World'})}}

            # Template from file
            {{render(
                get(blocks.read_template.outputs.files, 0, {}).content,
                {'project': inputs.project_name}
            )}}

            # Nested rendering (template within template)
            {{render(
                blocks.outer.content,
                {'inner': render(blocks.inner.content, {'value': 42})}
            )}}

            # With filters
            {{render(
                blocks.template.content | replace('OLD', 'NEW'),
                {'data': blocks.gen.output | trim}
            )}}
        """
        from jinja2 import StrictUndefined
        from jinja2.sandbox import SandboxedEnvironment

        env = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)
        template = env.from_string(template_str)
        return template.render(**variables)

    def _enhance_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Enhance context with proxy objects for shortcuts and special handling.

        Transformations:
        1. Wrap blocks with BlockProxy for attribute shortcuts
        2. Add SecretProxy for lazy secret loading
        3. Preserve special namespaces (each, inputs, metadata)

        Args:
            context: Raw workflow context

        Returns:
            Enhanced context with proxies
        """
        enhanced: dict[str, Any] = {}

        # Standard namespaces (pass through)
        for key in ["inputs", "metadata", "each"]:
            if key in context:
                enhanced[key] = context[key]

        # Wrap blocks with proxy for shortcuts
        if "blocks" in context:
            blocks = context["blocks"]
            if isinstance(blocks, dict):
                enhanced["blocks"] = {
                    block_id: self._wrap_block(block_data)
                    for block_id, block_data in blocks.items()
                }
            else:
                enhanced["blocks"] = blocks

        # Add secret proxy if provider available
        if self.secret_provider:
            enhanced["secrets"] = SecretProxy(self.secret_provider, self.audit_log)

        # Add any other root-level keys (except __internal__)
        for key, value in context.items():
            if key not in enhanced and key != "__internal__":
                enhanced[key] = value

        return enhanced

    def _wrap_block(self, block_data: Any) -> Any:
        """
        Recursively wrap block data with proxies.

        Args:
            block_data: Block execution data

        Returns:
            Wrapped block data (BlockProxy)
        """
        if not isinstance(block_data, dict):
            return block_data

        # Check if this is an Execution object (has blocks inside)
        if "blocks" in block_data:
            # Make a copy to avoid modifying original
            wrapped = dict(block_data)
            # Recursively wrap nested blocks
            wrapped["blocks"] = {k: self._wrap_block(v) for k, v in block_data["blocks"].items()}
            return BlockProxy(wrapped)
        else:
            return BlockProxy(block_data)

    def _apply_rules(self, expression: str) -> tuple[str, dict[str, Any]]:
        """
        Apply transformation rules in priority order.

        Args:
            expression: Expression to transform

        Returns:
            Tuple of (transformed_expression, metadata)
        """
        context = RuleContext(expression=expression, context=self.raw_context, metadata={})

        for rule in self.rules:
            if rule.applies_to(context):
                context = rule.transform(context)

        return context.expression, context.metadata

    async def resolve_async(self, value: Any) -> Any:
        """
        Main async resolution method with full capabilities.

        Process:
        1. Classify expression type
        2. Apply transformation rules
        3. Route to appropriate Jinja2 method
        4. Handle post-processing

        Args:
            value: Value to resolve (str, dict, list, or primitive)

        Returns:
            Resolved value with variables substituted
        """
        # Non-string values pass through
        if not isinstance(value, str):
            if isinstance(value, dict):
                resolved = {}
                for key, val in value.items():
                    resolved[key] = await self.resolve_async(val)
                return resolved
            elif isinstance(value, list):
                resolved_list = []
                for item in value:
                    resolved_list.append(await self.resolve_async(item))
                return resolved_list
            else:
                return value

        # Check for Jinja2 markers (both expressions and control structures)
        has_expressions = "{{" in value and "}}" in value
        has_control_structures = "{%" in value and "%}" in value

        # No Jinja2 markers - return as-is
        if not has_expressions and not has_control_structures:
            return value

        # If control structures are present, treat as full Jinja2 template
        # (control structures like {% for %}, {% if %} need full template rendering)
        if has_control_structures:
            # Apply transformation rules
            try:
                transformed, metadata = self._apply_rules(value)
            except SecurityError as e:
                raise ValueError(f"Security violation: {e}") from e

            # Evaluate as full Jinja2 template
            return await self._evaluate_template(transformed, metadata)

        # Only expressions (no control structures) - classify for optimal routing
        expr_type = self.classifier.classify(value)

        # Apply transformation rules
        try:
            transformed, metadata = self._apply_rules(value)
        except SecurityError as e:
            raise ValueError(f"Security violation: {e}") from e

        # Route based on type
        if expr_type == ExpressionType.LITERAL:
            return transformed

        elif expr_type in [
            ExpressionType.PURE_VARIABLE,
            ExpressionType.FILTER_EXPRESSION,
            ExpressionType.BOOLEAN_EXPRESSION,
            ExpressionType.MATH_EXPRESSION,
            ExpressionType.COMPLEX_EXPRESSION,
        ]:
            # Single expression - use compile_expression for type preservation
            return await self._evaluate_expression(transformed, metadata)

        elif expr_type == ExpressionType.TEMPLATE:
            # Template with multiple {{}} - always returns string
            return await self._evaluate_template(transformed, metadata)

        else:
            # Fallback to template evaluation
            return await self._evaluate_template(transformed, metadata)

    async def _evaluate_expression(self, expression: str, metadata: dict[str, Any]) -> Any:
        """
        Evaluate pure expression with type preservation.

        Args:
            expression: Expression to evaluate
            metadata: Rule metadata (e.g., contains_secrets)

        Returns:
            Evaluated value (type preserved)
        """
        # Extract inner expression
        stripped = expression.strip()
        if stripped.startswith("{{") and stripped.endswith("}}"):
            inner = stripped[2:-2].strip()
        else:
            inner = expression

        # Handle async operations (secrets)
        if metadata.get("contains_secrets"):
            await self._prepare_secrets(metadata.get("secret_keys", []))

        # Compile and evaluate
        try:
            compiled = self.env.compile_expression(inner)
            result = compiled(**self.jinja_context)

            return result

        except Exception as e:
            # Enhanced error message
            available_keys = list(self.jinja_context.keys())
            raise ValueError(
                f"Failed to evaluate expression: {inner}\n"
                f"Error: {str(e)}\n"
                f"Available variables: {available_keys}"
            ) from e

    async def _evaluate_template(self, template_str: str, metadata: dict[str, Any]) -> str:
        """
        Evaluate template string (always returns string).

        Args:
            template_str: Template string to evaluate
            metadata: Rule metadata (e.g., contains_secrets)

        Returns:
            Rendered template string
        """
        # Handle async operations
        if metadata.get("contains_secrets"):
            await self._prepare_secrets(metadata.get("secret_keys", []))

        # Render template
        try:
            template = self.env.from_string(template_str)
            result = template.render(**self.jinja_context)

            return result

        except Exception as e:
            raise ValueError(f"Failed to evaluate template: {template_str}\nError: {str(e)}") from e

    async def _prepare_secrets(self, secret_keys: list[str]) -> None:
        """
        Pre-fetch secrets and materialize into plain dict.

        Architecture (Option A):
        1. Detect secret references (sync - regex/AST scan)
        2. Pre-fetch secrets (async - supports Vault/AWS network calls)
        3. Materialize into plain dict (eliminates proxy during rendering)

        This ensures Jinja2 (synchronous) can access secrets without proxy magic
        while maintaining async compatibility for future providers.

        Args:
            secret_keys: List of secret keys to pre-fetch
        """
        if not self.secret_provider or not secret_keys:
            return

        secrets_value = self.jinja_context.get("secrets")

        # If already materialized (plain dict), fetch from original proxy
        if isinstance(secrets_value, dict) and not isinstance(secrets_value, SecretProxy):
            # Already materialized - need to fetch new secrets using the provider directly
            secret_proxy = SecretProxy(self.secret_provider, self.audit_log)
            existing_secrets = secrets_value
        elif isinstance(secrets_value, SecretProxy):
            # First time - use the proxy
            secret_proxy = secrets_value
            existing_secrets = {}
        else:
            # No secrets configured
            return

        # Pre-fetch all referenced secrets (async - supports network calls)
        tasks = [secret_proxy.get(key) for key in secret_keys]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Materialize: merge new secrets with existing ones
        secret_values = dict(existing_secrets)  # Copy existing secrets
        for key, result in zip(secret_keys, results):
            if isinstance(result, Exception):
                # Re-raise first error (audit logging already done in SecretProxy.get)
                raise result
            secret_values[key] = result

        # Replace proxy with materialized dict for Jinja2 rendering
        self.jinja_context["secrets"] = secret_values

    def resolve(self, value: Any) -> Any:
        """
        Synchronous resolution (no secret support).

        For backwards compatibility and non-async contexts.

        Args:
            value: Value to resolve

        Returns:
            Resolved value

        Raises:
            ValueError: If expression contains secrets (use resolve_async)
        """
        # Check if value contains secrets
        if isinstance(value, str) and "secrets." in value:
            raise ValueError(
                "Secret resolution requires async context. Use resolve_async() instead of resolve()"
            )

        # Create event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Remove secret provider for sync version
        original_provider = self.secret_provider
        self.secret_provider = None

        try:
            # Run async version
            return loop.run_until_complete(self.resolve_async(value))
        finally:
            self.secret_provider = original_provider
