"""
Variable resolution and conditional evaluation for workflow execution.

This module provides:
1. VariableResolver: Resolves {{var}} syntax in workflow inputs
2. ConditionEvaluator: Safely evaluates boolean expressions

Variable Resolution:
- {{inputs.param_name}} - References workflow inputs
- {{blocks.block_id.outputs.field}} - References block output fields
- {{blocks.block_id.inputs.field}} - References block input fields (debugging)
- {{blocks.block_id.metadata.field}} - References block metadata
- {{metadata.field}} - References workflow metadata
- {{secrets.SECRET_KEY}} - References secrets from SecretProvider (async)
- Recursive resolution in strings, dicts, lists

Conditional Evaluation:
- Safe AST-based evaluation (no arbitrary code execution)
- Supported operators: ==, !=, >, <, >=, <=, and, or, not, in
- Variables resolved before evaluation
"""

import ast
import operator
import re
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .secrets import SecretAuditLog, SecretProvider


class VariableNotFoundError(Exception):
    """Raised when a variable reference cannot be resolved."""

    pass


class InvalidConditionError(Exception):
    """Raised when a condition expression is invalid or unsafe."""

    pass


class VariableResolver:
    """
    Resolves {{var}} variable references from workflow context.

    Context Structure:
        {
            "inputs": {
                "working_dir": "/my/project",
                "python_version": "3.12"
            },
            "metadata": {
                "workflow_name": "my-workflow",
                "start_time": 1697123456.789
            },
            "blocks": {
                "run_tests": {
                    "inputs": {"command": "pytest"},
                    "outputs": {"exit_code": 0, "success": True},
                    "metadata": {"execution_time_ms": 1234}
                }
            }
        }

    Variable Syntax:
        - {{inputs.param_name}} - Workflow input
        - {{blocks.block_id.outputs.field}} - Block output
        - {{blocks.block_id.inputs.field}} - Block input (debugging)
        - {{blocks.block_id.metadata.field}} - Block metadata
        - {{blocks.block_id["iteration_key"].outputs.field}} - For_each iteration output (ADR-009)
        - {{metadata.field}} - Workflow metadata
        - {{secrets.SECRET_KEY}} - Secret from SecretProvider (async)
        - {{each.key}} - Current iteration key (within for_each) (ADR-009)
        - {{each.value}} - Current iteration value (within for_each) (ADR-009)
        - {{each.index}} - Current iteration index (within for_each) (ADR-009)
        - {{each.count}} - Total iteration count (within for_each) (ADR-009)

    Block Status References (ADR-007 - Industry-Aligned Three-Tier Model):

        Tier 1: Boolean Shortcuts (GitHub Actions style)
        - {{blocks.block_id.succeeded}} - True if completed successfully
        - {{blocks.block_id.failed}} - True if failed (any reason)
        - {{blocks.block_id.skipped}} - True if skipped

        Tier 2: Status String (Argo Workflows style)
        - {{blocks.block_id.status}} - Returns status as string
          Values: "pending"|"running"|"completed"|"failed"|"skipped"|"paused"

        Tier 3: Outcome String (Precision)
        - {{blocks.block_id.outcome}} - Returns outcome as string
          Values: "success"|"failure"|"n/a"

    Security:
        - {{__internal__.*}} - Access denied (internal system state)

    Example:
        context = {
            "inputs": {"branch": "main"},
            "blocks": {
                "create_worktree": {
                    "outputs": {"worktree_path": "/tmp/worktree"}
                }
            }
        }

        resolver = VariableResolver(context)
        result = resolver.resolve("Path: {{blocks.create_worktree.outputs.worktree_path}}")
        # Returns: "Path: /tmp/worktree"
    """

    # Pattern: Supports dot notation and bracket notation (ADR-009)
    # Examples:
    #   {{identifier.field}}                  - dot notation
    #   {{blocks.id["key"].outputs.field}}    - bracket notation with double quotes
    #   {{blocks.id['key'].field}}            - bracket notation with single quotes
    #   {{blocks.id[0].field}}                - bracket notation with numeric index
    #   {{blocks.id[{{var}}].field}}          - bracket notation with nested variable
    #   {{each.value.nested.field}}           - each namespace
    VAR_PATTERN = re.compile(
        r"\{\{([a-zA-Z_][a-zA-Z0-9_]*(?:"
        r"(?:\.[a-zA-Z_][a-zA-Z0-9_]*)|"  # Dot notation: .identifier
        r'(?:\["[^"]+"\])|'  # Bracket notation (double quotes): ["key"]
        r"(?:\['[^']+'\])|"  # Bracket notation (single quotes): ['key']
        r"(?:\[\d+\])|"  # Bracket notation (numeric index): [0], [123]
        r"(?:\[\{\{.+?\}\}\])"  # Bracket notation with nested variable: [{{var.path}}] (non-greedy)
        r")*)\}\}"
    )

    def __init__(
        self,
        context: dict[str, Any],
        secret_provider: "SecretProvider | None" = None,
        secret_audit_log: "SecretAuditLog | None" = None,
        workflow_name: str = "",
        block_id: str = "",
    ):
        """
        Initialize variable resolver with context.

        Args:
            context: Workflow context with inputs and block outputs
            secret_provider: Optional secret provider for resolving {{secrets.*}}
            secret_audit_log: Optional audit log for tracking secret access
            workflow_name: Workflow name for audit logging
            block_id: Block ID for audit logging
        """
        self.context = context
        self.secret_provider = secret_provider
        self.secret_audit_log = secret_audit_log
        self.workflow_name = workflow_name
        self.block_id = block_id

    @staticmethod
    def parse_variable_path(var_path: str) -> list[str]:
        """
        Parse variable path into segments, handling bracket notation (ADR-009).

        PUBLIC METHOD: Can be used by other modules (e.g., schema validator)
        for consistent variable path parsing across the codebase.

        Converts mixed dot and bracket notation into a list of segments:
        - "blocks.analyze_files.outputs" → ["blocks", "analyze_files", "outputs"]
        - 'blocks.analyze["file1"].outputs' → ["blocks", "analyze", "file1", "outputs"]
        - "each.value.nested" → ["each", "value", "nested"]
        - " branch_name " → ["branch_name"] (strips whitespace for Jinja2 compatibility)

        Args:
            var_path: Variable path string (e.g., 'blocks.id["key"].field')

        Returns:
            List of segments for dictionary navigation

        Raises:
            ValueError: If bracket notation is malformed (unclosed brackets, unquoted keys)

        Example:
            >>> VariableResolver.parse_variable_path('blocks.analyze["file1"].outputs.response')
            ["blocks", "analyze", "file1", "outputs", "response"]
            >>> VariableResolver.parse_variable_path(' branch_name ')
            ["branch_name"]
        """
        # Strip leading/trailing whitespace (Jinja2 templates like {{ var }} include spaces)
        var_path = var_path.strip()

        segments: list[str] = []
        current = ""  # Accumulator for current segment being parsed
        i = 0

        while i < len(var_path):
            char = var_path[i]

            if char == ".":
                # Dot separator - finalize current segment and move to next
                if current:
                    segments.append(current)
                    current = ""
                i += 1

            elif char == "[":
                # Bracket notation - finalize current identifier (if any)
                if current:
                    segments.append(current)
                    current = ""

                # ADR-009: For for_each blocks, insert 'blocks' namespace before bracket key
                # Transform blocks.id['key'] → blocks.id.blocks['key'] at parse time
                if len(segments) == 2 and segments[0] == "blocks":
                    segments.append("blocks")

                # Extract bracket content using regex to handle nested {{...}}
                # Match: [quoted-string], [digit], or [{{...}}]
                bracket_pattern = re.compile(
                    r"\["  # Opening bracket
                    r"(?:"
                    r'"([^"]+)"|'  # Double-quoted string
                    r"'([^']+)'|"  # Single-quoted string
                    r"(\d+)|"  # Numeric index
                    r"(\{\{.+?\}\})"  # Nested variable (non-greedy)
                    r")"
                    r"\]"  # Closing bracket
                )
                match = bracket_pattern.match(var_path[i:])
                if not match:
                    raise ValueError(
                        f"Invalid bracket notation at position {i}: {var_path[i : i + 20]}..."
                    )

                # Extract the matched content (one of the 4 groups)
                double_quoted, single_quoted, numeric, nested_var = match.groups()
                if double_quoted:
                    segments.append(double_quoted)
                elif single_quoted:
                    segments.append(single_quoted)
                elif numeric:
                    segments.append(numeric)
                elif nested_var:
                    # Store nested variable for later resolution
                    segments.append(nested_var)

                i += match.end()  # Move past the entire bracket expression
                continue  # Skip to next iteration, don't fall through to else

            else:
                # Regular identifier character - accumulate
                current += char
                i += 1

        # Add final segment (if any)
        if current:
            segments.append(current)

        return segments

    def resolve(self, value: Any, for_eval: bool = False) -> Any:
        """
        Recursively resolve variables in value (synchronous version).

        NOTE: This method does NOT resolve {{secrets.*}} references.
        Use resolve_async() for secret support.

        Args:
            value: Value to resolve (str, dict, list, or primitive)
            for_eval: If True, format string values for Python eval

        Returns:
            Resolved value with variables substituted

        Raises:
            VariableNotFoundError: If a variable reference cannot be resolved
            VariableNotFoundError: If {{secrets.*}} reference is found (use resolve_async)
        """
        if isinstance(value, str):
            return self._resolve_string(value, for_eval=for_eval)
        elif isinstance(value, dict):
            return {key: self.resolve(val, for_eval=for_eval) for key, val in value.items()}
        elif isinstance(value, list):
            return [self.resolve(item, for_eval=for_eval) for item in value]
        else:
            # Primitive types (int, float, bool, None) pass through
            return value

    async def resolve_async(self, value: Any, for_eval: bool = False) -> Any:
        """
        Recursively resolve variables in value (async version with secret support).

        This method supports {{secrets.*}} references via SecretProvider.
        Use this instead of resolve() when secret resolution is needed.

        Args:
            value: Value to resolve (str, dict, list, or primitive)
            for_eval: If True, format string values for Python eval

        Returns:
            Resolved value with variables and secrets substituted

        Raises:
            VariableNotFoundError: If a variable reference cannot be resolved
            SecretNotFoundError: If a secret reference cannot be resolved (from provider)
        """
        if isinstance(value, str):
            # Check if this is a pure variable reference (e.g., "{{inputs.services}}")
            # If so, return the actual object; otherwise return string interpolation
            match = self.VAR_PATTERN.fullmatch(value)
            if match:
                # Pure variable reference - return actual object (preserves type)
                # This is critical for for_each expressions that need dict/list objects
                var_path = match.group(1)

                # Security: Block access to internal namespace
                if var_path.startswith("__internal__") or ".__internal__" in var_path:
                    raise VariableNotFoundError(
                        f"Access to internal namespace is not allowed: {{{{{var_path}}}}}"
                    )

                # Handle {{secrets.*}} references
                if var_path.startswith("secrets."):
                    secret_key = var_path[8:]  # Remove "secrets." prefix

                    if not self.secret_provider:
                        raise VariableNotFoundError(
                            f"Secret provider not configured. Cannot resolve: {{{{{var_path}}}}}"
                        )

                    from .secrets import SecretNotFoundError

                    try:
                        secret_value = await self.secret_provider.get_secret(secret_key)

                        if self.secret_audit_log:
                            await self.secret_audit_log.log_access(
                                workflow_name=self.workflow_name,
                                block_id=self.block_id,
                                secret_key=secret_key,
                                success=True,
                            )

                        return secret_value  # Return actual object, not string

                    except SecretNotFoundError as e:
                        if self.secret_audit_log:
                            await self.secret_audit_log.log_access(
                                workflow_name=self.workflow_name,
                                block_id=self.block_id,
                                secret_key=secret_key,
                                success=False,
                                error_message=str(e),
                            )
                        raise
                else:
                    # Regular variable - resolve and return actual object
                    return self._get_variable_value(var_path)
            else:
                # String with surrounding text or multiple variables
                # Use string interpolation
                return await self._resolve_string_async(value, for_eval=for_eval)
        elif isinstance(value, dict):
            resolved: dict[Any, Any] = {}
            for key, val in value.items():
                resolved[key] = await self.resolve_async(val, for_eval=for_eval)
            return resolved
        elif isinstance(value, list):
            resolved_list: list[Any] = []
            for item in value:
                resolved_list.append(await self.resolve_async(item, for_eval=for_eval))
            return resolved_list
        else:
            # Primitive types (int, float, bool, None) pass through
            return value

    def _get_variable_value(self, var_path: str) -> Any:
        """
        Get the raw value of a variable without formatting (for pure variable references).

        This is used when resolving pure variable references like "{{inputs.services}}"
        where we need to preserve the actual object type (dict, list, etc.) instead
        of converting to a string.

        Args:
            var_path: Variable path (e.g., "inputs.services", "blocks.id.outputs.data")

        Returns:
            Raw variable value (preserving type: dict, list, str, int, etc.)

        Raises:
            VariableNotFoundError: If variable not found or inaccessible
        """
        # Security: Block access to internal namespace
        if var_path.startswith("__internal__") or ".__internal__" in var_path:
            raise VariableNotFoundError(
                f"Access to internal namespace is not allowed: {{{{{var_path}}}}}"
            )

        # Parse path with bracket notation support (ADR-009)
        segments = self.parse_variable_path(var_path)

        # ADR-009: Handle 'each' namespace (for_each iteration context)
        if segments[0] == "each":
            if "each" not in self.context:
                raise VariableNotFoundError(
                    f"Variable '{{{{{var_path}}}}}' not found. "
                    f"'each' namespace only available within for_each iterations."
                )
            # Navigate from 'each' namespace
            value = self.context["each"]
            for i, segment in enumerate(segments[1:], start=1):
                if not isinstance(value, dict):
                    partial_path = ".".join(segments[:i])
                    raise VariableNotFoundError(
                        f"Cannot access '{segment}' on non-dict value at '{{{{{partial_path}}}}}'"
                    )
                if segment not in value:
                    available = list(value.keys()) if isinstance(value, dict) else []
                    raise VariableNotFoundError(
                        f"Variable '{{{{{var_path}}}}}' not found. "
                        f"At segment '{segment}', available keys: {available}"
                    )
                value = value[segment]
            return value  # Return raw value

        # ADR-007: Three-tier status reference model
        # Transform shortcuts to metadata namespace
        if (
            len(segments) == 3
            and segments[0] == "blocks"
            and segments[2] in ("succeeded", "failed", "skipped")
        ):
            segments = segments[:2] + ["metadata"] + segments[2:]
        elif len(segments) == 3 and segments[0] == "blocks" and segments[2] == "status":
            segments = segments[:2] + ["metadata"] + segments[2:]
        elif len(segments) == 3 and segments[0] == "blocks" and segments[2] == "outcome":
            segments = segments[:2] + ["metadata"] + segments[2:]

        # Dictionary navigation with bracket notation support
        value = self.context
        for i, segment in enumerate(segments):
            # Check if segment contains nested variable interpolation (e.g., {{inputs.key}})
            if "{{" in segment and "}}" in segment:
                # Resolve the nested variable first
                resolved_segment = self._resolve_string(segment, for_eval=False)
                segment = resolved_segment

            if not isinstance(value, dict):
                partial_path = ".".join(segments[:i])
                raise VariableNotFoundError(
                    f"Cannot access '{segment}' on non-dict value at '{{{{{partial_path}}}}}'"
                )

            if segment not in value:
                available = list(value.keys()) if isinstance(value, dict) else []
                raise VariableNotFoundError(
                    f"Variable '{{{{{var_path}}}}}' not found. "
                    f"At segment '{segment}', available keys: {available}"
                )

            value = value[segment]

        return value  # Return raw value without formatting

    def _resolve_variable(self, var_path: str, for_eval: bool = False) -> str:
        """
        Resolve a single variable path to its value (DRY helper for both sync/async).

        Handles:
        - Security checks (__internal__ blocking)
        - Bracket notation parsing (ADR-009)
        - 'each' namespace (ADR-009)
        - Status shortcuts (ADR-007)
        - Dictionary navigation

        Args:
            var_path: Variable path (e.g., "blocks.id['key'].outputs.field")
            for_eval: If True, format for Python eval

        Returns:
            Formatted string value

        Raises:
            VariableNotFoundError: If variable not found or inaccessible
        """
        # Security: Block access to internal namespace
        if var_path.startswith("__internal__") or ".__internal__" in var_path:
            raise VariableNotFoundError(
                f"Access to internal namespace is not allowed: {{{{{var_path}}}}}"
            )

        # Parse path with bracket notation support (ADR-009)
        segments = self.parse_variable_path(var_path)

        # ADR-009: Handle 'each' namespace (for_each iteration context)
        if segments[0] == "each":
            if "each" not in self.context:
                raise VariableNotFoundError(
                    f"Variable '{{{{{var_path}}}}}' not found. "
                    f"'each' namespace only available within for_each iterations."
                )
            # Navigate from 'each' namespace
            value = self.context["each"]
            for i, segment in enumerate(segments[1:], start=1):
                if not isinstance(value, dict):
                    partial_path = ".".join(segments[:i])
                    raise VariableNotFoundError(
                        f"Cannot access '{segment}' on non-dict value at '{{{{{partial_path}}}}}'"
                    )
                if segment not in value:
                    available = list(value.keys()) if isinstance(value, dict) else []
                    raise VariableNotFoundError(
                        f"Variable '{{{{{var_path}}}}}' not found. "
                        f"At segment '{segment}', available keys: {available}"
                    )
                value = value[segment]

            return self._format_for_eval(value) if for_eval else self._format_for_string(value)

        # ADR-007: Three-tier status reference model
        # Transform shortcuts to metadata namespace
        if (
            len(segments) == 3
            and segments[0] == "blocks"
            and segments[2] in ("succeeded", "failed", "skipped")
        ):
            segments = segments[:2] + ["metadata"] + segments[2:]
        elif len(segments) == 3 and segments[0] == "blocks" and segments[2] == "status":
            segments = segments[:2] + ["metadata"] + segments[2:]
        elif len(segments) == 3 and segments[0] == "blocks" and segments[2] == "outcome":
            segments = segments[:2] + ["metadata"] + segments[2:]

        # Dictionary navigation with bracket notation support
        value = self.context
        for i, segment in enumerate(segments):
            # Check if segment contains nested variable interpolation (e.g., {{inputs.key}})
            if "{{" in segment and "}}" in segment:
                # Resolve the nested variable first
                resolved_segment = self._resolve_string(segment, for_eval=False)
                segment = resolved_segment

            if not isinstance(value, dict):
                partial_path = ".".join(segments[:i])
                raise VariableNotFoundError(
                    f"Cannot access '{segment}' on non-dict value at '{{{{{partial_path}}}}}'"
                )

            if segment not in value:
                available = list(value.keys()) if isinstance(value, dict) else []
                raise VariableNotFoundError(
                    f"Variable '{{{{{var_path}}}}}' not found. "
                    f"At segment '{segment}', available keys: {available}"
                )

            value = value[segment]

        # Format value for return
        return self._format_for_eval(value) if for_eval else self._format_for_string(value)

    def _resolve_string(self, text: str, for_eval: bool = False) -> str:
        """
        Replace {{var}} patterns with context values (synchronous version).

        NOTE: This method does NOT resolve {{secrets.*}} references.
        Use _resolve_string_async() for secret support.

        Args:
            text: String containing variable references
            for_eval: If True, format values for Python eval (quote strings, etc.)

        Returns:
            String with variables substituted

        Raises:
            VariableNotFoundError: If variable not found in context
            VariableNotFoundError: If {{secrets.*}} reference is found (use resolve_async)
        """

        def replace_var(match: re.Match[str]) -> str:
            """Replace a single variable match using shared resolution logic."""
            var_path = match.group(1)

            # Block {{secrets.*}} in synchronous resolution
            if var_path.startswith("secrets."):
                raise VariableNotFoundError(
                    f"Secret resolution requires async context. "
                    f"Use resolve_async() instead of resolve() for: {{{{{var_path}}}}}"
                )

            # Use shared resolution logic (DRY)
            return self._resolve_variable(var_path, for_eval=for_eval)

        return self.VAR_PATTERN.sub(replace_var, text)

    async def _resolve_string_async(self, text: str, for_eval: bool = False) -> str:
        """
        Replace {{var}} patterns with context values (async version with secrets).

        This method supports {{secrets.*}} references via SecretProvider.

        Args:
            text: String containing variable references
            for_eval: If True, format values for Python eval (quote strings, etc.)

        Returns:
            String with variables and secrets substituted

        Raises:
            VariableNotFoundError: If variable not found in context
            SecretNotFoundError: If secret not found in provider (from secrets module)
        """
        # Find all variable references
        matches = list(self.VAR_PATTERN.finditer(text))

        if not matches:
            return text

        # Process matches in reverse order to preserve positions
        result = text
        for match in reversed(matches):
            var_path = match.group(1)

            # Security: Block access to internal namespace
            if var_path.startswith("__internal__") or ".__internal__" in var_path:
                raise VariableNotFoundError(
                    f"Access to internal namespace is not allowed: {{{{{var_path}}}}}"
                )

            # Handle {{secrets.*}} references
            if var_path.startswith("secrets."):
                # Extract secret key: "secrets.API_KEY" → "API_KEY"
                secret_key = var_path[8:]  # Remove "secrets." prefix

                if not self.secret_provider:
                    raise VariableNotFoundError(
                        f"Secret provider not configured. Cannot resolve: {{{{{var_path}}}}}"
                    )

                # Import SecretNotFoundError here to avoid circular import
                from .secrets import SecretNotFoundError

                # Fetch secret from provider (async)
                try:
                    secret_value = await self.secret_provider.get_secret(secret_key)

                    # Log access to audit log
                    if self.secret_audit_log:
                        await self.secret_audit_log.log_access(
                            workflow_name=self.workflow_name,
                            block_id=self.block_id,
                            secret_key=secret_key,
                            success=True,
                        )

                    # Format and replace
                    formatted_value = (
                        self._format_for_eval(secret_value)
                        if for_eval
                        else self._format_for_string(secret_value)
                    )
                    result = result[: match.start()] + formatted_value + result[match.end() :]

                except SecretNotFoundError as e:
                    # Log failed access
                    if self.secret_audit_log:
                        await self.secret_audit_log.log_access(
                            workflow_name=self.workflow_name,
                            block_id=self.block_id,
                            secret_key=secret_key,
                            success=False,
                            error_message=str(e),
                        )
                    # Re-raise with context
                    raise

            else:
                # Handle regular variable using shared resolution logic (DRY)
                formatted_value = self._resolve_variable(var_path, for_eval=for_eval)
                result = result[: match.start()] + formatted_value + result[match.end() :]

        return result

    def _format_for_eval(self, value: Any) -> str:
        """Format value for Python eval (proper literals).

        Enums are converted to their quoted string values for eval (ADR-007):
        - ExecutionStatus.COMPLETED → "'completed'"
        - OperationOutcome.SUCCESS → "'success'"
        """

        # Check Enum BEFORE str because ExecutionStatus/OperationOutcome inherit from str
        if isinstance(value, Enum):
            # Convert enum to quoted string for eval (ADR-007: status/outcome in conditions)
            return repr(value.value)
        elif isinstance(value, str):
            return repr(value)
        elif isinstance(value, bool):
            return "True" if value else "False"
        elif value is None:
            return "None"
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            return repr(value)

    def _format_for_string(self, value: Any) -> str:
        """Format value for regular string substitution.

        Booleans are lowercased to match bash/YAML conventions:
        - True → "true" (compatible with bash `[ "true" = "true" ]`)
        - False → "false"

        Enums are converted to their string values (ADR-007):
        - ExecutionStatus.COMPLETED → "completed"
        - OperationOutcome.SUCCESS → "success"
        """

        # Check bool BEFORE Enum/str since bool is a subclass of int
        if isinstance(value, bool):
            return str(value).lower()  # "true" or "false"
        # Check Enum BEFORE str because ExecutionStatus/OperationOutcome inherit from str
        elif isinstance(value, Enum):
            # Convert enum to its string value (ADR-007: status/outcome strings)
            return str(value.value)
        elif isinstance(value, str):
            return value
        elif isinstance(value, (int, float)):
            return str(value)
        elif value is None:
            return ""
        else:
            return repr(value)


class ConditionEvaluator:
    """
    Safe AST-based boolean expression evaluator for conditional execution.

    Supported Operators:
        - Comparison: ==, !=, >, <, >=, <=
        - Boolean: and, or, not
        - Membership: in
        - Literals: strings, numbers, booleans, None

    Security:
        - Uses ast.parse() for safe evaluation (no code execution)
        - Whitelist of allowed operators
        - No function calls, attribute access, or imports

    Example:
        evaluator = ConditionEvaluator()
        context = {"run_tests.exit_code": 0, "coverage": 85}

        # Resolve variables first
        condition = "{{run_tests.exit_code}} == 0 and {{coverage}} >= 80"
        result = evaluator.evaluate(condition, context)
        # Returns: True
    """

    # Whitelist of safe operators
    SAFE_OPERATORS: dict[type, Any] = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.And: operator.and_,
        ast.Or: operator.or_,
        ast.Not: operator.not_,
        ast.In: lambda x, y: x in y,
        ast.NotIn: lambda x, y: x not in y,
    }

    def evaluate(self, condition: str, context: dict[str, Any]) -> bool:
        """
        Evaluate condition string against context.

        Args:
            condition: Boolean expression with variable references
            context: Workflow context for variable resolution

        Returns:
            Boolean result of evaluation

        Raises:
            InvalidConditionError: If condition is invalid or unsafe
            VariableNotFoundError: If variable reference not found

        Example:
            result = evaluator.evaluate(
                "{{run_tests.exit_code}} == 0 and {{coverage}} >= 80",
                {"run_tests.exit_code": 0, "coverage": 85}
            )
        """
        import logging

        logger = logging.getLogger(__name__)

        # Step 1: Resolve variables with for_eval=True to get proper Python literals
        try:
            resolver = VariableResolver(context)
            resolved_condition = resolver.resolve(condition, for_eval=True)
            logger.debug(f"Condition variable resolution: '{condition}' → '{resolved_condition}'")
        except VariableNotFoundError as e:
            logger.error(f"Variable resolution failed for condition: {condition}")
            logger.debug(f"Available blocks: {list(context.get('blocks', {}).keys())}")
            raise InvalidConditionError(f"Variable resolution failed: {e}") from e

        # Step 2: Parse and evaluate safely
        try:
            result = self._safe_eval(resolved_condition)
            logger.debug(f"Condition evaluated: '{resolved_condition}' → {result}")
            if not isinstance(result, bool):
                raise InvalidConditionError(
                    f"Condition must evaluate to boolean, got {type(result).__name__}"
                )
            return result
        except InvalidConditionError:
            raise
        except Exception as e:
            raise InvalidConditionError(f"Condition evaluation failed: {e}") from e

    def _safe_eval(self, expr: str) -> bool:
        """
        Safely evaluate boolean expression using AST with explicit operator whitelist.

        This approach is secure because:
        1. AST parsing - validates syntax before evaluation
        2. Explicit operator whitelist - only allowed operations can execute
        3. No function calls, imports, or attribute access
        4. Expression is already variable-resolved

        Normalizes YAML boolean literals (true/false) to Python (True/False)
        and string boolean representations ("True"/"False") to actual booleans.

        Security Model:
            - Whitelisted operators: ==, !=, <, >, <=, >=, and, or, not, in, not in
            - Allowed literals: strings, numbers, booleans, None, lists, tuples
            - Rejected: function calls, attribute access, imports, comprehensions

        Args:
            expr: Expression string (variables already resolved)

        Returns:
            Boolean evaluation result

        Raises:
            InvalidConditionError: If expression is unsafe or invalid

        Example:
            _safe_eval("True and False")  # Returns: False
            _safe_eval("5 > 3 and 10 <= 20")  # Returns: True
            _safe_eval("'x' in ['x', 'y']")  # Returns: True
        """
        try:
            # Normalize whitespace (ast.parse mode='eval' requires single-line expressions)
            # Replace newlines with spaces, then strip
            import re

            expr = expr.replace("\n", " ").replace("\r", " ").strip()

            # Normalize YAML boolean literals to Python (Bug #3 fix)
            # Use word boundaries to avoid false positives (e.g., "untrue" should not match)
            expr = re.sub(r"\btrue\b", "True", expr)
            expr = re.sub(r"\bfalse\b", "False", expr)

            # Normalize string boolean representations to actual booleans (Bug #4 fix)
            # Handle both single and double quotes
            expr = expr.replace("'True'", "True").replace("'False'", "False")
            expr = expr.replace('"True"', "True").replace('"False"', "False")

            # Parse expression to AST
            tree = ast.parse(expr, mode="eval")

            # Evaluate using whitelisted operators
            result = self._eval_node(tree.body)

            # Validate result is boolean
            if not isinstance(result, bool):
                raise InvalidConditionError(
                    f"Expression must evaluate to boolean, got {type(result).__name__}: {result}"
                )

            return result

        except SyntaxError as e:
            raise InvalidConditionError(f"Invalid syntax in expression: {e}") from e
        except InvalidConditionError:
            # Re-raise our own exceptions
            raise
        except Exception as e:
            raise InvalidConditionError(f"Evaluation error: {e}") from e

    def _eval_node(self, node: ast.AST) -> Any:
        """
        Recursively evaluate AST node with operator whitelist.

        Args:
            node: AST node to evaluate

        Returns:
            Evaluated value

        Raises:
            InvalidConditionError: If node type is not whitelisted
        """
        # Literals
        if isinstance(node, ast.Constant):
            return node.value

        # Boolean operations (and, or)
        elif isinstance(node, ast.BoolOp):
            bool_op_type = type(node.op)
            if bool_op_type not in self.SAFE_OPERATORS:
                raise InvalidConditionError(
                    f"Unsupported boolean operator: {bool_op_type.__name__}"
                )

            op_func = self.SAFE_OPERATORS[bool_op_type]
            values = [self._eval_node(val) for val in node.values]

            # Apply operator (and/or)
            result = values[0]
            for val in values[1:]:
                result = op_func(result, val)
            return result

        # Comparison operations (==, !=, <, >, <=, >=)
        elif isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            result = True

            for op, comparator in zip(node.ops, node.comparators):
                cmp_op_type = type(op)
                if cmp_op_type not in self.SAFE_OPERATORS:
                    raise InvalidConditionError(
                        f"Unsupported comparison operator: {cmp_op_type.__name__}"
                    )

                right = self._eval_node(comparator)
                op_func = self.SAFE_OPERATORS[cmp_op_type]

                result = result and op_func(left, right)
                left = right

            return result

        # Unary operations (not)
        elif isinstance(node, ast.UnaryOp):
            unary_op_type = type(node.op)
            if unary_op_type not in self.SAFE_OPERATORS:
                raise InvalidConditionError(f"Unsupported unary operator: {unary_op_type.__name__}")

            operand = self._eval_node(node.operand)
            op_func = self.SAFE_OPERATORS[unary_op_type]
            return op_func(operand)

        # Lists and tuples (for 'in' operator)
        elif isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval_node(elt) for elt in node.elts]

        # Unsupported node type
        else:
            raise InvalidConditionError(
                f"Unsupported expression type: {type(node).__name__}. "
                f"Only literals, comparisons, and boolean operators are allowed."
            )
