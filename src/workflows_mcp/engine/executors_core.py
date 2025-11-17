"""Core workflow executors - Shell and Workflow."""

import asyncio
import json
import os
import shlex
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, field_validator

from .block import BlockInput, BlockOutput
from .context_vars import block_custom_outputs
from .execution import Execution
from .executor_base import (
    BlockExecutor,
    ExecutorCapabilities,
    ExecutorSecurityLevel,
)
from .interpolation import (
    interpolatable_numeric_validator,
    resolve_interpolatable_numeric,
)

# ============================================================================
# Shell Executor
# ============================================================================


class ShellInput(BlockInput):
    """Input model for Shell executor.

    Architecture (ADR-006):
    - Execute returns ShellOutput directly
    - Operation outcome determined by exit_code
    - Exceptions indicate execution failure
    """

    model_config = {"extra": "forbid"}

    command: str = Field(description="Shell command to execute")
    working_dir: str = Field(default="", description="Working directory (empty = current dir)")
    timeout: int | str = Field(
        default=120, description="Timeout in seconds (or interpolation string)"
    )
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    capture_output: bool = Field(default=True, description="Capture stdout/stderr")
    shell: bool = Field(default=True, description="Execute via shell")

    # Validator for numeric field with interpolation support
    _validate_timeout = field_validator("timeout", mode="before")(
        interpolatable_numeric_validator(int, ge=1, le=3600)
    )


class ShellOutput(BlockOutput):
    """Output model for Shell executor.

    Custom outputs declared in YAML are merged directly as fields via extra="allow".
    No separate custom_outputs dict needed.

    All fields have defaults to support graceful degradation when commands fail.
    A default-constructed instance represents a failed/crashed command execution.
    """

    exit_code: int = Field(
        default=0,
        description="Process exit code (0 if command crashed before execution)",
    )
    stdout: str = Field(
        default="",
        description="Standard output (empty if command crashed)",
    )
    stderr: str = Field(
        default="",
        description="Standard error (empty if command crashed)",
    )

    model_config = {"extra": "allow"}  # Allow dynamic custom output fields


class OutputSecurityError(Exception):
    """Raised when output path violates security constraints."""

    pass


class OutputNotFoundError(Exception):
    """Raised when output file not found."""

    pass


def validate_output_path(
    output_name: str, path: str, working_dir: Path, unsafe: bool = False
) -> Path:
    """
    Validate output file path with security checks.

    Security rules:
    - Safe mode (default): Relative paths only, within working_dir
    - Unsafe mode (opt-in): Allows absolute paths
    - Always: No symlinks, size limit (10MB), no path traversal

    Args:
        output_name: Name of the output (for error messages)
        path: File path to validate (can contain env vars)
        working_dir: Working directory for relative paths
        unsafe: Allow absolute paths (default: False)

    Returns:
        Validated absolute Path

    Raises:
        OutputSecurityError: If path violates security constraints
        OutputNotFoundError: If file doesn't exist
    """
    # Expand environment variables
    expanded_path = os.path.expandvars(path)

    # Convert to Path object
    file_path = Path(expanded_path)

    # Security check: reject absolute paths in safe mode
    if file_path.is_absolute() and not unsafe:
        raise OutputSecurityError(
            f"Output '{output_name}': Absolute paths not allowed in safe mode. "
            f"Path: {path}. Set 'unsafe: true' to allow absolute paths."
        )

    # Build absolute path (without resolving symlinks yet)
    if file_path.is_absolute():
        absolute_path = file_path
    else:
        absolute_path = working_dir / file_path

    # Security check: no symlinks (check BEFORE resolving)
    # Must check before resolve() because resolve() follows symlinks
    if absolute_path.is_symlink():
        raise OutputSecurityError(
            f"Output '{output_name}': Symlinks not allowed for security. Path: {absolute_path}"
        )

    # Security check: path traversal - check BEFORE file existence
    # This prevents information leakage about files outside working directory
    # Resolve the path to handle .. and any symlinks in parent directories
    resolved_path = absolute_path.resolve()

    if not unsafe:
        try:
            resolved_path.relative_to(working_dir.resolve())
        except ValueError:
            raise OutputSecurityError(
                f"Output '{output_name}': Path escapes working directory. "
                f"Path: {path}, Resolved: {resolved_path}, Working dir: {working_dir}"
            )

    # Check file exists (after security checks)
    if not resolved_path.exists():
        raise OutputNotFoundError(f"Output '{output_name}': File not found at {resolved_path}")

    # Security check: must be a file
    if not resolved_path.is_file():
        raise OutputSecurityError(
            f"Output '{output_name}': Path is not a file. Path: {resolved_path}"
        )

    # Security check: size limit (10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    file_size = resolved_path.stat().st_size
    if file_size > max_size:
        raise OutputSecurityError(
            f"Output '{output_name}': File too large ({file_size} bytes, max {max_size} bytes). "
            f"Path: {resolved_path}"
        )

    return resolved_path


def parse_output_value(content: str, output_type: str) -> Any:
    """
    Parse file content according to declared type.

    Args:
        content: Raw file content
        output_type: One of ValueType values: str, num, bool, json

    Returns:
        Parsed value with correct Python type

    Raises:
        ValueError: If content doesn't match declared type
    """
    content = content.strip()

    if output_type == "str":
        return str(content)  # Explicit cast for consistency
    elif output_type == "num":
        # Parse as number (int or float)
        try:
            # Try float first (accepts both int and float strings)
            value = float(content)
            # Return int if it's a whole number, otherwise float
            return int(value) if value.is_integer() else value
        except ValueError:
            raise ValueError(f"Cannot parse as num: {content}")
    elif output_type == "bool":
        # Accept: true/false, 1/0, yes/no (case-insensitive)
        lower = content.lower()
        if lower in ["true", "1", "yes"]:
            return True
        elif lower in ["false", "0", "no"]:
            return False
        else:
            raise ValueError(
                f"Cannot parse as bool: {content}. "
                f"Accepted values: true/false, 1/0, yes/no (case-insensitive)"
            )
    elif output_type == "json":
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Cannot parse as JSON: {e}")
    else:
        raise ValueError(f"Unknown output type: {output_type}")


def coerce_value_type(value: Any, output_type: str) -> Any:
    """
    Coerce a value to the declared type (flexible version of parse_output_value).

    Handles both string values that need parsing and already-typed values.
    Used for workflow output type coercion where values may already be correctly typed
    from nested workflows or block outputs.

    Args:
        value: Value to coerce (can be str, int, float, bool, dict, list, etc.)
        output_type: One of ValueType values: str, num, bool, json, list, dict

    Returns:
        Value coerced to correct Python type

    Raises:
        ValueError: If value cannot be coerced to declared type

    Examples:
        # String values (parsed)
        coerce_value_type("42", "num") -> 42
        coerce_value_type("42.5", "num") -> 42.5
        coerce_value_type("true", "bool") -> True
        coerce_value_type('{"a":1}', "json") -> {"a": 1}

        # Already-typed values (validated and passed through)
        coerce_value_type(42, "num") -> 42
        coerce_value_type(42.5, "num") -> 42.5
        coerce_value_type(True, "bool") -> True
        coerce_value_type({"a": 1}, "json") -> {"a": 1}

        # Type conversion when needed
        coerce_value_type(42, "str") -> "42"
        coerce_value_type(42.5, "str") -> "42.5"
    """
    # Handle string values using existing parser
    if isinstance(value, str):
        # Use existing parse_output_value for strings
        return parse_output_value(value, output_type)

    # Handle already-typed values with validation and conversion
    if output_type == "str":
        # Convert any value to string
        # Special handling for booleans to produce JSON-compatible strings
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    elif output_type == "num":
        # Accept both int and float (exclude bool since it's subclass of int)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        # Try to convert
        try:
            result = float(value)
            # Return int if whole number, otherwise float
            return int(result) if result.is_integer() else result
        except (ValueError, TypeError):
            raise ValueError(f"Cannot coerce {type(value).__name__} to num: {value}")

    elif output_type == "bool":
        # If already bool, return as-is
        if isinstance(value, bool):
            return value
        # Try to convert using standard Python truthiness
        # But be strict - only accept actual booleans or 1/0 integers
        if isinstance(value, int) and value in [0, 1]:
            return bool(value)
        raise ValueError(
            f"Cannot coerce {type(value).__name__} to bool: {value}. "
            f"Only bool values or integers 0/1 are accepted."
        )

    elif output_type == "json":
        # Accept dict, list, or JSON-compatible primitives
        if isinstance(value, (dict, list, str, int, float, bool, type(None))):
            return value
        raise ValueError(f"Cannot coerce {type(value).__name__} to json: {value}")

    elif output_type == "list":
        # If already list, return as-is
        if isinstance(value, list):
            return value
        raise ValueError(f"Cannot coerce {type(value).__name__} to list: {value}")

    elif output_type == "dict":
        # If already dict, return as-is
        if isinstance(value, dict):
            return value
        raise ValueError(f"Cannot coerce {type(value).__name__} to dict: {value}")

    else:
        raise ValueError(f"Unknown output type: {output_type}")


class ShellExecutor(BlockExecutor):
    """
    Shell command executor.

    Architecture (ADR-006):
    - Execute returns ShellOutput directly
    - Raises exceptions for execution failures
    - Operation outcome determined by exit_code in output
    - Orchestrator creates Metadata based on exit_code and exceptions

    Features:
    - Async subprocess execution
    - Timeout support
    - Environment variable injection
    - Working directory control
    - Output capture (stdout/stderr)
    - Shell/direct execution modes
    - Exit code reporting
    - Custom file-based outputs
    - Scratch directory management
    """

    type_name: ClassVar[str] = "Shell"
    input_type: ClassVar[type[BlockInput]] = ShellInput
    output_type: ClassVar[type[BlockOutput]] = ShellOutput

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.PRIVILEGED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(
        can_execute_commands=True,
        can_read_files=True,
        can_write_files=True,
        can_network=True,
    )

    async def execute(  # type: ignore[override]
        self, inputs: ShellInput, context: Execution
    ) -> ShellOutput:
        """Execute shell command.

        Returns:
            ShellOutput with exit_code, stdout, stderr, and custom outputs

        Raises:
            FileNotFoundError: If working directory doesn't exist
            TimeoutError: If command times out
            Exception: For other execution failures
        """
        # Resolve interpolatable fields to their actual types
        timeout = resolve_interpolatable_numeric(inputs.timeout, int, "timeout", ge=1, le=3600)

        # Prepare working directory
        cwd = Path(inputs.working_dir) if inputs.working_dir else Path.cwd()
        if not cwd.exists():
            raise FileNotFoundError(f"Working directory does not exist: {cwd}")

        # Get workflow-scoped scratch directory from execution context
        scratch_dir = context.scratch_dir
        if scratch_dir is None:
            raise RuntimeError(
                "Scratch directory not initialized in execution context. "
                "This indicates a workflow runner initialization issue."
            )

        # Prepare environment with SCRATCH (absolute path)
        env = dict(os.environ)
        if inputs.env:
            env.update(inputs.env)

        # Execute command
        if inputs.shell:
            # Execute via shell (supports pipes, redirects, etc.)
            process = await asyncio.create_subprocess_shell(
                inputs.command,
                stdout=asyncio.subprocess.PIPE if inputs.capture_output else None,
                stderr=asyncio.subprocess.PIPE if inputs.capture_output else None,
                cwd=cwd,
                env=env,
            )
        else:
            # Execute directly (safer, but no shell features)
            args = shlex.split(inputs.command)
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE if inputs.capture_output else None,
                stderr=asyncio.subprocess.PIPE if inputs.capture_output else None,
                cwd=cwd,
                env=env,
            )

        # Wait for completion with timeout
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(f"Command timed out after {timeout} seconds: {inputs.command}")

        # Decode output
        stdout = stdout_bytes.decode("utf-8") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8") if stderr_bytes else ""
        exit_code = process.returncode or 0

        # Build output dict with default fields
        output_dict = {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
        }

        # Read and merge custom outputs (if declared)
        # Access from context variable (async-safe, no race condition)
        custom_outputs = block_custom_outputs.get()
        if custom_outputs:
            for output_name, output_schema in custom_outputs.items():
                try:
                    raw_path = output_schema["path"]

                    # The variable {{tmp}} is resolved before this executor is called,
                    # so raw_path can be an absolute path.
                    # We need to check if the path is inside the scratch directory.
                    is_in_scratch = False
                    if Path(raw_path).is_absolute():
                        try:
                            Path(raw_path).relative_to(scratch_dir)
                            is_in_scratch = True
                        except ValueError:
                            is_in_scratch = False

                    allow_absolute = output_schema.get("unsafe", False) or is_in_scratch

                    # Validate path
                    file_path = validate_output_path(
                        output_name,
                        raw_path,
                        cwd,
                        allow_absolute,
                    )

                    # Read file
                    content = file_path.read_text()

                    # Parse type
                    value = parse_output_value(content, output_schema["type"])

                    # Merge directly into output dict
                    output_dict[output_name] = value

                except (OutputSecurityError, OutputNotFoundError, ValueError) as e:
                    if output_schema.get("required", True):
                        raise ValueError(f"Output '{output_name}' error: {e}")
                    # Optional output, continue without it

        # Create output with merged fields (extra="allow" handles custom fields)
        return ShellOutput(**output_dict)  # type: ignore[arg-type]
