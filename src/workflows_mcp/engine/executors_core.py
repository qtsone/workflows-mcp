"""Core workflow executors - Shell and Workflow."""

import asyncio
import json
import os
import shlex
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from .block import BlockInput, BlockOutput
from .context_vars import block_custom_outputs
from .execution import Execution
from .executor_base import (
    BlockExecutor,
    ExecutorCapabilities,
    ExecutorSecurityLevel,
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
    timeout: int = Field(default=120, description="Timeout in seconds")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    capture_output: bool = Field(default=True, description="Capture stdout/stderr")
    shell: bool = Field(default=True, description="Execute via shell")
    custom_outputs: dict[str, Any] | None = Field(
        default=None,
        description="Custom file-based outputs to read after execution",
        exclude=True,
    )


class ShellOutput(BlockOutput):
    """Output model for Shell executor.

    Custom outputs declared in YAML are merged directly as fields via extra="allow".
    No separate custom_outputs dict needed.
    """

    exit_code: int = Field(description="Process exit code")
    stdout: str = Field(description="Standard output")
    stderr: str = Field(description="Standard error")

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
        output_type: One of ValueType values: str, int, float, bool, json

    Returns:
        Parsed value with correct Python type

    Raises:
        ValueError: If content doesn't match declared type
    """
    content = content.strip()

    if output_type == "str":
        return str(content)  # Explicit cast for consistency
    elif output_type == "int":
        try:
            return int(content)
        except ValueError:
            raise ValueError(f"Cannot parse as int: {content}")
    elif output_type == "float":
        try:
            return float(content)
        except ValueError:
            raise ValueError(f"Cannot parse as float: {content}")
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
        # Prepare working directory
        cwd = Path(inputs.working_dir) if inputs.working_dir else Path.cwd()
        if not cwd.exists():
            raise FileNotFoundError(f"Working directory does not exist: {cwd}")

        # Setup scratch directory
        scratch_dir = cwd / ".scratch"
        scratch_dir.mkdir(exist_ok=True, mode=0o700)

        # Update .gitignore if it exists
        gitignore = cwd / ".gitignore"
        if gitignore.exists():
            content = gitignore.read_text()
            if ".scratch/" not in content:
                with gitignore.open("a") as f:
                    f.write("\n.scratch/\n")

        # Prepare environment with SCRATCH
        env = dict(os.environ)
        if inputs.env:
            env.update(inputs.env)
        env["SCRATCH"] = ".scratch"

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
                process.communicate(), timeout=inputs.timeout
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(
                f"Command timed out after {inputs.timeout} seconds: {inputs.command}"
            )

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
            # Set SCRATCH for path expansion
            original_env = os.environ.get("SCRATCH")
            os.environ["SCRATCH"] = ".scratch"

            try:
                for output_name, output_schema in custom_outputs.items():
                    try:
                        # Validate path
                        file_path = validate_output_path(
                            output_name,
                            output_schema["path"],
                            cwd,
                            output_schema.get("unsafe", False),
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
            finally:
                # Restore original environment
                if original_env is not None:
                    os.environ["SCRATCH"] = original_env
                elif "SCRATCH" in os.environ:
                    del os.environ["SCRATCH"]

        # Create output with merged fields (extra="allow" handles custom fields)
        return ShellOutput(**output_dict)  # type: ignore[arg-type]
