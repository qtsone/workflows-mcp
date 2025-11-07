"""File operation executors for ADR-006 - CreateFile, ReadFile, RenderTemplate.

Architecture (ADR-006):
- Execute returns output directly (no Result wrapper)
- Raises exceptions for failures
- Uses Execution context (not dict)
- Orchestrator creates Metadata based on success/exceptions
"""

from __future__ import annotations

from typing import Any, ClassVar

from jinja2 import Environment, StrictUndefined
from pydantic import Field, field_validator

from .block import BlockInput, BlockOutput
from .block_utils import FileOperations, PathResolver
from .execution import Execution
from .executor_base import (
    BlockExecutor,
    ExecutorCapabilities,
    ExecutorSecurityLevel,
)
from .interpolation import (
    interpolatable_boolean_validator,
    resolve_interpolatable_boolean,
)

# ============================================================================
# CreateFile Executor
# ============================================================================


class CreateFileInput(BlockInput):
    """Input model for CreateFile executor."""

    path: str = Field(description="File path (absolute or relative)")
    content: str = Field(description="File content to write")
    encoding: str = Field(default="utf-8", description="Text encoding")
    mode: int | str | None = Field(
        default=None,
        description="File permissions (Unix only, e.g., 0o644, 644, or '644')",
    )
    overwrite: bool | str = Field(
        default=True, description="Whether to overwrite existing file (or interpolation string)"
    )
    create_parents: bool | str = Field(
        default=True,
        description="Create parent directories if missing (or interpolation string)",
    )

    # Validators for boolean fields with interpolation support
    _validate_overwrite = field_validator("overwrite", mode="before")(
        interpolatable_boolean_validator()
    )
    _validate_create_parents = field_validator("create_parents", mode="before")(
        interpolatable_boolean_validator()
    )


class CreateFileOutput(BlockOutput):
    """Output model for CreateFile executor.

    All fields have defaults to support graceful degradation when file creation fails.
    A default-constructed instance represents a failed/crashed file creation operation.
    """

    path: str = Field(
        default="",
        description="Absolute path to created file (empty string if failed)",
    )
    size_bytes: int = Field(
        default=0,
        description="File size in bytes (0 if failed)",
    )
    created: bool = Field(
        default=False,
        description="True if file was created, False if overwritten or failed",
    )


class CreateFileExecutor(BlockExecutor):
    """
    File creation executor.

    Architecture (ADR-006):
    - Returns CreateFileOutput directly
    - Raises exceptions for failures (ValueError, FileExistsError, etc.)
    - Uses Execution context

    Features:
    - Write content to file path (absolute or relative)
    - Support text encoding modes
    - Create parent directories automatically (optional)
    - Overwrite protection (optional, default: allow overwrite)
    - File permissions setting (optional, Unix-style)
    - Path traversal protection via PathResolver
    """

    type_name: ClassVar[str] = "CreateFile"
    input_type: ClassVar[type[BlockInput]] = CreateFileInput
    output_type: ClassVar[type[BlockOutput]] = CreateFileOutput

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(can_write_files=True)

    async def execute(  # type: ignore[override]
        self, inputs: CreateFileInput, context: Execution
    ) -> CreateFileOutput:
        """Create file with content.

        Returns:
            CreateFileOutput with path, size, created flag

        Raises:
            ValueError: Invalid path or mode
            FileExistsError: File exists and overwrite=False
            Exception: Other I/O errors
        """
        # Resolve interpolatable fields to their actual types
        overwrite = resolve_interpolatable_boolean(inputs.overwrite, "overwrite")
        create_parents = resolve_interpolatable_boolean(inputs.create_parents, "create_parents")

        # Resolve path with security validation
        path_result = PathResolver.resolve_and_validate(inputs.path, allow_traversal=True)
        if not path_result.is_success:
            raise ValueError(f"Invalid path: {path_result.error}")

        # Type narrowing: is_success guarantees value is not None
        assert path_result.value is not None
        file_path = path_result.value

        # Check overwrite protection
        file_existed = file_path.exists()
        if file_existed and not overwrite:
            raise FileExistsError(f"File exists and overwrite=False: {file_path}")

        # Convert mode to integer if it's a string
        mode_int: int | None = None
        if inputs.mode is not None:
            try:
                if isinstance(inputs.mode, str):
                    # Convert string like "644" to octal integer 0o644
                    mode_int = int(inputs.mode, 8)
                else:
                    mode_int = inputs.mode
            except ValueError as e:
                raise ValueError(
                    f"Invalid mode value: {inputs.mode}. "
                    f"Expected octal string (e.g., '644') or integer (e.g., 0o644): {e}"
                ) from e

        # Write file using utility
        write_result = FileOperations.write_text(
            path=file_path,
            content=inputs.content,
            encoding=inputs.encoding,
            mode=mode_int,
            create_parents=create_parents,
        )

        if not write_result.is_success:
            raise OSError(write_result.error)

        # Build output
        assert write_result.value is not None
        return CreateFileOutput(
            path=str(file_path),
            size_bytes=write_result.value,
            created=(not file_existed),
        )


# ============================================================================
# ReadFile Executor
# ============================================================================


class ReadFileInput(BlockInput):
    """Input model for ReadFile executor."""

    path: str = Field(description="File path to read (absolute or relative)")
    encoding: str = Field(default="utf-8", description="Text encoding")
    required: bool | str = Field(
        default=True,
        description=(
            "If False, missing file returns empty content instead of error "
            "(or interpolation string)"
        ),
    )

    # Validator for boolean field with interpolation support
    _validate_required = field_validator("required", mode="before")(
        interpolatable_boolean_validator()
    )


class ReadFileOutput(BlockOutput):
    """Output model for ReadFile executor.

    All fields have defaults to support graceful degradation when file reading fails.
    A default-constructed instance represents a failed/crashed file read operation.
    """

    content: str = Field(
        default="",
        description="File content (empty string if failed)",
    )
    path: str = Field(
        default="",
        description="Absolute path to file (empty string if failed)",
    )
    size_bytes: int = Field(
        default=0,
        description="File size in bytes (0 if failed or not found)",
    )
    found: bool = Field(
        default=False,
        description="True if file was found, False if missing (required=False) or failed",
    )


class ReadFileExecutor(BlockExecutor):
    """
    File reading executor.

    Architecture (ADR-006):
    - Returns ReadFileOutput directly
    - Raises FileNotFoundError if required=True and file missing
    - Returns empty content if required=False and file missing
    """

    type_name: ClassVar[str] = "ReadFile"
    input_type: ClassVar[type[BlockInput]] = ReadFileInput
    output_type: ClassVar[type[BlockOutput]] = ReadFileOutput

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(can_read_files=True)

    async def execute(  # type: ignore[override]
        self, inputs: ReadFileInput, context: Execution
    ) -> ReadFileOutput:
        """Read file content.

        Returns:
            ReadFileOutput with content, path, size, found flag

        Raises:
            ValueError: Invalid path
            FileNotFoundError: File not found and required=True
            Exception: Other I/O errors
        """
        # Resolve interpolatable fields to their actual types
        required = resolve_interpolatable_boolean(inputs.required, "required")

        # Resolve path
        path_result = PathResolver.resolve_and_validate(inputs.path, allow_traversal=True)
        if not path_result.is_success:
            raise ValueError(f"Invalid path: {path_result.error}")

        # Type narrowing: is_success guarantees value is not None
        assert path_result.value is not None
        file_path = path_result.value

        # Check if file exists
        if not file_path.exists():
            if required:
                raise FileNotFoundError(f"File not found: {file_path}")
            else:
                # Graceful: return empty content
                return ReadFileOutput(
                    content="",
                    path=str(file_path),
                    size_bytes=0,
                    found=False,
                )

        # Read file using utility
        read_result = FileOperations.read_text(
            path=file_path,
            encoding=inputs.encoding,
        )

        if not read_result.is_success:
            raise OSError(read_result.error)

        # Build output
        assert read_result.value is not None
        return ReadFileOutput(
            content=read_result.value,
            path=str(file_path),
            size_bytes=file_path.stat().st_size,
            found=True,
        )


# ============================================================================
# RenderTemplate Executor
# ============================================================================


class RenderTemplateInput(BlockInput):
    """Input model for RenderTemplate executor."""

    template: str = Field(description="Jinja2 template string")
    variables: dict[str, Any] = Field(
        default_factory=dict,
        description="Variables to substitute in template",
    )
    output_path: str | None = Field(
        default=None,
        description="Optional file path to write rendered content",
    )
    encoding: str = Field(default="utf-8", description="Text encoding for output file")
    overwrite: bool = Field(
        default=True,
        description="Whether to overwrite existing output file",
    )
    create_parents: bool = Field(
        default=True,
        description="Create parent directories for output file",
    )


class RenderTemplateOutput(BlockOutput):
    """Output model for RenderTemplate executor.

    All fields have defaults to support graceful degradation when template rendering fails.
    A default-constructed instance represents a failed/crashed rendering operation.
    """

    content: str = Field(
        default="",
        description="Rendered template content (empty string if failed)",
    )
    output_path: str | None = Field(
        default=None,
        description="Absolute path to output file (None if not specified or failed)",
    )
    size_bytes: int | None = Field(
        default=None,
        description="Output file size in bytes (None if not written or failed)",
    )


class RenderTemplateExecutor(BlockExecutor):
    """
    Jinja2 template rendering executor.

    Architecture (ADR-006):
    - Returns RenderTemplateOutput directly
    - Raises TemplateSyntaxError, UndefinedError for template issues
    - Raises exceptions for file write failures
    """

    type_name: ClassVar[str] = "RenderTemplate"
    input_type: ClassVar[type[BlockInput]] = RenderTemplateInput
    output_type: ClassVar[type[BlockOutput]] = RenderTemplateOutput

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(
        can_read_files=True,
        can_write_files=True,
    )

    async def execute(  # type: ignore[override]
        self, inputs: RenderTemplateInput, context: Execution
    ) -> RenderTemplateOutput:
        """RenderTemplate Jinja2 template.

        Returns:
            RenderTemplateOutput with rendered content and optional file path

        Raises:
            TemplateSyntaxError: Invalid template syntax
            UndefinedError: Undefined variable in template
            ValueError: Invalid output path
            FileExistsError: Output file exists and overwrite=False
            Exception: Other errors
        """
        # RenderTemplate template (exceptions bubble up)
        env = Environment(undefined=StrictUndefined, autoescape=False)
        template = env.from_string(inputs.template)
        rendered = template.render(**inputs.variables)

        # Write to file if output_path specified
        output_path_str: str | None = None
        size_bytes: int | None = None

        if inputs.output_path:
            # Resolve path
            path_result = PathResolver.resolve_and_validate(
                inputs.output_path, allow_traversal=True
            )
            if not path_result.is_success:
                raise ValueError(f"Invalid output_path: {path_result.error}")

            # Type narrowing: is_success guarantees value is not None
            assert path_result.value is not None
            file_path = path_result.value

            # Check overwrite protection
            if file_path.exists() and not inputs.overwrite:
                raise FileExistsError(f"Output file exists and overwrite=False: {file_path}")

            # Write file using utility
            write_result = FileOperations.write_text(
                path=file_path,
                content=rendered,
                encoding=inputs.encoding,
                mode=None,
                create_parents=inputs.create_parents,
            )

            if not write_result.is_success:
                raise OSError(write_result.error)

            output_path_str = str(file_path)
            size_bytes = write_result.value

        # Build output
        return RenderTemplateOutput(
            content=rendered,
            output_path=output_path_str,
            size_bytes=size_bytes,
        )
