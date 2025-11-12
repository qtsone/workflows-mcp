"""File operation executors for ADR-006 - CreateFile, ReadFile, RenderTemplate, EditFile.

Architecture (ADR-006):
- Execute returns output directly (no Result wrapper)
- Raises exceptions for failures
- Uses Execution context (not dict)
- Orchestrator creates Metadata based on success/exceptions
"""

from __future__ import annotations

import difflib
import re
from typing import Any, ClassVar, Literal

from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel, Field, field_validator

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
    interpolatable_numeric_validator,
    resolve_interpolatable_boolean,
    resolve_interpolatable_numeric,
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
    overwrite: bool | str = Field(
        default=True,
        description="Whether to overwrite existing output file (or interpolation string)",
    )
    create_parents: bool | str = Field(
        default=True,
        description="Create parent directories for output file (or interpolation string)",
    )

    # Validators for boolean fields with interpolation support
    _validate_overwrite = field_validator("overwrite", mode="before")(
        interpolatable_boolean_validator()
    )
    _validate_create_parents = field_validator("create_parents", mode="before")(
        interpolatable_boolean_validator()
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
        # Resolve interpolatable fields to their actual types
        overwrite = resolve_interpolatable_boolean(inputs.overwrite, "overwrite")
        create_parents = resolve_interpolatable_boolean(inputs.create_parents, "create_parents")

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
            if file_path.exists() and not overwrite:
                raise FileExistsError(f"Output file exists and overwrite=False: {file_path}")

            # Write file using utility
            write_result = FileOperations.write_text(
                path=file_path,
                content=rendered,
                encoding=inputs.encoding,
                mode=None,
                create_parents=create_parents,
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


# ============================================================================
# EditFile Executor
# ============================================================================


class EditOperation(BaseModel):
    """Single edit operation - multiple strategies supported.

    This model supports six different edit strategies:
    1. replace_text: Find/replace exact text
    2. replace_lines: Replace line range
    3. insert_lines: Insert at line number
    4. delete_lines: Delete line range
    5. patch: Apply unified diff
    6. regex_replace: Regex find/replace
    """

    type: Literal[
        "replace_text",
        "replace_lines",
        "insert_lines",
        "delete_lines",
        "patch",
        "regex_replace",
    ] = Field(description="Edit operation type")

    # For replace_text
    old_text: str | None = Field(
        default=None,
        description="Text to find and replace (required for replace_text)",
    )
    new_text: str | None = Field(
        default=None,
        description="Replacement text (required for replace_text)",
    )
    count: int | str = Field(
        default=-1,
        description=(
            "Maximum number of replacements (-1 = all occurrences, supports interpolation)"
        ),
    )

    # For replace_lines, insert_lines, delete_lines
    line_start: int | str | None = Field(
        default=None,
        description=(
            "1-indexed line number (required for line operations, supports interpolation)"
        ),
    )
    line_end: int | str | None = Field(
        default=None,
        description=(
            "Inclusive end line number "
            "(required for replace_lines, delete_lines, supports interpolation)"
        ),
    )

    # Validators for numeric fields with interpolation support
    _validate_count = field_validator("count", mode="before")(
        interpolatable_numeric_validator(int)
    )
    _validate_line_start = field_validator("line_start", mode="before")(
        interpolatable_numeric_validator(int, ge=1)
    )
    _validate_line_end = field_validator("line_end", mode="before")(
        interpolatable_numeric_validator(int, ge=1)
    )
    content: str | None = Field(
        default=None,
        description="Content to insert/replace (required for insert_lines, replace_lines)",
    )

    # For patch
    patch: str | None = Field(
        default=None,
        description="Unified diff format patch (required for patch)",
    )

    # For regex_replace
    pattern: str | None = Field(
        default=None,
        description="Regex pattern to match (required for regex_replace)",
    )
    replacement: str | None = Field(
        default=None,
        description="Replacement string (required for regex_replace)",
    )
    flags: list[Literal["IGNORECASE", "MULTILINE", "DOTALL"]] = Field(
        default_factory=list,
        description="Regex flags to apply",
    )


class EditFileInput(BlockInput):
    """Input model for EditFile executor.

    Supports multiple edit strategies with deterministic execution:
    - All edits applied atomically (by default)
    - Backup creation before editing (optional)
    - Dry-run mode for previewing changes (optional)
    - Multiple operations applied sequentially
    """

    path: str = Field(description="Path to file to edit (relative or absolute)")
    operations: list[EditOperation] = Field(
        description="List of edit operations to apply sequentially"
    )
    encoding: str = Field(default="utf-8", description="File encoding")
    create_if_missing: bool | str = Field(
        default=False,
        description="Create file if it doesn't exist (or interpolation string)",
    )
    backup: bool | str = Field(
        default=True,
        description="Create .bak backup before editing (or interpolation string)",
    )
    dry_run: bool | str = Field(
        default=False,
        description="Preview changes without applying (returns diff) (or interpolation string)",
    )
    atomic: bool | str = Field(
        default=True,
        description="All operations succeed or none applied (or interpolation string)",
    )

    # Validators for boolean fields with interpolation support
    _validate_create_if_missing = field_validator("create_if_missing", mode="before")(
        interpolatable_boolean_validator()
    )
    _validate_backup = field_validator("backup", mode="before")(
        interpolatable_boolean_validator()
    )
    _validate_dry_run = field_validator("dry_run", mode="before")(
        interpolatable_boolean_validator()
    )
    _validate_atomic = field_validator("atomic", mode="before")(
        interpolatable_boolean_validator()
    )


class EditFileOutput(BlockOutput):
    """Output model for EditFile executor.

    Provides comprehensive feedback about edit operations:
    - Number of operations applied
    - Line statistics (added/removed/modified)
    - Unified diff of changes
    - Backup file path (if created)
    - Success status and error details
    """

    operations_applied: int = Field(
        default=0,
        description="Number of operations successfully applied",
    )
    lines_added: int = Field(default=0, description="Number of lines added")
    lines_removed: int = Field(default=0, description="Number of lines removed")
    lines_modified: int = Field(default=0, description="Number of lines modified")
    diff: str | None = Field(
        default=None,
        description="Unified diff of changes (always provided)",
    )
    backup_path: str | None = Field(
        default=None,
        description="Path to backup file (if backup=true)",
    )
    success: bool = Field(
        default=True,
        description="True if all operations succeeded",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Error messages if atomic=false and some operations failed",
    )


class EditFileExecutor(BlockExecutor):
    """File editing executor with multiple strategies.

    Architecture (ADR-006):
    - Returns EditFileOutput directly
    - Raises exceptions for failures (ValueError, FileNotFoundError, etc.)
    - Uses Execution context

    Features:
    - Six edit strategies: replace_text, replace_lines, insert_lines,
      delete_lines, patch, regex_replace
    - Atomic transactions (all-or-nothing by default)
    - Backup creation before editing
    - Dry-run mode for previewing changes
    - Comprehensive diff generation
    - Path traversal protection via PathResolver

    Edit Strategies:
    1. replace_text: Simple find/replace for exact text matches
    2. replace_lines: Replace specific line ranges
    3. insert_lines: Insert content at specific line numbers
    4. delete_lines: Remove specific line ranges
    5. patch: Apply unified diff patches
    6. regex_replace: Pattern-based replacements
    """

    type_name: ClassVar[str] = "EditFile"
    input_type: ClassVar[type[BlockInput]] = EditFileInput
    output_type: ClassVar[type[BlockOutput]] = EditFileOutput

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(
        can_read_files=True,
        can_write_files=True,
    )

    async def execute(  # type: ignore[override]
        self, inputs: EditFileInput, context: Execution
    ) -> EditFileOutput:
        """Execute file edit operations.

        Returns:
            EditFileOutput with operations count, statistics, diff, and status

        Raises:
            ValueError: Invalid path, operation parameters, or patch format
            FileNotFoundError: File not found and create_if_missing=False
            Exception: Other I/O errors
        """
        # Resolve interpolatable fields
        create_if_missing = resolve_interpolatable_boolean(
            inputs.create_if_missing, "create_if_missing"
        )
        backup = resolve_interpolatable_boolean(inputs.backup, "backup")
        dry_run = resolve_interpolatable_boolean(inputs.dry_run, "dry_run")
        atomic = resolve_interpolatable_boolean(inputs.atomic, "atomic")

        # Resolve path with security validation
        path_result = PathResolver.resolve_and_validate(inputs.path, allow_traversal=True)
        if not path_result.is_success:
            raise ValueError(f"Invalid path: {path_result.error}")

        assert path_result.value is not None
        file_path = path_result.value

        # Read file or create empty content
        if not file_path.exists():
            if create_if_missing:
                original_content = ""
            else:
                raise FileNotFoundError(f"File not found: {file_path}")
        else:
            read_result = FileOperations.read_text(
                path=file_path,
                encoding=inputs.encoding,
            )
            if not read_result.is_success:
                raise OSError(f"Failed to read file: {read_result.error}")
            assert read_result.value is not None
            original_content = read_result.value

        # Create backup if requested (and not dry-run)
        backup_path_str: str | None = None
        if backup and not dry_run and file_path.exists():
            backup_path = file_path.with_suffix(file_path.suffix + ".bak")
            write_result = FileOperations.write_text(
                path=backup_path,
                content=original_content,
                encoding=inputs.encoding,
                mode=None,
                create_parents=False,
            )
            if not write_result.is_success:
                raise OSError(f"Failed to create backup: {write_result.error}")
            backup_path_str = str(backup_path)

        # Apply operations
        content = original_content
        errors: list[str] = []
        operations_applied = 0

        for i, op in enumerate(inputs.operations):
            try:
                content = self._apply_operation(content, op)
                operations_applied += 1
            except Exception as e:
                error_msg = f"Operation {i} ({op.type}) failed: {e}"
                if atomic:
                    # Atomic mode: revert and fail
                    raise ValueError(error_msg) from e
                else:
                    # Non-atomic: track error and continue
                    errors.append(error_msg)

        # Compute diff
        diff = self._compute_diff(original_content, content, str(file_path))

        # Compute statistics
        stats = self._compute_stats(original_content, content)

        # Write file (unless dry-run)
        if not dry_run:
            write_result = FileOperations.write_text(
                path=file_path,
                content=content,
                encoding=inputs.encoding,
                mode=None,
                create_parents=create_if_missing,
            )
            if not write_result.is_success:
                raise OSError(f"Failed to write file: {write_result.error}")

        # Build output
        return EditFileOutput(
            operations_applied=operations_applied,
            lines_added=stats["added"],
            lines_removed=stats["removed"],
            lines_modified=stats["modified"],
            diff=diff,
            backup_path=backup_path_str,
            success=(len(errors) == 0),
            errors=errors,
        )

    def _apply_operation(self, content: str, op: EditOperation) -> str:
        """Apply single edit operation to content.

        Args:
            content: Current file content
            op: Edit operation to apply

        Returns:
            Modified content

        Raises:
            ValueError: Invalid operation parameters
            Exception: Operation-specific errors
        """
        if op.type == "replace_text":
            if op.old_text is None or op.new_text is None:
                raise ValueError("replace_text requires old_text and new_text")
            # Resolve count to int (handles interpolation)
            count = resolve_interpolatable_numeric(op.count, int, "count")
            return content.replace(op.old_text, op.new_text, count)

        elif op.type == "replace_lines":
            if op.line_start is None or op.line_end is None or op.content is None:
                raise ValueError("replace_lines requires line_start, line_end, and content")
            # Get lines first to determine file-specific bounds
            lines = content.splitlines(keepends=True)
            # Resolve line numbers with file-specific validation
            line_start = resolve_interpolatable_numeric(
                op.line_start, int, "line_start", ge=1, le=len(lines) + 1
            )
            line_end = resolve_interpolatable_numeric(
                op.line_end, int, "line_end", ge=line_start, le=len(lines)
            )
            # Ensure content ends with newline if original had newlines
            replacement = op.content
            if lines and not replacement.endswith("\n"):
                replacement += "\n"
            new_lines = lines[: line_start - 1] + [replacement] + lines[line_end:]
            return "".join(new_lines)

        elif op.type == "insert_lines":
            if op.line_start is None or op.content is None:
                raise ValueError("insert_lines requires line_start and content")
            # Get lines first to determine file-specific bounds
            lines = content.splitlines(keepends=True)
            # Resolve line number with file-specific validation
            line_start = resolve_interpolatable_numeric(
                op.line_start, int, "line_start", ge=1, le=len(lines) + 1
            )
            # Ensure content ends with newline if original had newlines
            insertion = op.content
            if lines and not insertion.endswith("\n"):
                insertion += "\n"
            new_lines = lines[: line_start - 1] + [insertion] + lines[line_start - 1 :]
            return "".join(new_lines)

        elif op.type == "delete_lines":
            if op.line_start is None or op.line_end is None:
                raise ValueError("delete_lines requires line_start and line_end")
            # Get lines first to determine file-specific bounds
            lines = content.splitlines(keepends=True)
            # Resolve line numbers with file-specific validation
            line_start = resolve_interpolatable_numeric(
                op.line_start, int, "line_start", ge=1, le=len(lines)
            )
            line_end = resolve_interpolatable_numeric(
                op.line_end, int, "line_end", ge=line_start, le=len(lines)
            )
            new_lines = lines[: line_start - 1] + lines[line_end:]
            return "".join(new_lines)

        elif op.type == "patch":
            if op.patch is None:
                raise ValueError("patch requires patch content")
            return self._apply_patch(content, op.patch)

        elif op.type == "regex_replace":
            if op.pattern is None or op.replacement is None:
                raise ValueError("regex_replace requires pattern and replacement")
            flags = 0
            for flag_name in op.flags:
                flags |= getattr(re, flag_name)
            return re.sub(op.pattern, op.replacement, content, flags=flags)

        else:
            raise ValueError(f"Unknown operation type: {op.type}")

    def _compute_diff(self, original: str, modified: str, filepath: str) -> str:
        """Generate unified diff between original and modified content.

        Args:
            original: Original file content
            modified: Modified file content
            filepath: File path for diff header

        Returns:
            Unified diff string
        """
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)

        diff_lines = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
            lineterm="",
        )

        return "".join(diff_lines)

    def _compute_stats(self, original: str, modified: str) -> dict[str, int]:
        """Calculate line statistics between original and modified content.

        Args:
            original: Original file content
            modified: Modified file content

        Returns:
            Dictionary with 'added', 'removed', 'modified' counts
        """
        original_lines = original.splitlines()
        modified_lines = modified.splitlines()

        # Use SequenceMatcher to get detailed diff
        matcher = difflib.SequenceMatcher(None, original_lines, modified_lines)

        lines_added: int = 0
        lines_removed: int = 0
        lines_modified: int = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                # Lines were modified
                lines_modified += max(i2 - i1, j2 - j1)
            elif tag == "delete":
                # Lines were removed
                lines_removed += i2 - i1
            elif tag == "insert":
                # Lines were added
                lines_added += j2 - j1
            # 'equal' tag means no change, skip

        return {"added": lines_added, "removed": lines_removed, "modified": lines_modified}

    def _apply_patch(self, content: str, patch_text: str) -> str:
        """Apply unified diff patch to content.

        Args:
            content: Original file content
            patch_text: Unified diff format patch

        Returns:
            Patched content

        Raises:
            ValueError: Invalid patch format or patch application failure
        """
        lines = content.splitlines(keepends=True)
        patch_lines = patch_text.splitlines()

        # Parse patch hunks
        hunks: list[dict[str, Any]] = []
        current_hunk: dict[str, Any] | None = None

        for line in patch_lines:
            # Skip file headers (--- a/file, +++ b/file)
            if line.startswith("---") or line.startswith("+++"):
                continue

            # Hunk header: @@ -start,count +start,count @@
            if line.startswith("@@"):
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if not match:
                    raise ValueError(f"Invalid hunk header: {line}")

                old_start = int(match.group(1))
                old_count = int(match.group(2)) if match.group(2) else 1
                new_start = int(match.group(3))
                new_count = int(match.group(4)) if match.group(4) else 1

                current_hunk = {
                    "old_start": old_start,
                    "old_count": old_count,
                    "new_start": new_start,
                    "new_count": new_count,
                    "changes": [],
                }
                hunks.append(current_hunk)

            elif current_hunk is not None:
                # Change lines: ' ' (context), '-' (remove), '+' (add)
                if line.startswith(" ") or line.startswith("-") or line.startswith("+"):
                    current_hunk["changes"].append(line)
                elif line:  # Non-empty, non-change line
                    raise ValueError(f"Invalid patch line: {line}")

        if not hunks:
            raise ValueError("No valid hunks found in patch")

        # Apply hunks in reverse order to maintain line numbers
        result_lines = lines.copy()

        for hunk in reversed(hunks):
            old_start = hunk["old_start"] - 1  # Convert to 0-indexed
            changes = hunk["changes"]

            # Extract old and new line sequences
            old_lines_in_hunk = []
            new_lines_in_hunk = []

            for change in changes:
                if change.startswith(" "):
                    # Context line (in both old and new)
                    line_content = change[1:] + ("\n" if not change[1:].endswith("\n") else "")
                    old_lines_in_hunk.append(line_content)
                    new_lines_in_hunk.append(line_content)
                elif change.startswith("-"):
                    # Line removed from old
                    line_content = change[1:] + ("\n" if not change[1:].endswith("\n") else "")
                    old_lines_in_hunk.append(line_content)
                elif change.startswith("+"):
                    # Line added to new
                    line_content = change[1:] + ("\n" if not change[1:].endswith("\n") else "")
                    new_lines_in_hunk.append(line_content)

            # Verify context matches
            old_end = old_start + len(old_lines_in_hunk)
            if old_end > len(result_lines):
                raise ValueError(
                    f"Patch hunk extends beyond file end: line {old_end} > {len(result_lines)}"
                )

            for i, expected_line in enumerate(old_lines_in_hunk):
                actual_line = result_lines[old_start + i]
                # Strip trailing newlines for comparison (patch might not include them)
                if actual_line.rstrip("\n") != expected_line.rstrip("\n"):
                    raise ValueError(
                        f"Patch context mismatch at line {old_start + i + 1}: "
                        f"expected {expected_line!r}, got {actual_line!r}"
                    )

            # Apply hunk: replace old lines with new lines
            result_lines[old_start:old_end] = new_lines_in_hunk

        return "".join(result_lines)
