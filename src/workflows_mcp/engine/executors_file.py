"""File operation executors - CreateFile, ReadFiles, EditFile.

Architecture:
- Execute returns output directly (no Result wrapper)
- Raises exceptions for failures
- Uses Execution context (not dict)
- Orchestrator creates Metadata based on success/exceptions
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, Field, computed_field, field_validator

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
    interpolatable_literal_validator,
    interpolatable_numeric_validator,
    resolve_interpolatable_boolean,
    resolve_interpolatable_literal,
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
    content: str = Field(
        default="",
        description="Content written to the file (empty string if failed)",
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
            content=inputs.content,
        )


# ============================================================================
# ReadFiles Executor
# ============================================================================


class FileInfo(BaseModel):
    """File content and metadata."""

    path: str = Field(description="Relative path from base_path")
    content: str = Field(description="File content (full/outline/summary)")
    size_bytes: int = Field(description="File size in bytes")


class SkippedFileInfo(BaseModel):
    """Information about skipped files."""

    path: str = Field(description="Relative path from base_path")
    reason: str = Field(description="Reason for skipping (e.g., 'too_large', 'binary', 'excluded')")
    details: str | None = Field(
        default=None,
        description="Additional details (e.g., 'size: 500KB > 100KB limit')",
    )


class ReadFilesInput(BlockInput):
    """Input model for ReadFiles executor with full interpolation support."""

    patterns: list[str] = Field(
        description="Glob patterns for files to read (e.g., ['*.py', '**/*.ts', 'docs/**/*.md'])",
        min_length=1,
        max_length=50,
    )

    base_path: str = Field(
        default=".",
        description="Base directory to search from (relative or absolute)",
    )

    mode: Literal["full", "outline", "summary"] | str = Field(
        default="full",
        description=(
            "Output mode: 'full' (complete content), "
            "'outline' (symbol tree with line ranges), "
            "'summary' (outline + docstrings)"
        ),
    )

    exclude_patterns: list[str] = Field(
        default_factory=list,
        description="Additional patterns to exclude beyond defaults (e.g., ['*test*', '*.min.js'])",
    )

    max_files: int | str = Field(
        default=20,
        description="Maximum number of files to read (1-100, supports interpolation)",
    )

    max_file_size_kb: int | str = Field(
        default=100,
        description="Maximum individual file size in KB (supports interpolation)",
    )

    respect_gitignore: bool | str = Field(
        default=True,
        description="Whether to respect .gitignore patterns (supports interpolation)",
    )

    encoding: str = Field(default="utf-8", description="Text encoding for reading files")

    # Validators for interpolatable fields
    _validate_mode = field_validator("mode", mode="before")(
        interpolatable_literal_validator("full", "outline", "summary")
    )

    _validate_max_files = field_validator("max_files", mode="before")(
        interpolatable_numeric_validator(int, ge=1, le=100)
    )

    _validate_max_file_size_kb = field_validator("max_file_size_kb", mode="before")(
        interpolatable_numeric_validator(int, ge=1, le=10240)  # Max 10MB
    )

    _validate_respect_gitignore = field_validator("respect_gitignore", mode="before")(
        interpolatable_boolean_validator()
    )


class ReadFilesOutput(BlockOutput):
    """Output model for ReadFiles executor with YAML-formatted content."""

    files: list[FileInfo] = Field(
        default_factory=list,
        description="List of successfully processed files with content",
    )

    total_files: int = Field(
        default=0,
        description="Number of files successfully processed",
    )

    total_size_kb: int = Field(
        default=0,
        description="Total size in KB of all processed files",
    )

    skipped_files: list[SkippedFileInfo] = Field(
        default_factory=list,
        description="Files that were skipped (too large, binary, excluded, etc.)",
    )

    patterns_matched: int = Field(
        default=0,
        description="Total number of files matching patterns before filtering",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content(self) -> str:
        """Content output - simplified for single file, YAML for multiple files.

        No files: Returns empty string.
        Single file: Returns the file content directly (string).
        Multiple files: Returns YAML-formatted output with file list.
        Single source of truth: files list.
        """
        # No files: return empty string
        if not self.files:
            return ""

        # Single file: return content directly
        if len(self.files) == 1:
            return self.files[0].content

        # Multiple files: use YAML format
        return self._format_as_yaml()

    def _format_as_yaml(self) -> str:
        """Format files as YAML with literal block scalars.

        Uses PyYAML safe_dump with custom representer for '|' style.
        Industry standard approach from PyYAML documentation.
        """
        if not self.files:
            return "files: []"

        # Custom representer for literal block scalars
        def str_representer(dumper: Any, data: str) -> Any:
            if "\n" in data:  # Multi-line: use literal style
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        yaml.add_representer(str, str_representer, Dumper=yaml.SafeDumper)

        # Convert to plain dicts (Pydantic → dict)
        files_data = [
            {
                "path": f.path,
                "content": f.content,
                "size_bytes": f.size_bytes,
            }
            for f in self.files
        ]

        return yaml.safe_dump(
            {"files": files_data},
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,  # Preserve order
        )


class ReadFilesExecutor(BlockExecutor):
    """File reading executor with multi-file and outline support.

    Architecture (ADR-006):
    - Returns ReadFilesOutput directly
    - Raises exceptions for failures (ValueError, FileNotFoundError, etc.)
    - Uses Execution context

    Features:
    - Read single or multiple files via glob patterns
    - Three modes: full, outline (90-97% reduction), summary
    - Gitignore respect, size limits, exclusion patterns
    - Base64 encoding for binary files
    - AST-based Python outline extraction
    """

    type_name: ClassVar[str] = "ReadFiles"
    input_type: ClassVar[type[BlockInput]] = ReadFilesInput
    output_type: ClassVar[type[BlockOutput]] = ReadFilesOutput

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(can_read_files=True)

    async def execute(  # type: ignore[override]
        self, inputs: ReadFilesInput, context: Execution
    ) -> ReadFilesOutput:
        """Execute file reading operations.

        Returns:
            ReadFilesOutput with concatenated content and metadata

        Raises:
            ValueError: Invalid patterns or parameters
            FileNotFoundError: No files match patterns
            Exception: Other I/O errors
        """
        # Import file_outline utilities
        from .file_outline import (
            BASE64_ENCODE_EXTENSIONS,
            DEFAULT_EXCLUDE_PATTERNS,
            create_gitignore_spec,
            generate_file_outline,
            is_binary,
            load_gitignore_patterns,
            matches_gitignore,
            matches_pattern,
        )

        # 1. Resolve interpolatable fields
        mode = resolve_interpolatable_literal(inputs.mode, ("full", "outline", "summary"), "mode")
        max_files = resolve_interpolatable_numeric(inputs.max_files, int, "max_files", ge=1, le=100)
        max_file_size_kb = resolve_interpolatable_numeric(
            inputs.max_file_size_kb, int, "max_file_size_kb", ge=1, le=10240
        )
        respect_gitignore = resolve_interpolatable_boolean(
            inputs.respect_gitignore, "respect_gitignore"
        )

        # 2. Resolve and validate base_path
        base_path_result = PathResolver.resolve_and_validate(inputs.base_path, allow_traversal=True)
        if not base_path_result.is_success:
            raise ValueError(f"Invalid base_path: {base_path_result.error}")

        assert base_path_result.value is not None
        base_path = base_path_result.value

        if not base_path.is_dir():
            raise ValueError(f"base_path is not a directory: {base_path}")

        # 3. Build exclusion patterns
        exclude_patterns = list(DEFAULT_EXCLUDE_PATTERNS)
        exclude_patterns.extend(inputs.exclude_patterns)

        gitignore_spec = None
        if respect_gitignore:
            gitignore_patterns = load_gitignore_patterns(base_path)
            if gitignore_patterns:
                # Try to use pathspec if available
                gitignore_spec = create_gitignore_spec(gitignore_patterns)
                if gitignore_spec is None:
                    # Fallback: add to exclude_patterns
                    exclude_patterns.extend(gitignore_patterns)

        # 4. Find files matching patterns
        found_files: set[Path] = set()
        for pattern in inputs.patterns:
            try:
                for file_path in base_path.glob(pattern):
                    if file_path.is_file():
                        found_files.add(file_path)
            except ValueError as e:
                raise ValueError(f"Invalid glob pattern '{pattern}': {e}") from e

        patterns_matched = len(found_files)

        # 5. Apply exclusion filters
        filtered_files: set[Path] = set()
        for file_path in found_files:
            # Check gitignore first (if pathspec available)
            if gitignore_spec and matches_gitignore(file_path, base_path, gitignore_spec):
                continue

            # Check other exclusion patterns
            is_excluded = any(
                matches_pattern(file_path, base_path, pattern) for pattern in exclude_patterns
            )
            if not is_excluded:
                filtered_files.add(file_path)

        # 6. Sort and limit files
        sorted_files = sorted(list(filtered_files))[:max_files]

        if not sorted_files:
            raise FileNotFoundError(
                f"No files matched patterns {inputs.patterns} in {base_path} after filtering"
            )

        # 7. Process files
        file_count = 0
        total_size_kb = 0
        processed_files: list[FileInfo] = []
        skipped_files: list[SkippedFileInfo] = []

        for file_path in sorted_files:
            try:
                # Check file size
                file_size_kb = file_path.stat().st_size // 1024

                if file_size_kb > max_file_size_kb:
                    skipped_files.append(
                        SkippedFileInfo(
                            path=str(file_path.relative_to(base_path)),
                            reason="too_large",
                            details=f"size: {file_size_kb}KB > {max_file_size_kb}KB",
                        )
                    )
                    continue

                total_size_kb += file_size_kb
                relative_path = file_path.relative_to(base_path)
                file_ext = file_path.suffix.lower()

                # Determine how to read file based on mode
                file_content: str
                if mode in ("outline", "summary"):
                    # Generate outline instead of full content
                    file_content = generate_file_outline(file_path, mode)

                elif file_ext in BASE64_ENCODE_EXTENSIONS:
                    # Base64 encode binary files
                    import base64

                    with open(file_path, "rb") as f:
                        file_content = base64.b64encode(f.read()).decode("ascii")

                elif is_binary(file_path):
                    skipped_files.append(
                        SkippedFileInfo(
                            path=str(relative_path),
                            reason="binary",
                            details="Binary file detected (contains null bytes)",
                        )
                    )
                    continue

                else:
                    # Read as text
                    read_result = FileOperations.read_text(
                        path=file_path,
                        encoding=inputs.encoding,
                    )
                    if not read_result.is_success:
                        raise OSError(read_result.error)

                    assert read_result.value is not None
                    file_content = read_result.value

                file_count += 1
                processed_files.append(
                    FileInfo(
                        path=str(relative_path),
                        content=file_content,
                        size_bytes=file_path.stat().st_size,
                    )
                )

            except Exception as e:
                skipped_files.append(
                    SkippedFileInfo(
                        path=str(file_path.relative_to(base_path)),
                        reason="error",
                        details=f"{type(e).__name__}: {e}",
                    )
                )
                continue

        # 8. Return output (content computed via property)
        return ReadFilesOutput(
            files=processed_files,
            total_files=file_count,
            total_size_kb=total_size_kb,
            skipped_files=skipped_files,
            patterns_matched=patterns_matched,
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
    _validate_count = field_validator("count", mode="before")(interpolatable_numeric_validator(int))
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
    _validate_backup = field_validator("backup", mode="before")(interpolatable_boolean_validator())
    _validate_dry_run = field_validator("dry_run", mode="before")(
        interpolatable_boolean_validator()
    )
    _validate_atomic = field_validator("atomic", mode="before")(interpolatable_boolean_validator())


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
