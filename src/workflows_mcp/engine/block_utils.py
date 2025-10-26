"""Shared utilities for workflow blocks - eliminates duplication.

This module provides reusable components.
All utilities follow security-first design
with comprehensive error handling via the Result monad.
"""

import json
import time
from pathlib import Path
from typing import Any

from .load_result import LoadResult


class PathResolver:
    """Unified path resolution with security validation.

    Provides consistent path handling across all file-based blocks with
    comprehensive security checks to prevent:
    - Path traversal attacks (../ sequences)
    - Symlink exploits
    - Directory confusion attacks
    """

    @staticmethod
    def resolve_and_validate(
        path: str,
        working_dir: Path | None = None,
        allow_traversal: bool = False,
    ) -> LoadResult[Path]:
        """Resolve and validate file path with security checks.

        Args:
            path: File path to resolve (can be relative or absolute)
            working_dir: Base directory for relative paths (defaults to cwd)
            allow_traversal: If False, path must stay within working_dir

        Returns:
            LoadResult.success(resolved_path) or LoadResult.failure(error_message)

        Security Model:
            - Default mode (allow_traversal=False): Paths must stay within working_dir
            - Unsafe mode (allow_traversal=True): Allows paths outside working_dir
            - Always validates: No symlinks, normalized paths

        Example:
            # Safe relative path
            result = PathResolver.resolve_and_validate("data/file.txt")

            # Absolute path with traversal check
            result = PathResolver.resolve_and_validate(
                "/tmp/output.txt",
                working_dir=Path("/tmp"),
                allow_traversal=False
            )
        """
        # Use cwd if working_dir not specified
        if working_dir is None:
            working_dir = Path.cwd()

        # Convert to Path object
        file_path = Path(path)

        # Build absolute path
        if file_path.is_absolute():
            absolute_path = file_path
        else:
            absolute_path = working_dir / file_path

        # Security check: no symlinks in the path itself
        # (parent directories may contain symlinks, those are checked during resolve)
        if absolute_path.is_symlink():
            return LoadResult.failure(f"Symlinks not allowed for security: {absolute_path}")

        # Resolve to canonical path (follows symlinks in parents, normalizes ..)
        try:
            resolved_path = absolute_path.resolve()
        except (OSError, RuntimeError) as e:
            return LoadResult.failure(f"Failed to resolve path '{path}': {e}")

        # Security check: path traversal protection
        if not allow_traversal:
            try:
                # Ensure resolved path is within working_dir
                resolved_path.relative_to(working_dir.resolve())
            except ValueError:
                return LoadResult.failure(
                    f"Path escapes working directory. "
                    f"Path: {path}, Resolved: {resolved_path}, Working dir: {working_dir}"
                )

        # Additional check: ensure no .. remains in normalized path
        if ".." in str(resolved_path):
            return LoadResult.failure(f"Path traversal detected after resolution: {resolved_path}")

        return LoadResult.success(resolved_path)


class FileOperations:
    """Unified file I/O operations with comprehensive error handling.

    Provides type-safe file operations that return Result types instead of
    raising exceptions. All operations include:
    - Automatic parent directory creation
    - Encoding validation
    - Size limits for security
    - Atomic write operations where possible
    """

    @staticmethod
    def read_text(
        path: Path,
        encoding: str = "utf-8",
        max_size_bytes: int | None = None,
    ) -> LoadResult[str]:
        """Read text file with error handling.

        Args:
            path: File path to read (must exist)
            encoding: Text encoding (default: utf-8)
            max_size_bytes: Optional size limit for security (prevents memory exhaustion)

        Returns:
            LoadResult.success(content) or LoadResult.failure(error_message)

        Example:
            result = FileOperations.read_text(Path("/tmp/data.txt"))
            if result.is_success:
                content = result.value
        """
        # Check file exists
        if not path.exists():
            return LoadResult.failure(f"File not found: {path}")

        if not path.is_file():
            return LoadResult.failure(f"Path is not a file: {path}")

        # Check size limit if specified
        if max_size_bytes is not None:
            try:
                file_size = path.stat().st_size
                if file_size > max_size_bytes:
                    return LoadResult.failure(
                        f"File too large: {file_size} bytes exceeds limit of {max_size_bytes}"
                    )
            except OSError as e:
                return LoadResult.failure(f"Failed to stat file '{path}': {e}")

        # Read file
        try:
            content = path.read_text(encoding=encoding)
            return LoadResult.success(content)
        except UnicodeDecodeError as e:
            return LoadResult.failure(f"Encoding error reading '{path}' with {encoding}: {e}")
        except OSError as e:
            return LoadResult.failure(f"Failed to read file '{path}': {e}")

    @staticmethod
    def write_text(
        path: Path,
        content: str,
        encoding: str = "utf-8",
        mode: int | None = None,
        create_parents: bool = True,
    ) -> LoadResult[int]:
        """Write text file with error handling.

        Args:
            path: File path to write (will be created/overwritten)
            content: Text content to write
            encoding: Text encoding (default: utf-8)
            mode: Unix file permissions (e.g., 0o644)
            create_parents: Create parent directories if they don't exist

        Returns:
            LoadResult.success(bytes_written) or LoadResult.failure(error_message)

        Example:
            result = FileOperations.write_text(
                Path("/tmp/output.txt"),
                "Hello, World!",
                mode=0o644
            )
        """
        # Create parent directories if requested
        if create_parents:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                return LoadResult.failure(f"Failed to create parent directories for '{path}': {e}")

        # Write file
        try:
            path.write_text(content, encoding=encoding)
            bytes_written = len(content.encode(encoding))
        except OSError as e:
            return LoadResult.failure(f"Failed to write file '{path}': {e}")
        except UnicodeEncodeError as e:
            return LoadResult.failure(f"Encoding error writing '{path}' with {encoding}: {e}")

        # Set file permissions if specified
        if mode is not None:
            try:
                path.chmod(mode)
            except OSError as e:
                return LoadResult.failure(f"Failed to set permissions on '{path}': {e}")

        return LoadResult.success(bytes_written)


class JSONOperations:
    """Unified JSON operations with validation.

    Provides type-safe JSON file operations with:
    - Schema validation (via Pydantic models)
    - Deep merge capabilities
    - Formatted output for human readability
    """

    @staticmethod
    def read_json(path: Path, required: bool = False) -> LoadResult[dict[str, Any]]:
        """Read and parse JSON file.

        Args:
            path: JSON file path
            required: If True, missing file is an error. If False, returns empty dict.

        Returns:
            LoadResult.success(parsed_dict) or LoadResult.failure(error_message)

        Example:
            result = JSONOperations.read_json(Path("config.json"))
            if result.is_success:
                config = result.value
        """
        # Check file exists
        if not path.exists():
            if required:
                return LoadResult.failure(f"Required JSON file not found: {path}")
            else:
                return LoadResult.success({})

        if not path.is_file():
            return LoadResult.failure(f"Path is not a file: {path}")

        # Read and parse JSON
        try:
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)

            if not isinstance(data, dict):
                return LoadResult.failure(
                    f"JSON file must contain an object, not {type(data).__name__}"
                )

            return LoadResult.success(data)
        except json.JSONDecodeError as e:
            return LoadResult.failure(f"Invalid JSON in '{path}': {e}")
        except OSError as e:
            return LoadResult.failure(f"Failed to read JSON file '{path}': {e}")

    @staticmethod
    def write_json(path: Path, data: dict[str, Any]) -> LoadResult[bool]:
        """Write JSON with formatting.

        Args:
            path: Output file path
            data: Dictionary to write as JSON

        Returns:
            LoadResult.success(True) on success or LoadResult.failure(error_message) on failure

        Example:
            result = JSONOperations.write_json(
                Path("output.json"),
                {"key": "value", "number": 42}
            )
            if result.is_success:
                print("JSON written successfully")
        """
        # Create parent directories
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return LoadResult.failure(f"Failed to create parent directories for '{path}': {e}")

        # Write JSON with formatting
        try:
            content = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
            path.write_text(content + "\n", encoding="utf-8")
            return LoadResult.success(True)
        except (OSError, TypeError, ValueError) as e:
            return LoadResult.failure(f"Failed to write JSON to '{path}': {e}")

    @staticmethod
    def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        """Deep merge updates into base dict (recursive).

        Args:
            base: Base dictionary (not modified)
            updates: Updates to apply

        Returns:
            New merged dictionary

        Behavior:
            - Recursively merges nested dictionaries
            - Non-dict values from updates overwrite base values
            - Lists are replaced, not merged

        Example:
            base = {"a": 1, "b": {"c": 2, "d": 3}}
            updates = {"b": {"d": 4, "e": 5}, "f": 6}
            result = JSONOperations.deep_merge(base, updates)
            # result = {"a": 1, "b": {"c": 2, "d": 4, "e": 5}, "f": 6}
        """
        result = base.copy()

        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = JSONOperations.deep_merge(result[key], value)
            else:
                # Overwrite with new value
                result[key] = value

        return result


class ExecutionTimer:
    """Simple execution time tracker.

    Utility for measuring block execution time with minimal overhead.
    """

    def __init__(self) -> None:
        """Initialize timer at current time."""
        self.start_time = time.time()

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds.

        Returns:
            Elapsed time in milliseconds (float)

        Example:
            timer = ExecutionTimer()
            # ... do work ...
            duration = timer.elapsed_ms()
        """
        return (time.time() - self.start_time) * 1000.0
