"""JSON state management executors for ADR-006.

Architecture (ADR-006):
- Execute returns output directly (no Result wrapper)
- Raises exceptions for failures
- Uses Execution context (not dict)
- Orchestrator creates Metadata based on success/exceptions
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from .block import BlockInput, BlockOutput
from .block_utils import JSONOperations, PathResolver
from .execution import Execution
from .executor_base import (
    BlockExecutor,
    ExecutorCapabilities,
    ExecutorSecurityLevel,
)

# ============================================================================
# ReadJSONState Executor
# ============================================================================


class ReadJSONStateInput(BlockInput):
    """Input for ReadJSONState executor."""

    path: str = Field(description="Path to JSON file")
    required: bool = Field(
        default=False, description="Whether file must exist (False returns empty dict)"
    )


class ReadJSONStateOutput(BlockOutput):
    """Output for ReadJSONState executor."""

    data: dict[str, Any] = Field(description="JSON data from file")
    found: bool = Field(description="Whether file was found")
    path: str = Field(description="Absolute path to file")


class ReadJSONStateExecutor(BlockExecutor):
    """
    Read JSON state file executor.

    Architecture (ADR-006):
    - Returns ReadJSONStateOutput directly
    - Raises FileNotFoundError if required=True and file missing
    - Returns empty dict if required=False and file missing
    """

    type_name: ClassVar[str] = "ReadJSONState"
    input_type: ClassVar[type[BlockInput]] = ReadJSONStateInput
    output_type: ClassVar[type[BlockOutput]] = ReadJSONStateOutput

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.SAFE
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(can_read_files=True)

    async def execute(  # type: ignore[override]
        self, inputs: ReadJSONStateInput, context: Execution
    ) -> ReadJSONStateOutput:
        """Read JSON state file.

        Returns:
            ReadJSONStateOutput with data, found flag, and path

        Raises:
            ValueError: Invalid path
            FileNotFoundError: File not found and required=True
            Exception: JSON parsing errors or I/O errors
        """
        # Resolve path with security validation
        path_result = PathResolver.resolve_and_validate(inputs.path, allow_traversal=True)
        if not path_result.is_success:
            raise ValueError(f"Invalid path: {path_result.error}")

        file_path = path_result.value
        assert file_path is not None

        # Read JSON using utility (handles missing files gracefully)
        read_result = JSONOperations.read_json(file_path, required=inputs.required)

        if not read_result.is_success:
            assert read_result.error is not None
            # JSONOperations returns specific error messages
            # Convert to appropriate exception type
            if "not found" in read_result.error.lower():
                raise FileNotFoundError(read_result.error)
            else:
                raise ValueError(read_result.error)

        # Build output
        assert read_result.value is not None
        return ReadJSONStateOutput(
            data=read_result.value,
            found=file_path.exists(),
            path=str(file_path),
        )


# ============================================================================
# WriteJSONState Executor
# ============================================================================


class WriteJSONStateInput(BlockInput):
    """Input for WriteJSONState executor."""

    path: str = Field(description="Path to JSON file")
    data: dict[str, Any] = Field(description="JSON data to write")
    create_parents: bool = Field(default=True, description="Create parent directories if missing")


class WriteJSONStateOutput(BlockOutput):
    """Output for WriteJSONState executor."""

    path: str = Field(description="Absolute path to file")
    size_bytes: int = Field(description="Size of written file in bytes")


class WriteJSONStateExecutor(BlockExecutor):
    """
    Write JSON state file executor.

    Architecture (ADR-006):
    - Returns WriteJSONStateOutput directly
    - Raises exceptions for failures
    """

    type_name: ClassVar[str] = "WriteJSONState"
    input_type: ClassVar[type[BlockInput]] = WriteJSONStateInput
    output_type: ClassVar[type[BlockOutput]] = WriteJSONStateOutput

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(can_write_files=True)

    async def execute(  # type: ignore[override]
        self, inputs: WriteJSONStateInput, context: Execution
    ) -> WriteJSONStateOutput:
        """Write JSON state file.

        Returns:
            WriteJSONStateOutput with path and size

        Raises:
            ValueError: Invalid path
            OSError: Failed to create directories or write file
            Exception: JSON serialization errors
        """
        # Resolve path with security validation
        path_result = PathResolver.resolve_and_validate(inputs.path, allow_traversal=True)
        if not path_result.is_success:
            raise ValueError(f"Invalid path: {path_result.error}")

        file_path = path_result.value
        assert file_path is not None

        # Create parents if needed
        if inputs.create_parents:
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise OSError(f"Failed to create parent directories: {e}") from e
        elif not file_path.parent.exists():
            raise FileNotFoundError(f"Parent directory missing: {file_path.parent}")

        # Write JSON using utility
        write_result = JSONOperations.write_json(file_path, inputs.data)

        if not write_result.is_success:
            assert write_result.error is not None
            raise OSError(write_result.error)

        # Get file size
        try:
            size_bytes = file_path.stat().st_size
        except OSError as e:
            raise OSError(f"Failed to get file size: {e}") from e

        # Build output
        return WriteJSONStateOutput(
            path=str(file_path),
            size_bytes=size_bytes,
        )


# ============================================================================
# MergeJSONState Executor
# ============================================================================


class MergeJSONStateInput(BlockInput):
    """Input for MergeJSONState executor."""

    path: str = Field(description="Path to JSON file")
    updates: dict[str, Any] = Field(description="Updates to merge")
    create_if_missing: bool = Field(default=True, description="Create file if it doesn't exist")
    create_parents: bool = Field(default=True, description="Create parent directories if missing")


class MergeJSONStateOutput(BlockOutput):
    """Output for MergeJSONState executor."""

    path: str = Field(description="Absolute path to file")
    created: bool = Field(description="Whether file was created (vs updated)")
    merged_data: dict[str, Any] = Field(description="Result after merge")


class MergeJSONStateExecutor(BlockExecutor):
    """
    Merge JSON state file executor.

    Performs a deep merge of updates into existing JSON state.
    Nested dictionaries are merged recursively, while other values
    (lists, primitives) are replaced.

    Architecture (ADR-006):
    - Returns MergeJSONStateOutput directly
    - Raises exceptions for failures
    """

    type_name: ClassVar[str] = "MergeJSONState"
    input_type: ClassVar[type[BlockInput]] = MergeJSONStateInput
    output_type: ClassVar[type[BlockOutput]] = MergeJSONStateOutput

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(
        can_read_files=True,
        can_write_files=True,
    )

    async def execute(  # type: ignore[override]
        self, inputs: MergeJSONStateInput, context: Execution
    ) -> MergeJSONStateOutput:
        """Merge updates into JSON state file.

        Returns:
            MergeJSONStateOutput with path, created flag, and merged data

        Raises:
            ValueError: Invalid path
            FileNotFoundError: File missing and create_if_missing=False
            OSError: Failed to create directories or write file
            Exception: JSON errors
        """
        # Resolve path with security validation
        path_result = PathResolver.resolve_and_validate(inputs.path, allow_traversal=True)
        if not path_result.is_success:
            raise ValueError(f"Invalid path: {path_result.error}")

        file_path = path_result.value
        assert file_path is not None

        # Check if file exists
        file_existed = file_path.exists()
        if not file_existed and not inputs.create_if_missing:
            raise FileNotFoundError(f"File does not exist and create_if_missing=False: {file_path}")

        # Read existing data or start with empty dict
        existing_data: dict[str, Any] = {}
        if file_existed:
            read_result = JSONOperations.read_json(file_path, required=False)
            if not read_result.is_success:
                raise ValueError(f"Failed to read existing JSON: {read_result.error}")
            assert read_result.value is not None
            existing_data = read_result.value

        # Deep merge function
        def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
            """Deep merge updates into base dict."""
            result = base.copy()
            for key, value in updates.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        # Merge updates
        merged_data = deep_merge(existing_data, inputs.updates)

        # Create parent directories if needed
        if inputs.create_parents:
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise OSError(f"Failed to create parent directories: {e}") from e
        elif not file_path.parent.exists():
            raise FileNotFoundError(f"Parent directory missing: {file_path.parent}")

        # Write merged data
        write_result = JSONOperations.write_json(file_path, merged_data)
        if not write_result.is_success:
            raise OSError(write_result.error)

        # Build output
        return MergeJSONStateOutput(
            path=str(file_path),
            created=not file_existed,
            merged_data=merged_data,
        )
