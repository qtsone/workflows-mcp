"""JSON state management executors for ADR-006.

Architecture (ADR-006):
- Execute returns output directly (no Result wrapper)
- Raises exceptions for failures
- Uses Execution context (not dict)
- Orchestrator creates Metadata based on success/exceptions
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field, field_validator

from .block import BlockInput, BlockOutput
from .block_utils import JSONOperations, PathResolver
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
# ReadJSONState Executor
# ============================================================================


class ReadJSONStateInput(BlockInput):
    """Input for ReadJSONState executor."""

    path: str = Field(description="Path to JSON file")
    required: bool | str = Field(
        default=False,
        description="Whether file must exist (False returns empty dict, or interpolation string)",
    )

    # Validator for boolean field with interpolation support
    _validate_required = field_validator("required", mode="before")(
        interpolatable_boolean_validator()
    )


class ReadJSONStateOutput(BlockOutput):
    """Output for ReadJSONState executor.

    All fields have defaults to support graceful degradation when reading fails.
    A default-constructed instance represents a failed/crashed read operation.
    """

    data: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON data from file (empty dict if failed or not found)",
    )
    found: bool = Field(
        default=False,
        description="Whether file was found (False if failed or not found)",
    )
    path: str = Field(
        default="",
        description="Absolute path to file (empty string if failed)",
    )


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
        # Resolve interpolatable fields to their actual types
        required = resolve_interpolatable_boolean(inputs.required, "required")

        # Resolve path with security validation
        path_result = PathResolver.resolve_and_validate(inputs.path, allow_traversal=True)
        if not path_result.is_success:
            raise ValueError(f"Invalid path: {path_result.error}")

        file_path = path_result.value
        assert file_path is not None

        # Read JSON directly (no serialization needed - reads don't cause race conditions)
        read_result = JSONOperations.read_json(file_path, required=required)
        if not read_result.is_success:
            assert read_result.error is not None
            # JSONOperations returns specific error messages
            # Convert to appropriate exception type
            if "not found" in read_result.error.lower():
                raise FileNotFoundError(read_result.error)
            else:
                raise ValueError(read_result.error)
        assert read_result.value is not None
        data = read_result.value
        found = file_path.exists()

        # Build output
        return ReadJSONStateOutput(
            data=data,
            found=found,
            path=str(file_path),
        )


# ============================================================================
# WriteJSONState Executor
# ============================================================================


class WriteJSONStateInput(BlockInput):
    """Input for WriteJSONState executor."""

    path: str = Field(description="Path to JSON file")
    data: dict[str, Any] = Field(description="JSON data to write")
    create_parents: bool | str = Field(
        default=True, description="Create parent directories if missing (or interpolation string)"
    )

    # Validator for boolean field with interpolation support
    _validate_create_parents = field_validator("create_parents", mode="before")(
        interpolatable_boolean_validator()
    )


class WriteJSONStateOutput(BlockOutput):
    """Output for WriteJSONState executor.

    All fields have defaults to support graceful degradation when writing fails.
    A default-constructed instance represents a failed/crashed write operation.
    """

    path: str = Field(
        default="",
        description="Absolute path to file (empty string if failed)",
    )
    size_bytes: int = Field(
        default=0,
        description="Size of written file in bytes (0 if failed)",
    )


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
        # Resolve interpolatable fields to their actual types
        create_parents = resolve_interpolatable_boolean(inputs.create_parents, "create_parents")

        # Resolve path with security validation
        path_result = PathResolver.resolve_and_validate(inputs.path, allow_traversal=True)
        if not path_result.is_success:
            raise ValueError(f"Invalid path: {path_result.error}")

        file_path = path_result.value
        assert file_path is not None

        # Create parents if needed
        if create_parents:
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise OSError(f"Failed to create parent directories: {e}") from e
        elif not file_path.parent.exists():
            raise FileNotFoundError(f"Parent directory missing: {file_path.parent}")

        # Define write operation for serialization
        def write_operation() -> int:
            """Write JSON and return file size."""
            write_result = JSONOperations.write_json(file_path, inputs.data)
            if not write_result.is_success:
                assert write_result.error is not None
                raise OSError(write_result.error)
            return file_path.stat().st_size

        # Execute via IO queue if available (serializes parallel writes)
        if context.execution_context and context.execution_context.io_queue:
            size_bytes = await context.execution_context.io_queue.submit(write_operation)
        else:
            # Fallback: direct execution (backward compatibility)
            size_bytes = write_operation()

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
    create_if_missing: bool | str = Field(
        default=True, description="Create file if it doesn't exist (or interpolation string)"
    )
    create_parents: bool | str = Field(
        default=True, description="Create parent directories if missing (or interpolation string)"
    )

    # Validators for boolean fields with interpolation support
    _validate_create_if_missing = field_validator("create_if_missing", mode="before")(
        interpolatable_boolean_validator()
    )
    _validate_create_parents = field_validator("create_parents", mode="before")(
        interpolatable_boolean_validator()
    )


class MergeJSONStateOutput(BlockOutput):
    """Output for MergeJSONState executor.

    All fields have defaults to support graceful degradation when merging fails.
    A default-constructed instance represents a failed/crashed merge operation.
    """

    path: str = Field(
        default="",
        description="Absolute path to file (empty string if failed)",
    )
    created: bool = Field(
        default=False,
        description="Whether file was created (vs updated), False if failed",
    )
    merged_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Result after merge (empty dict if failed)",
    )


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
        # Resolve interpolatable fields to their actual types
        create_if_missing = resolve_interpolatable_boolean(
            inputs.create_if_missing, "create_if_missing"
        )
        create_parents = resolve_interpolatable_boolean(inputs.create_parents, "create_parents")

        # Resolve path with security validation
        path_result = PathResolver.resolve_and_validate(inputs.path, allow_traversal=True)
        if not path_result.is_success:
            raise ValueError(f"Invalid path: {path_result.error}")

        file_path = path_result.value
        assert file_path is not None

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

        # Create parent directories if needed (before atomic operation)
        if create_parents:
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise OSError(f"Failed to create parent directories: {e}") from e
        elif not file_path.parent.exists():
            raise FileNotFoundError(f"Parent directory missing: {file_path.parent}")

        # Define atomic read-modify-write operation for serialization
        def merge_operation() -> tuple[bool, dict[str, Any]]:
            """Atomically read, merge, and write JSON. Returns (created, merged_data)."""
            # Check if file exists
            file_existed = file_path.exists()
            if not file_existed and not create_if_missing:
                raise FileNotFoundError(
                    f"File does not exist and create_if_missing=False: {file_path}"
                )

            # Read existing data or start with empty dict
            existing_data: dict[str, Any] = {}
            if file_existed:
                read_result = JSONOperations.read_json(file_path, required=False)
                if not read_result.is_success:
                    raise ValueError(f"Failed to read existing JSON: {read_result.error}")
                assert read_result.value is not None
                existing_data = read_result.value

            # Merge updates
            merged_data = deep_merge(existing_data, inputs.updates)

            # Write merged data
            write_result = JSONOperations.write_json(file_path, merged_data)
            if not write_result.is_success:
                raise OSError(write_result.error)

            return (not file_existed, merged_data)

        # Execute via IO queue if available (prevents race conditions)
        if context.execution_context and context.execution_context.io_queue:
            created, merged_data = await context.execution_context.io_queue.submit(merge_operation)
        else:
            # Fallback: direct execution (backward compatibility)
            created, merged_data = merge_operation()

        # Build output
        return MergeJSONStateOutput(
            path=str(file_path),
            created=created,
            merged_data=merged_data,
        )
