"""LoadResult for safe workflow loading and validation operations.

This is a specialized error monad used exclusively by the loader/registry
layer for file I/O operations. It is NOT used for workflow execution.

For execution results, see:
- Metadata (ADR-009: universal node metadata for fractal for_each)
- ExecutionResult (workflow execution monad)
- BlockExecution (block execution result with Metadata)

Post ADR-006: Executors return BaseModel directly and raise exceptions.
Post ADR-009: Metadata provides fractal metadata structure.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class LoadStatus(str, Enum):
    """Status of a loading/validation operation.

    Using a discriminated union pattern ensures type safety by preventing
    invalid state combinations.

    Used exclusively by loader/registry layer for:
    - YAML file loading
    - Schema validation
    - Workflow discovery
    - Dependency resolution
    """

    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class LoadResult(Generic[T]):  # noqa: UP046
    """
    LoadResult for safe loading/validation operations (loader/registry layer).

    Uses a discriminated union pattern with LoadStatus enum to ensure
    type-safe state management and prevent invalid state combinations.

    Post ADR-006: Executors return BaseModel directly and raise exceptions.
    This LoadResult class is used only by loader/registry for file operations.

    Usage:
        # Safe file loading
        load_result = load_workflow_from_file(path)
        if load_result.is_success:
            schema = load_result.value
        else:
            print(f"Load error: {load_result.error}")
    """

    status: LoadStatus
    value: T | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate state consistency after initialization.

        Ensures that the result data matches the declared state:
        - SUCCESS results must have a value
        - FAILED results must have an error message
        """
        if self.status == LoadStatus.SUCCESS and self.value is None:
            raise ValueError("Success result must have a value")
        if self.status == LoadStatus.FAILED and not self.error:
            raise ValueError("Failed result must have an error message")

    @property
    def is_success(self) -> bool:
        """Check if load operation was successful."""
        return self.status == LoadStatus.SUCCESS

    @property
    def is_failure(self) -> bool:
        """Check if load operation failed.

        Note: This checks for FAILED status (operation could not complete).
        """
        return self.status == LoadStatus.FAILED

    @classmethod
    def success(cls, value: T, metadata: dict[str, Any] | None = None) -> "LoadResult[T]":
        """Create a successful load result.

        Args:
            value: The loaded/validated value
            metadata: Optional metadata dictionary (defaults to empty dict)

        Returns:
            LoadResult with SUCCESS status and the provided value
        """
        return cls(
            status=LoadStatus.SUCCESS,
            value=value,
            metadata=metadata or {},
        )

    @classmethod
    def failure(cls, error: str, metadata: dict[str, Any] | None = None) -> "LoadResult[T]":
        """Create a failed load result.

        Use this when a load operation failed (file not found, parsing error, etc.).

        Args:
            error: Error message describing the failure
            metadata: Optional metadata dictionary (defaults to empty dict)

        Returns:
            LoadResult with FAILED status and the error message
        """
        return cls(
            status=LoadStatus.FAILED,
            error=error,
            metadata=metadata or {},
        )

    def __bool__(self) -> bool:
        """Allow using result in if statements."""
        return self.is_success

    def unwrap(self) -> T:
        """Get value or raise exception if failed."""
        if not self.is_success:
            raise ValueError(f"Cannot unwrap failed result: {self.error}")
        if self.value is None:
            raise ValueError("Cannot unwrap result: value is None")
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Get value or return default if failed."""
        if self.is_success and self.value is not None:
            return self.value
        return default
