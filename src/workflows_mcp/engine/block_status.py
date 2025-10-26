"""Execution status and operation outcome enums for ADR-006 unified execution model."""

from enum import Enum


class ExecutionStatus(str, Enum):
    """
    Execution lifecycle states (fractal - same for workflows and blocks).

    Represents whether the executor ran, not whether the operation succeeded.
    """

    PENDING = "pending"
    """Queued but not started."""

    RUNNING = "running"
    """Currently executing."""

    COMPLETED = "completed"
    """Executor finished running (operation may have succeeded or failed)."""

    FAILED = "failed"
    """Executor crashed / couldn't run (variable error, validation, exception)."""

    SKIPPED = "skipped"
    """Did not execute (condition false or dependency not met)."""

    PAUSED = "paused"
    """Paused waiting for external input."""

    def is_pending(self) -> bool:
        """Check if execution is pending."""
        return self == ExecutionStatus.PENDING

    def is_running(self) -> bool:
        """Check if execution is running."""
        return self == ExecutionStatus.RUNNING

    def is_completed(self) -> bool:
        """Check if execution completed."""
        return self == ExecutionStatus.COMPLETED

    def is_failed(self) -> bool:
        """Check if execution failed."""
        return self == ExecutionStatus.FAILED

    def is_skipped(self) -> bool:
        """Check if execution was skipped."""
        return self == ExecutionStatus.SKIPPED

    def is_paused(self) -> bool:
        """Check if execution is paused."""
        return self == ExecutionStatus.PAUSED


class OperationOutcome(str, Enum):
    """
    Operation outcome (separate from execution state).

    Represents whether the operation succeeded, independent of whether
    the executor ran.
    """

    SUCCESS = "success"
    """Operation succeeded (e.g., Shell exit 0, file created)."""

    FAILURE = "failure"
    """Operation failed (e.g., Shell exit 1, validation error)."""

    NOT_APPLICABLE = "n/a"
    """No operation outcome (FAILED, SKIPPED, PAUSED executions)."""

    def is_success(self) -> bool:
        """Check if operation succeeded."""
        return self == OperationOutcome.SUCCESS

    def is_failure(self) -> bool:
        """Check if operation failed."""
        return self == OperationOutcome.FAILURE

    def is_not_applicable(self) -> bool:
        """Check if outcome is not applicable."""
        return self == OperationOutcome.NOT_APPLICABLE
