"""Metadata model for ADR-006 unified execution model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .block_status import ExecutionStatus, OperationOutcome


class Metadata(BaseModel):
    """
    Execution metadata (fractal - same for workflows and blocks).

    Single source of truth combining execution state and operation outcome.
    """

    # Execution state (did executor run?)
    status: ExecutionStatus
    """Execution lifecycle state."""

    # Operation outcome (did operation succeed?)
    outcome: OperationOutcome
    """Operation outcome."""

    # Timing
    execution_time_ms: float
    """Execution time in milliseconds."""

    started_at: str
    """ISO 8601 timestamp when execution started."""

    completed_at: str
    """ISO 8601 timestamp when execution completed."""

    # Position in parent execution
    wave: int = Field(default=0, ge=0)
    """Wave number in parallel execution (0-indexed)."""

    execution_order: int = Field(default=0, ge=0)
    """Execution order within wave (0-indexed)."""

    # Message (for failures, skips)
    message: str | None = None
    """Informational message (error details, skip reason, etc.)."""

    @classmethod
    def from_success(
        cls,
        execution_time_ms: float,
        started_at: str,
        completed_at: str,
        wave: int = 0,
        execution_order: int = 0,
    ) -> Metadata:
        """
        Executor ran, operation succeeded.

        Example: Shell exit 0, file created successfully.
        Result: status=COMPLETED, outcome=SUCCESS
        """
        return cls(
            status=ExecutionStatus.COMPLETED,
            outcome=OperationOutcome.SUCCESS,
            execution_time_ms=execution_time_ms,
            started_at=started_at,
            completed_at=completed_at,
            wave=wave,
            execution_order=execution_order,
            message=None,
        )

    @classmethod
    def from_operation_failure(
        cls,
        message: str,
        execution_time_ms: float,
        started_at: str,
        completed_at: str,
        wave: int = 0,
        execution_order: int = 0,
    ) -> Metadata:
        """
        Executor ran, but operation failed.

        Example: Shell exit 1, command returned error.
        Result: status=COMPLETED, outcome=FAILURE
        """
        return cls(
            status=ExecutionStatus.COMPLETED,
            outcome=OperationOutcome.FAILURE,
            execution_time_ms=execution_time_ms,
            started_at=started_at,
            completed_at=completed_at,
            wave=wave,
            execution_order=execution_order,
            message=message,
        )

    @classmethod
    def from_execution_failure(
        cls,
        message: str,
        execution_time_ms: float,
        started_at: str,
        completed_at: str,
        wave: int = 0,
        execution_order: int = 0,
    ) -> Metadata:
        """
        Executor crashed / couldn't run.

        Example: Variable resolution failed, validation error, exception.
        Result: status=FAILED, outcome=NOT_APPLICABLE
        """
        return cls(
            status=ExecutionStatus.FAILED,
            outcome=OperationOutcome.NOT_APPLICABLE,
            execution_time_ms=execution_time_ms,
            started_at=started_at,
            completed_at=completed_at,
            wave=wave,
            execution_order=execution_order,
            message=message,
        )

    @classmethod
    def from_skipped(
        cls,
        message: str,
        timestamp: str,
        wave: int = 0,
        execution_order: int = 0,
    ) -> Metadata:
        """
        Block skipped (condition false or dependency not met).

        Result: status=SKIPPED, outcome=NOT_APPLICABLE
        """
        return cls(
            status=ExecutionStatus.SKIPPED,
            outcome=OperationOutcome.NOT_APPLICABLE,
            execution_time_ms=0.0,
            started_at=timestamp,
            completed_at=timestamp,
            wave=wave,
            execution_order=execution_order,
            message=message,
        )

    @classmethod
    def from_paused(
        cls,
        message: str,
        execution_time_ms: float,
        started_at: str,
        paused_at: str,
        wave: int = 0,
        execution_order: int = 0,
    ) -> Metadata:
        """
        Block paused (waiting for external input).

        Result: status=PAUSED, outcome=NOT_APPLICABLE
        """
        return cls(
            status=ExecutionStatus.PAUSED,
            outcome=OperationOutcome.NOT_APPLICABLE,
            execution_time_ms=execution_time_ms,
            started_at=started_at,
            completed_at=paused_at,
            wave=wave,
            execution_order=execution_order,
            message=message,
        )

    # ADR-005: Shortcut accessors for block state
    @property
    def succeeded(self) -> bool:
        """
        True if block executed successfully (COMPLETED + SUCCESS).

        Shortcut accessor for: status == COMPLETED and outcome == SUCCESS
        """
        return self.status == ExecutionStatus.COMPLETED and self.outcome == OperationOutcome.SUCCESS

    @property
    def failed(self) -> bool:
        """
        True if block failed (execution crashed OR operation failed).

        Shortcut accessor for: status == FAILED or (status == COMPLETED and outcome == FAILURE)
        """
        return self.status == ExecutionStatus.FAILED or (
            self.status == ExecutionStatus.COMPLETED and self.outcome == OperationOutcome.FAILURE
        )

    @property
    def skipped(self) -> bool:
        """
        True if block was skipped (condition false or dependency not met).

        Shortcut accessor for: status == SKIPPED
        """
        return self.status == ExecutionStatus.SKIPPED

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """
        Override model_dump to include computed properties.

        ADR-005 shortcut accessors (succeeded, failed, skipped) must be
        available in the dict for variable resolution.
        """
        data = super().model_dump(**kwargs)
        # Add computed properties to the dict
        data["succeeded"] = self.succeeded
        data["failed"] = self.failed
        data["skipped"] = self.skipped
        return data

    def requires_dependent_skip(self, required: bool = True) -> bool:
        """
        Check if child block (dependent) should skip based on this parent block's state.

        Args:
            required: True if child requires parent success (default), False if optional

        Returns:
            True if child block should skip execution

        Dependency Logic (from child's perspective):
            required=True (default):
                - Child REQUIRES parent to succeed
                - Skip child unless parent: COMPLETED + SUCCESS
                - Cascade skip for parent states: FAILED, SKIPPED, COMPLETED+FAILURE

            required=False (optional - ordering only):
                - Child runs even if parent fails/skips
                - Skip child only if parent: FAILED (executor crashed)
                - Run child if parent: SKIPPED or COMPLETED (regardless of outcome)
        """
        if required:
            # Required dependency - child requires parent to complete successfully
            return not (
                self.status == ExecutionStatus.COMPLETED
                and self.outcome == OperationOutcome.SUCCESS
            )
        else:
            # Optional dependency - child runs unless parent executor crashed
            # Skip child only if parent FAILED (executor crashed - couldn't run)
            return self.status == ExecutionStatus.FAILED
