"""
Execution result monad for workflow operations.

Aligned with LoadResult pattern - provides type-safe result handling
with full execution context preservation for debugging.

Design Principles:
- Execution field ALWAYS present (never None)
- Status determines interpretation (success/failure/paused)
- Factory methods ensure valid state combinations
- to_response() is single source of truth for formatting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .execution import Execution


@dataclass
class PauseData:
    """Pause-specific metadata for paused workflows."""

    checkpoint_id: str
    prompt: str
    metadata: dict[str, Any] | None = None

    @property
    def resume_message(self) -> str:
        """Generate user-friendly resume instruction."""
        return (
            f'Use resume_workflow(checkpoint_id: "{self.checkpoint_id}", '
            f'response: "<your-response>") to continue'
        )


@dataclass
class ExecutionResult:
    """
    Monad for workflow execution results (success/failure/paused).

    Inspired by LoadResult pattern - ensures execution context is ALWAYS
    preserved regardless of outcome. This enables:

    - Rich error debugging (partial execution state on failures)
    - Consistent response formatting (single code path for all statuses)
    - Type-safe result handling (factory methods prevent invalid states)

    Example Usage:
        # Success
        result = ExecutionResult.success(execution)

        # Failure with partial execution
        result = ExecutionResult.failure("Build failed", partial_execution)

        # Paused for user input
        result = ExecutionResult.paused(checkpoint_id, prompt, execution)

        # Format for MCP tool
        return result.to_response(response_format="detailed")
    """

    status: Literal["success", "failure", "paused"]
    execution: Execution  # ALWAYS present - complete execution context
    error: str | None = None
    pause_data: PauseData | None = None

    # Factory Methods (Type-Safe Construction)

    @staticmethod
    def success(execution: Execution) -> ExecutionResult:
        """
        Create success result with complete execution.

        Args:
            execution: Complete execution context with outputs

        Returns:
            ExecutionResult with status="success"
        """
        return ExecutionResult(
            status="success",
            execution=execution,
            error=None,
            pause_data=None,
        )

    @staticmethod
    def failure(error: str, partial_execution: Execution) -> ExecutionResult:
        """
        Create failure result with partial execution for debugging.

        Unlike traditional error handling that loses context, this preserves:
        - All completed blocks up to point of failure
        - Execution metadata (start time, wave count, etc.)
        - Partial outputs from successful blocks

        Args:
            error: Error message describing what failed
            partial_execution: Execution context up to failure point

        Returns:
            ExecutionResult with status="failure" and full debug context
        """
        return ExecutionResult(
            status="failure",
            execution=partial_execution,
            error=error,
            pause_data=None,
        )

    @staticmethod
    def paused(
        checkpoint_id: str,
        prompt: str,
        execution: Execution,
        pause_metadata: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """
        Create paused result with execution state for resume.

        Args:
            checkpoint_id: Checkpoint ID for resuming
            prompt: Prompt to show user
            execution: Current execution state
            pause_metadata: Optional pause-specific metadata

        Returns:
            ExecutionResult with status="paused"
        """
        pause_data = PauseData(
            checkpoint_id=checkpoint_id,
            prompt=prompt,
            metadata=pause_metadata,
        )
        return ExecutionResult(
            status="paused",
            execution=execution,
            error=None,
            pause_data=pause_data,
        )

    # Formatting Methods

    def to_response(self, response_format: Literal["minimal", "detailed"]) -> dict[str, Any]:
        """
        Format execution result for MCP tool response.

        Single source of truth for ALL response formatting across all statuses.
        Handles success, failure, and paused states consistently.

        Formatting Rules:
        - minimal: status + essential fields (outputs/error/checkpoint_id)
        - detailed: minimal + blocks + metadata (for debugging)

        Args:
            response_format: Verbosity level

        Returns:
            Dict ready for MCP tool return

        Examples:
            # Success - minimal
            {"status": "success", "outputs": {...}}

            # Success - detailed
            {"status": "success", "outputs": {...}, "blocks": {...}, "metadata": {...}}

            # Failure - minimal
            {"status": "failure", "error": "..."}

            # Failure - detailed (KEY FIX: includes debug context!)
            {"status": "failure", "error": "...", "blocks": {...}, "metadata": {...}}

            # Paused - minimal
            {"status": "paused", "checkpoint_id": "...", "prompt": "..."}
        """
        # Convert execution components to dict format
        blocks_dict = self._blocks_to_dict()
        metadata_dict = self._metadata_to_dict()

        # Base response (status always included)
        response: dict[str, Any] = {
            "status": self.status,
        }

        # Add status-specific essential fields
        if self.status == "success":
            response["outputs"] = self.execution.outputs

        elif self.status == "failure":
            response["error"] = self.error
            # Note: outputs is None for failures (not applicable)

        elif self.status == "paused":
            assert self.pause_data is not None
            response["checkpoint_id"] = self.pause_data.checkpoint_id
            response["prompt"] = self.pause_data.prompt
            response["message"] = self.pause_data.resume_message

        # Add detailed context if requested (works for ALL statuses!)
        if response_format == "detailed":
            response["blocks"] = blocks_dict
            response["metadata"] = metadata_dict

        return response

    def _blocks_to_dict(self) -> dict[str, Any]:
        """Convert execution.blocks to dict format for JSON serialization."""
        blocks_dict = {}
        for block_id, block_exec in self.execution.blocks.items():
            if isinstance(block_exec, Execution):
                # Nested execution (from WorkflowExecutor)
                blocks_dict[block_id] = block_exec.model_dump()
            else:
                blocks_dict[block_id] = block_exec
        return blocks_dict

    def _metadata_to_dict(self) -> dict[str, Any]:
        """Convert execution.metadata to dict format for JSON serialization."""
        if isinstance(self.execution.metadata, dict):
            return self.execution.metadata
        else:
            return self.execution.metadata.model_dump()


__all__ = ["ExecutionResult", "PauseData"]
