"""Workflow execution exceptions for ADR-006 unified execution model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .execution import Execution


class RecursionDepthExceededError(Exception):
    """
    Workflow recursion depth limit exceeded.

    Raised when a workflow attempts to call itself (or create a recursive call chain)
    beyond the configured maximum recursion depth. This prevents infinite recursion
    and stack overflow scenarios.

    The maximum recursion depth is controlled by the WORKFLOWS_MAX_RECURSION_DEPTH
    environment variable (default: 50).

    Attributes:
        workflow_name: Name of the workflow that exceeded the limit
        current_depth: Current recursion depth when limit was exceeded
        max_depth: Configured maximum recursion depth
        workflow_stack: Full workflow call stack showing the recursion chain
    """

    def __init__(
        self,
        workflow_name: str,
        current_depth: int,
        max_depth: int,
        workflow_stack: list[str],
    ):
        """
        Initialize recursion depth exception.

        Args:
            workflow_name: Name of workflow exceeding limit
            current_depth: Current recursion depth
            max_depth: Maximum allowed depth
            workflow_stack: Workflow call stack
        """
        self.workflow_name = workflow_name
        self.current_depth = current_depth
        self.max_depth = max_depth
        self.workflow_stack = workflow_stack

        call_chain = " → ".join(workflow_stack + [workflow_name])
        super().__init__(
            f"Recursion depth limit exceeded for workflow '{workflow_name}' "
            f"(depth: {current_depth}, limit: {max_depth}). "
            f"Call chain: {call_chain}\n\n"
            f"This may indicate infinite recursion. To increase the limit, set the "
            f"WORKFLOWS_MAX_RECURSION_DEPTH environment variable to a higher value."
        )

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"RecursionDepthExceededError(workflow={self.workflow_name!r}, "
            f"depth={self.current_depth}, limit={self.max_depth})"
        )


class ExecutionPaused(Exception):  # noqa: N818
    # Not an error - control flow mechanism (like StopIteration)
    """
    Workflow execution paused for external input.

    This is NOT an error - it's a control flow mechanism for interactive workflows.
    Similar to how StopIteration controls iterator flow in Python.

    When raised by an executor (e.g., Prompt block), it signals that:
    1. Execution cannot continue without external input
    2. Workflow should checkpoint and return to caller
    3. Resume will be called later with LLM response

    The orchestrator catches this exception and:
    - Creates Metadata.create_leaf_paused()
    - Stores checkpoint with pause data
    - Returns BlockExecution with paused=True

    MCP Flow:
        1. Prompt block raises ExecutionPaused
        2. Exception bubbles through call stack (nested workflows included)
        3. Orchestrator catches → creates checkpoint → returns to MCP client
        4. LLM receives pause prompt
        5. LLM calls resume_workflow tool with checkpoint_id + response
        6. Workflow resumes from checkpoint

    Attributes:
        prompt: Message/question to present to LLM
        checkpoint_data: Data needed to resume execution (block inputs, state, etc.)
        execution: Full execution context at pause point (for ExecutionResult)
    """

    def __init__(self, prompt: str, checkpoint_data: dict[str, Any], execution: Execution):
        """
        Initialize pause exception.

        Args:
            prompt: Message/question for LLM (e.g., "Approve deployment to production?")
            checkpoint_data: Serializable data for resume (block inputs, context state)
            execution: Full execution context at pause point
        """
        self.prompt = prompt
        self.checkpoint_data = checkpoint_data
        self.execution = execution
        super().__init__(f"Execution paused: {prompt}")

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"ExecutionPaused(prompt={self.prompt!r})"
