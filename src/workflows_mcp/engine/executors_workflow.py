"""
WorkflowExecutor for fractal workflow composition.

This executor enables workflows to execute other workflows as blocks,
creating a fractal structure where workflows can be nested arbitrarily.

Uses ExecutionContext for dependency injection and WorkflowRunner for execution.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from .block import BlockInput
from .exceptions import ExecutionPaused
from .execution import Execution
from .executor_base import BlockExecutor, ExecutorCapabilities, ExecutorSecurityLevel


class WorkflowInput(BlockInput):
    """
    Input model for Workflow executor.

    Supports variable references from parent context:
    - {{inputs.field}}: Parent workflow inputs
    - {{blocks.block_id.outputs.field}}: Parent block outputs
    - {{metadata.field}}: Parent workflow metadata

    Variable resolution happens in parent context before passing to child.
    """

    workflow: str = Field(description="Workflow name to execute")
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Inputs to pass to child workflow (variables resolved in parent context)",
    )
    timeout_ms: int | None = Field(
        default=None,
        description="Optional timeout for child execution in milliseconds",
    )


class WorkflowExecutor(BlockExecutor):
    """
    Workflow composition executor (fractal pattern).

    This executor is SPECIAL - it returns the full child Execution object,
    not a BlockOutput. The orchestrator recognizes this and stores the child
    Execution directly in parent.blocks, enabling true fractal nesting.

    Architecture:
    - Returns full child Execution (includes child's blocks!)
    - Orchestrator stores child Execution in parent.blocks[block_id]
    - Enables deep access: {{blocks.run_tests.blocks.pytest.outputs.exit_code}}
    - Uses ExecutionContext for workflow registry access
    - Uses WorkflowRunner for child workflow execution
    - Recursion depth limiting via context.check_recursion_depth()
    - Supports recursive workflows up to max_recursion_depth (default: 50)
    - Raises ExecutionPaused if child pauses (automatic bubbling)

    Fractal Pattern:
        parent_execution.blocks = {
            "run_tests": Execution(  # ← Full child execution embedded!
                outputs={"test_passed": True},
                blocks={  # ← Child's internal blocks preserved!
                    "pytest": Execution(...),
                    "coverage": Execution(...),
                }
            )
        }

    Variable Access:
        {{blocks.run_tests.outputs.test_passed}}  # Child's workflow output
        {{blocks.run_tests.blocks.pytest.outputs.exit_code}}  # Drill down!

    Pause Propagation:
        If child workflow pauses (Prompt block), ExecutionPaused exception
        automatically bubbles through call stack to top-level orchestrator.

    Usage:
        blocks:
          - id: run_tests
            type: Workflow
            inputs:
              workflow: python-ci-pipeline
              inputs:
                project_path: "{{inputs.project_path}}"
    """

    type_name: ClassVar[str] = "Workflow"
    input_type: ClassVar[type[BlockInput]] = WorkflowInput
    output_type: ClassVar[type] = type(None)  # Special: returns Execution, not BlockOutput

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(
        can_modify_state=True  # Can execute other workflows
    )

    async def execute(  # type: ignore[override]
        self, inputs: WorkflowInput, context: Execution
    ) -> Execution:
        """
        Execute child workflow with full embedding.

        Args:
            inputs: Validated WorkflowInput
            context: Parent execution context (fractal structure)

        Returns:
            Full child Execution (stored directly in parent.blocks[block_id])

        Raises:
            RecursionDepthExceededError: Recursion depth limit exceeded
            ValueError: Workflow not found or child execution failed
            ExecutionPaused: Child workflow paused (bubbles automatically)
            Exception: Any other child execution failure
        """
        # 1. Get ExecutionContext from parent context (typed accessor)
        exec_context = context.execution_context
        if exec_context is None:
            raise RuntimeError(
                "ExecutionContext not found - workflow composition not supported in this context"
            )

        workflow_name = inputs.workflow

        # 2. Recursion depth check (allows recursion up to max_recursion_depth)
        exec_context.check_recursion_depth(workflow_name)

        # 3. Get workflow from registry
        workflow = exec_context.get_workflow(workflow_name)
        if workflow is None:
            available = exec_context.workflow_registry.list_names()
            raise ValueError(
                f"Workflow '{workflow_name}' not found in registry. "
                f"Available: {', '.join(available[:10])}"
            )

        # 4. Create WorkflowRunner and execute child workflow
        from .workflow_runner import WorkflowRunner

        # No checkpointing for nested workflows - parent handles all checkpointing
        runner = WorkflowRunner()

        # Create child execution context
        child_context = exec_context.create_child_context(
            parent_execution=context,
            workflow_name=workflow_name,
        )

        try:
            # Execute child workflow
            child_execution_result = await runner.execute(
                workflow=workflow,
                runtime_inputs=inputs.inputs,
                context=child_context,
            )

            # Check if child execution succeeded
            if child_execution_result.status == "failure":
                # Check if the failure was due to recursion depth exceeded
                # If so, re-raise the original exception to preserve its type
                from .exceptions import RecursionDepthExceededError

                # Check if error is from recursion depth limit (match error message, not class name)
                if "Recursion depth limit exceeded" in (child_execution_result.error or ""):
                    # Re-raise as RecursionDepthExceededError to preserve exception type
                    # This ensures proper handling in parent workflows
                    raise RecursionDepthExceededError(
                        workflow_name=workflow_name,
                        current_depth=len(child_context.workflow_stack),
                        max_depth=child_context.max_recursion_depth,
                        workflow_stack=child_context.workflow_stack,
                    )
                else:
                    raise ValueError(
                        f"Child workflow '{workflow_name}' failed: {child_execution_result.error}"
                    )
            elif child_execution_result.status == "paused":
                # Child paused - re-raise with context, preserving execution state
                # (Unified Job Architecture - no checkpoint_id, use execution_state)
                child_execution_state = child_execution_result.execution_state
                if not child_execution_state:
                    raise RuntimeError(
                        "Paused child workflow missing execution_state - invalid pause"
                    )

                child_pause_data = child_execution_result.pause_data
                child_metadata = child_pause_data.metadata if child_pause_data else {}

                checkpoint_data = {
                    "child_execution_state": child_execution_state,
                    "child_workflow": child_execution_state.workflow_name,
                }
                # Preserve entire child metadata structure for nested pause chains
                if child_metadata:
                    checkpoint_data["child_pause_metadata"] = child_metadata

                raise ExecutionPaused(
                    prompt=child_pause_data.prompt if child_pause_data else "Child workflow paused",
                    checkpoint_data=checkpoint_data,
                    execution=context,  # Parent execution context
                )

            # Success - return child execution
            return child_execution_result.execution

        except ExecutionPaused as child_pause:
            # Child workflow paused - wrap with parent metadata (fractal pattern)
            # Preserve nested pause metadata for multi-level nesting
            # (Unified Job Architecture - extract execution_state, not checkpoint_id)
            pause_data = child_pause.checkpoint_data

            # Get child's execution state (required for resume)
            child_execution_state = pause_data.get("child_execution_state") or pause_data.get(
                "execution_state"
            )
            if not child_execution_state:
                raise RuntimeError(
                    "ExecutionPaused from child workflow missing execution_state - invalid pause"
                )

            # Get child's workflow name from execution state
            child_workflow = (
                child_execution_state.workflow_name
                if hasattr(child_execution_state, "workflow_name")
                else workflow_name
            )

            raise ExecutionPaused(
                prompt=child_pause.prompt,
                checkpoint_data={
                    "child_execution_state": child_execution_state,
                    "child_workflow": child_workflow,  # Immediate child's name
                    "child_pause_metadata": child_pause.checkpoint_data,  # Nested metadata
                    "parent_workflow": context.parent_workflow or "",
                },
                execution=context,  # Parent execution context
            )

    async def resume(
        self,
        inputs: BlockInput,
        context: Execution,
        response: str,
        pause_metadata: dict[str, Any],
    ) -> Execution:
        """
        Resume child workflow execution after pause.

        When a parent workflow resumes and this block was paused (nested workflow),
        we extract the child's execution_state and delegate resume to the child.

        Args:
            inputs: Original WorkflowInput
            context: Parent execution context
            response: LLM's response to pause prompt
            pause_metadata: Pause metadata containing child_execution_state

        Returns:
            Full child Execution (resumed state)

        Raises:
            ValueError: Missing execution state or workflow not found
            ExecutionPaused: Child paused again (bubbles automatically)
            Exception: Any other child execution failure
        """
        # Type assertion
        assert isinstance(inputs, WorkflowInput)

        # 1. Extract child execution state from pause metadata (unified Job architecture)
        child_execution_state = pause_metadata.get("child_execution_state")
        if not child_execution_state:
            raise ValueError(
                "Missing child_execution_state in pause_metadata - cannot resume nested workflow"
            )

        # 2. Get ExecutionContext from parent context (typed accessor)
        exec_context = context.execution_context
        if exec_context is None:
            raise RuntimeError(
                "ExecutionContext not found - workflow composition not supported in this context"
            )

        # 3. Create WorkflowRunner and resume child workflow
        from .workflow_runner import WorkflowRunner

        # No checkpointing for nested workflows - parent handles all checkpointing
        runner = WorkflowRunner()

        # Resume child workflow using unified Job architecture (ExecutionState)
        from .execution_result import ExecutionState

        if not isinstance(child_execution_state, ExecutionState):
            raise ValueError(
                f"Invalid child execution state type: {type(child_execution_state)}. "
                "Expected ExecutionState."
            )

        try:
            # Resume child workflow from ExecutionState
            child_execution_result = await runner.resume_from_state(
                execution_state=child_execution_state,
                response=response,
                context=exec_context,
            )
        except Exception as e:
            raise ValueError(f"Failed to resume child workflow: {e}") from e

        # Check result status
        if child_execution_result.status == "failure":
            raise ValueError(f"Child workflow resume failed: {child_execution_result.error}")
        elif child_execution_result.status == "paused":
            # Child paused again - extract execution state and re-raise
            child_execution_state_new = child_execution_result.execution_state
            if not child_execution_state_new:
                raise RuntimeError("Paused child workflow missing execution_state - invalid pause")

            raise ExecutionPaused(
                prompt=child_execution_result.pause_data.prompt
                if child_execution_result.pause_data
                else "Child workflow paused",
                checkpoint_data={
                    "child_execution_state": child_execution_state_new,
                    "child_workflow": child_execution_state_new.workflow_name,
                },
                execution=context,
            )

        # Success - return child execution
        return child_execution_result.execution


__all__ = ["WorkflowExecutor", "WorkflowInput"]
