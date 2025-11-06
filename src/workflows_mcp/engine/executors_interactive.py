"""Interactive workflow executor for ADR-006 - Simplified to single Prompt type.

Architecture (ADR-006):
- Execute raises ExecutionPaused (not Result.pause())
- Resume returns PromptOutput directly
- Uses Execution context (not dict)
- Exception bubbles automatically through nested workflows

Philosophy: YAGNI (You Aren't Gonna Need It)
- Single executor type instead of three specialized types
- No built-in validation or choice parsing
- Workflows handle response interpretation using conditions and Shell blocks
- Maximum simplicity and flexibility
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from .block import BlockInput, BlockOutput
from .exceptions import ExecutionPaused
from .execution import Execution
from .executor_base import (
    BlockExecutor,
    ExecutorCapabilities,
    ExecutorSecurityLevel,
)

# ============================================================================
# Prompt Executor - Single Interactive Type
# ============================================================================


class PromptInput(BlockInput):
    """Input for Prompt executor.

    Simple design: single prompt field for maximum flexibility.
    """

    prompt: str = Field(
        description="Prompt/question to display to LLM. The LLM will provide a response."
    )


class PromptOutput(BlockOutput):
    """Output for Prompt executor.

    All fields have defaults to support graceful degradation when prompting fails.
    A default-constructed instance represents a failed/crashed prompt operation.

    Simple design: single response field containing raw LLM response.
    """

    response: str = Field(
        default="",
        description="Raw LLM response to the prompt (empty string if failed or crashed)",
    )


class PromptExecutor(BlockExecutor):
    """Interactive prompt executor - pauses workflow for LLM input.

    This is the ONLY interactive executor type. All interaction patterns
    (yes/no confirmation, multiple choice, free-form input) are handled
    through prompt wording and conditional logic in workflows.

    Architecture (ADR-006):
    - Execute raises ExecutionPaused exception (not Result.pause())
    - Exception automatically bubbles through call stack to orchestrator
    - Orchestrator catches and creates checkpoint
    - Resume returns PromptOutput directly

    Design Philosophy:
    - KISS (Keep It Simple, Stupid): Single input, single output
    - YAGNI (You Aren't Gonna Need It): No validation, no parsing, no special cases
    - DRY (Don't Repeat Yourself): One executor handles all interactive patterns

    Example YAML - Yes/No Confirmation:
        - id: confirm_deploy
          type: Prompt
          inputs:
            prompt: |
              Deploy to production?

              Respond with 'yes' or 'no'

        # Parse response with condition
        - id: deploy
          type: Shell
          inputs:
            command: "./deploy.sh"
          condition: "{{blocks.confirm_deploy.outputs.response}} == 'yes'"
          depends_on: [confirm_deploy]

    Example YAML - Multiple Choice:
        - id: select_env
          type: Prompt
          inputs:
            prompt: |
              Select deployment environment:

              1. development
              2. staging
              3. production

              Respond with the number of your choice.

        # Parse response with conditions
        - id: deploy_dev
          type: Shell
          inputs:
            command: "./deploy.sh dev"
          condition: "{{blocks.select_env.outputs.response}} == '1'"
          depends_on: [select_env]

    Example YAML - Free-form Input:
        - id: get_commit_msg
          type: Prompt
          inputs:
            prompt: |
              Generate a semantic commit message following Conventional Commits.

              Format: type(scope): description

              Respond with ONLY the commit message.

        # Use response directly
        - id: create_commit
          type: Shell
          inputs:
            command: git commit -m "{{blocks.get_commit_msg.outputs.response}}"
          depends_on: [get_commit_msg]

    Benefits of Simplified Design:
    - No complex validation logic to maintain
    - No choice parsing edge cases
    - Workflows have full control over response interpretation
    - Easy to understand and extend
    - Follows YAGNI principle
    """

    type_name: ClassVar[str] = "Prompt"
    input_type: ClassVar[type[BlockInput]] = PromptInput
    output_type: ClassVar[type[BlockOutput]] = PromptOutput

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.SAFE
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities()

    async def execute(  # type: ignore[override]
        self, inputs: PromptInput, context: Execution
    ) -> PromptOutput:
        """Execute prompt - always pauses for LLM input.

        Args:
            inputs: Validated PromptInput
            context: Execution context (unused by this executor)

        Raises:
            ExecutionPaused: Always raised to pause workflow execution

        Note:
            This function never returns normally - it always raises
            ExecutionPaused to signal that workflow should pause and
            wait for LLM response.
        """
        # Raise pause exception (bubbles to orchestrator)
        # Orchestrator will catch, create checkpoint, and return to MCP
        raise ExecutionPaused(
            prompt=inputs.prompt,
            checkpoint_data={
                "block_inputs": inputs.model_dump(),
                "type": "prompt",
            },
            execution=context,
        )

    async def resume(
        self,
        inputs: BlockInput,
        context: Execution,
        response: str,
        pause_metadata: dict[str, Any],
    ) -> PromptOutput:
        """Resume with LLM response.

        Args:
            inputs: Original PromptInput
            context: Execution context
            response: LLM's response to the prompt
            pause_metadata: Metadata from the pause (unused)

        Returns:
            PromptOutput containing raw response

        Note:
            No validation or parsing - just return the raw response.
            Workflows interpret the response using conditions.
        """
        assert isinstance(inputs, PromptInput)

        # Simply return the raw response - no validation, no parsing
        # Workflows handle response interpretation
        return PromptOutput(response=response)


# ============================================================================
# Registration
# ============================================================================

# Executor is registered via create_default_registry() in executor_base.py
# This enables dependency injection and test isolation
