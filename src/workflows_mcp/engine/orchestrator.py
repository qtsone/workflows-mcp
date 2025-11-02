"""Block execution orchestrator for ADR-006.

The orchestrator wraps executor.execute() calls to:
1. Catch exceptions → create appropriate Metadata
2. Catch ExecutionPaused → signal pause to caller
3. Determine operation outcome (for Shell: exit_code)
4. Return structured BlockExecution result
5. Resolve secrets in inputs and redact secrets from outputs

This is the bridge between executors (which return BaseModel or raise exceptions)
and the workflow execution layer (which needs Metadata).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .block import BlockInput
from .exceptions import ExecutionPaused, RecursionDepthExceededError
from .execution import Execution
from .executor_base import BlockExecutor
from .metadata import Metadata

if TYPE_CHECKING:
    from .secrets import SecretProvider, SecretRedactor


class BlockExecution(BaseModel):
    """
    Result of executing a single block.

    This wraps the executor output with execution metadata.
    """

    # Executor output (or None if paused/failed)
    output: BaseModel | None

    # Execution metadata (always present)
    metadata: Metadata

    # Pause information (only if paused)
    paused: bool = False
    pause_prompt: str | None = None
    pause_checkpoint_data: dict[str, Any] | None = None


class BlockOrchestrator:
    """
    Orchestrates block execution with exception handling and metadata creation.

    This is the bridge between:
    - Executors (which return BaseModel or raise exceptions)
    - Workflow execution layer (which needs Metadata)

    Responsibilities:
    - Call executor.execute()
    - Catch exceptions → create Metadata
    - Catch ExecutionPaused → signal pause
    - Determine operation outcome (e.g., Shell exit_code)
    - Resolve secrets in inputs before execution
    - Redact secrets from outputs after execution
    - Return BlockExecution with output + metadata
    """

    def __init__(
        self,
        secret_provider: SecretProvider | None = None,
        secret_redactor: SecretRedactor | None = None,
    ):
        """
        Initialize block orchestrator with optional secret management.

        Args:
            secret_provider: Optional secret provider for resolving {{secrets.*}}
            secret_redactor: Optional secret redactor for output sanitization
        """
        self.secret_provider = secret_provider
        self.secret_redactor = secret_redactor

    async def execute_block(
        self,
        executor: BlockExecutor,
        inputs: BlockInput,
        context: Execution,
        wave: int = 0,
        execution_order: int = 0,
    ) -> BlockExecution:
        """
        Execute a block with orchestration (exception handling + metadata).

        Args:
            executor: Block executor to run
            inputs: Validated block inputs
            context: Current execution context
            wave: Wave number (for metadata)
            execution_order: Execution order within wave (for metadata)

        Returns:
            BlockExecution with output and metadata

        Pause Handling:
            If executor raises ExecutionPaused, returns BlockExecution with:
            - paused=True
            - pause_prompt set
            - pause_checkpoint_data set
            - metadata.status=PAUSED
        """
        # Timing
        started_at = datetime.now(UTC).isoformat()
        start_time_ms = datetime.now(UTC).timestamp() * 1000

        try:
            # Execute block
            output = await executor.execute(inputs, context)

            # Redact secrets from output (if redactor is configured)
            if self.secret_redactor:
                # Convert output to dict, redact, then reconstruct
                output_dict = output.model_dump()
                redacted_dict = self.secret_redactor.redact(output_dict)
                # Reconstruct output with redacted values
                output = type(output)(**redacted_dict)

            # Success! Determine operation outcome
            completed_at = datetime.now(UTC).isoformat()
            execution_time_ms = datetime.now(UTC).timestamp() * 1000 - start_time_ms

            # Check if this is a Shell block with exit_code
            # If exit_code != 0, operation failed (but execution succeeded)
            if hasattr(output, "exit_code"):
                exit_code = output.exit_code
                if exit_code == 0:
                    # Shell command succeeded
                    metadata = Metadata.from_success(
                        execution_time_ms=execution_time_ms,
                        started_at=started_at,
                        completed_at=completed_at,
                        wave=wave,
                        execution_order=execution_order,
                    )
                else:
                    # Shell command failed (exit_code != 0)
                    # Execution succeeded, but operation failed
                    metadata = Metadata.from_operation_failure(
                        message=f"Command exited with code {exit_code}",
                        execution_time_ms=execution_time_ms,
                        started_at=started_at,
                        completed_at=completed_at,
                        wave=wave,
                        execution_order=execution_order,
                    )
            else:
                # Non-shell block - operation succeeded
                metadata = Metadata.from_success(
                    execution_time_ms=execution_time_ms,
                    started_at=started_at,
                    completed_at=completed_at,
                    wave=wave,
                    execution_order=execution_order,
                )

            return BlockExecution(
                output=output,
                metadata=metadata,
                paused=False,
            )

        except ExecutionPaused as e:
            # Block paused for external input
            completed_at = datetime.now(UTC).isoformat()
            execution_time_ms = datetime.now(UTC).timestamp() * 1000 - start_time_ms

            metadata = Metadata.from_paused(
                message=e.prompt,
                execution_time_ms=execution_time_ms,
                started_at=started_at,
                paused_at=completed_at,
                wave=wave,
                execution_order=execution_order,
            )

            return BlockExecution(
                output=None,
                metadata=metadata,
                paused=True,
                pause_prompt=e.prompt,
                pause_checkpoint_data=e.checkpoint_data,
            )

        except RecursionDepthExceededError:
            # Recursion depth limit exceeded - this is a critical error that should
            # bubble up to fail the entire workflow, not just mark the block as failed
            raise

        except Exception as e:
            # Execution failed (executor crashed)
            completed_at = datetime.now(UTC).isoformat()
            execution_time_ms = datetime.now(UTC).timestamp() * 1000 - start_time_ms

            # Get error message
            error_msg = f"{type(e).__name__}: {str(e)}"

            metadata = Metadata.from_execution_failure(
                message=error_msg,
                execution_time_ms=execution_time_ms,
                started_at=started_at,
                completed_at=completed_at,
                wave=wave,
                execution_order=execution_order,
            )

            return BlockExecution(
                output=None,
                metadata=metadata,
                paused=False,
            )

    async def resume_block(
        self,
        executor: BlockExecutor,
        inputs: BlockInput,
        context: Execution,
        response: str,
        pause_metadata: dict[str, Any],
        wave: int = 0,
        execution_order: int = 0,
    ) -> BlockExecution:
        """
        Resume a paused block execution.

        Args:
            executor: Block executor to resume
            inputs: Original block inputs
            context: Current execution context
            response: LLM's response to pause prompt
            pause_metadata: Metadata from pause
            wave: Wave number (for metadata)
            execution_order: Execution order within wave (for metadata)

        Returns:
            BlockExecution with output and metadata

        Note:
            Resume can also raise ExecutionPaused if block pauses again
            (e.g., multi-step interactive workflow)
        """
        # Timing
        started_at = datetime.now(UTC).isoformat()
        start_time_ms = datetime.now(UTC).timestamp() * 1000

        try:
            # Resume block
            output = await executor.resume(inputs, context, response, pause_metadata)

            # Success!
            completed_at = datetime.now(UTC).isoformat()
            execution_time_ms = datetime.now(UTC).timestamp() * 1000 - start_time_ms

            metadata = Metadata.from_success(
                execution_time_ms=execution_time_ms,
                started_at=started_at,
                completed_at=completed_at,
                wave=wave,
                execution_order=execution_order,
            )

            return BlockExecution(
                output=output,
                metadata=metadata,
                paused=False,
            )

        except ExecutionPaused as e:
            # Block paused again (multi-step interaction)
            completed_at = datetime.now(UTC).isoformat()
            execution_time_ms = datetime.now(UTC).timestamp() * 1000 - start_time_ms

            metadata = Metadata.from_paused(
                message=e.prompt,
                execution_time_ms=execution_time_ms,
                started_at=started_at,
                paused_at=completed_at,
                wave=wave,
                execution_order=execution_order,
            )

            return BlockExecution(
                output=None,
                metadata=metadata,
                paused=True,
                pause_prompt=e.prompt,
                pause_checkpoint_data=e.checkpoint_data,
            )

        except Exception as e:
            # Resume failed
            completed_at = datetime.now(UTC).isoformat()
            execution_time_ms = datetime.now(UTC).timestamp() * 1000 - start_time_ms

            error_msg = f"{type(e).__name__}: {str(e)}"

            metadata = Metadata.from_execution_failure(
                message=error_msg,
                execution_time_ms=execution_time_ms,
                started_at=started_at,
                completed_at=completed_at,
                wave=wave,
                execution_order=execution_order,
            )

            return BlockExecution(
                output=None,
                metadata=metadata,
                paused=False,
            )
