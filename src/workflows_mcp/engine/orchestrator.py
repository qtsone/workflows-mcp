"""Block execution orchestrator for ADR-006 and ADR-009.

The orchestrator wraps executor.execute() calls to:
1. Catch exceptions → create appropriate Metadata
2. Catch ExecutionPaused → signal pause to caller
3. Determine operation outcome (for Shell: exit_code)
4. Return structured BlockExecution result
5. Resolve secrets in inputs and redact secrets from outputs
6. Execute for_each iterations (parallel/sequential)

This is the bridge between executors (which return BaseModel or raise exceptions)
and the workflow execution layer (which needs Metadata).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from .block import BlockInput
from .exceptions import ExecutionPaused, RecursionDepthExceededError
from .execution import Execution
from .executor_base import BlockExecutor
from .metadata import Metadata
from .validation import validate_iteration_keys

if TYPE_CHECKING:
    from .secrets import SecretProvider, SecretRedactor


class BlockExecution(BaseModel):
    """
    Result of executing a single block.

    This wraps the executor output with execution metadata.
    Mirrors Execution structure for fractal consistency.
    """

    # Resolved inputs used for execution (mirrors Execution.inputs)
    inputs: dict[str, Any] = {}

    # Executor output (or None if paused/failed)
    output: BaseModel | None

    # Execution metadata (always present) - ADR-009 fractal design
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
    - Execute for_each iterations (parallel/sequential)
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
        id: str,
        executor: BlockExecutor,
        inputs: BlockInput,
        context: Execution,
        wave: int = 0,
        execution_order: int = 0,
        depth: int = 0,
    ) -> BlockExecution:
        """
        Execute a block with orchestration (exception handling + metadata).

        Args:
            id: Block identifier or iteration key (for Metadata)
            executor: Block executor to run
            inputs: Validated block inputs
            context: Current execution context
            wave: Wave number (for metadata)
            execution_order: Execution order within wave (for metadata)
            depth: Nesting depth (0 for root blocks, 1+ for iterations)

        Returns:
            BlockExecution with output and metadata

        Pause Handling:
            If executor raises ExecutionPaused, returns BlockExecution with:
            - paused=True
            - pause_prompt set
            - pause_checkpoint_data set
            - metadata indicates paused state
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

            # Success! Create Metadata from output
            execution_time_ms = int(datetime.now(UTC).timestamp() * 1000 - start_time_ms)

            # Extract executor-specific metadata from output.meta
            executor_fields = getattr(output, "meta", {}).copy() if hasattr(output, "meta") else {}

            # Check for exit_code field on output (Shell executor stores it on output directly)
            exit_code = getattr(output, "exit_code", None) if hasattr(output, "exit_code") else None
            if exit_code is not None:
                executor_fields["exit_code"] = exit_code

            # Determine success/failure based on executor-specific logic
            # Shell blocks: exit_code != 0 means operation failure
            # Other blocks: execution success = operation success (default)
            if "exit_code" in executor_fields and executor_fields["exit_code"] != 0:
                # Shell command failed (non-zero exit code)
                exit_code = executor_fields["exit_code"]
                executor_fields["message"] = f"Command exited with code {exit_code}"
                metadata = Metadata.create_leaf_failure(
                    type=executor.type_name,
                    id=id,
                    duration_ms=execution_time_ms,
                    started_at=started_at,
                    wave=wave,
                    execution_order=execution_order,
                    index=execution_order,
                    depth=depth,
                    **executor_fields,
                )
            else:
                # Operation succeeded
                metadata = Metadata.create_leaf_success(
                    type=executor.type_name,
                    id=id,
                    duration_ms=execution_time_ms,
                    started_at=started_at,
                    wave=wave,
                    execution_order=execution_order,
                    index=execution_order,
                    depth=depth,
                    **executor_fields,
                )

            return BlockExecution(
                inputs=inputs.model_dump(),
                output=output,
                metadata=metadata,
                paused=False,
            )

        except ExecutionPaused as e:
            # Block paused for external input
            execution_time_ms = int(datetime.now(UTC).timestamp() * 1000 - start_time_ms)

            metadata = Metadata.create_leaf_paused(
                type=executor.type_name,
                id=id,
                duration_ms=execution_time_ms,
                started_at=started_at,
                wave=wave,
                execution_order=execution_order,
                index=execution_order,
                depth=depth,
                message=e.prompt,
            )

            return BlockExecution(
                inputs=inputs.model_dump(),
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
            execution_time_ms = int(datetime.now(UTC).timestamp() * 1000 - start_time_ms)

            # Get error message
            error_msg = f"{type(e).__name__}: {str(e)}"

            metadata = Metadata.create_leaf_failure(
                type=executor.type_name,
                id=id,
                duration_ms=execution_time_ms,
                started_at=started_at,
                wave=wave,
                execution_order=execution_order,
                index=execution_order,
                depth=depth,
                outcome="crash",  # Distinguish crash from operation failure
                message=error_msg,
            )

            # Create default output instance to prevent VariableNotFoundError
            # When a block crashes, downstream blocks/outputs may reference its outputs
            # All executor output models define defaults, so instantiation with no args
            # produces a semantically meaningful "failed state" (e.g., status_code=0)
            output = executor.output_type()

            return BlockExecution(
                inputs=inputs.model_dump(),
                output=output,
                metadata=metadata,
                paused=False,
            )

    async def resume_block(
        self,
        id: str,
        executor: BlockExecutor,
        inputs: BlockInput,
        context: Execution,
        response: str,
        pause_metadata: dict[str, Any],
        wave: int = 0,
        execution_order: int = 0,
        depth: int = 0,
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
            execution_time_ms = int(datetime.now(UTC).timestamp() * 1000 - start_time_ms)

            # Extract executor-specific metadata from output.meta
            executor_fields = getattr(output, "meta", {}).copy() if hasattr(output, "meta") else {}

            metadata = Metadata.create_leaf_success(
                type=executor.type_name,
                id=id,
                duration_ms=execution_time_ms,
                started_at=started_at,
                wave=wave,
                execution_order=execution_order,
                index=execution_order,
                depth=depth,
                **executor_fields,
            )

            return BlockExecution(
                inputs=inputs.model_dump(),
                output=output,
                metadata=metadata,
                paused=False,
            )

        except ExecutionPaused as e:
            # Block paused again (multi-step interaction)
            execution_time_ms = int(datetime.now(UTC).timestamp() * 1000 - start_time_ms)

            metadata = Metadata.create_leaf_paused(
                type=executor.type_name,
                id=id,
                duration_ms=execution_time_ms,
                started_at=started_at,
                wave=wave,
                execution_order=execution_order,
                index=execution_order,
                depth=depth,
                message=e.prompt,
            )

            return BlockExecution(
                inputs=inputs.model_dump(),
                output=None,
                metadata=metadata,
                paused=True,
                pause_prompt=e.prompt,
                pause_checkpoint_data=e.checkpoint_data,
            )

        except Exception as e:
            # Resume failed
            execution_time_ms = int(datetime.now(UTC).timestamp() * 1000 - start_time_ms)

            error_msg = f"{type(e).__name__}: {str(e)}"

            metadata = Metadata.create_leaf_failure(
                type=executor.type_name,
                id=id,
                duration_ms=execution_time_ms,
                started_at=started_at,
                wave=wave,
                execution_order=execution_order,
                index=execution_order,
                depth=depth,
                message=error_msg,
            )

            return BlockExecution(
                inputs=inputs.model_dump(),
                output=None,
                metadata=metadata,
                paused=False,
            )

    async def resume_for_each(
        self,
        pause_metadata: dict[str, Any],
        response: str,
        context: Execution,
        wave: int = 0,
        depth: int = 0,
    ) -> tuple[dict[str, BlockExecution], Metadata]:
        """
        Resume a paused for_each block from iteration checkpoint (ADR-010).

        Args:
            pause_metadata: for_each checkpoint from ExecutionState.pause_metadata
            response: User's response to the paused iteration's prompt
            context: Current execution context
            wave: Wave number
            depth: Nesting depth

        Returns:
            Tuple of (iteration_results, parent_metadata)

        Raises:
            ExecutionPaused: If another iteration pauses during resume
        """
        from .resolver import UnifiedVariableResolver

        # Extract for_each state from pause_metadata
        block_id = pause_metadata["for_each_block_id"]
        current_key = pause_metadata["current_iteration_key"]
        current_idx = pause_metadata["current_iteration_index"]
        completed_keys = pause_metadata["completed_iterations"]
        remaining_keys = pause_metadata["remaining_iteration_keys"]
        all_iterations = pause_metadata["all_iterations"]
        executor_type = pause_metadata["executor_type"]
        inputs_template = pause_metadata["inputs_template"]
        mode = pause_metadata["mode"]
        max_parallel = pause_metadata["max_parallel"]
        continue_on_error = pause_metadata["continue_on_error"]
        iteration_count = pause_metadata["iteration_count"]
        paused_iter_checkpoint = pause_metadata["paused_iteration_checkpoint"]

        # Get executor
        if not context.execution_context:
            raise RuntimeError("Execution context not available for resume_for_each")
        executor = context.execution_context.executor_registry.get(executor_type)
        if not executor:
            raise ValueError(f"Executor not found: {executor_type}")

        # Reconstruct completed iterations from checkpoint (ADR-010)
        iteration_results: dict[str, BlockExecution] = {}

        # First try to deserialize from checkpoint metadata (new approach)
        completed_iteration_results = pause_metadata.get("completed_iteration_results", {})

        for key in completed_keys:
            if key in completed_iteration_results:
                # Deserialize from checkpoint
                serialized = completed_iteration_results[key]
                metadata = Metadata(**serialized["metadata"])

                # Reconstruct output from serialized data
                # Use Any type because output can be Execution (Workflow) or BlockOutput
                output: Any = None
                if serialized["output"] is not None:
                    # Special case: WorkflowExecutor returns Execution, not BlockOutput
                    # Check by executor type name (Workflow blocks have output_type=type(None))
                    if executor_type == "Workflow":
                        # Workflow block - output is serialized Execution object
                        output = Execution.model_validate(serialized["output"])
                    else:
                        output = executor.output_type(**serialized["output"])

                iteration_results[key] = BlockExecution(
                    inputs=serialized["inputs"],
                    output=output,
                    metadata=metadata,
                    paused=serialized["paused"],
                )
            else:
                # Fallback: try to retrieve from execution context (backward compatibility)
                parent_block = context.blocks.get(block_id)
                if parent_block:
                    iter_exec = (
                        parent_block.blocks.get(key) if hasattr(parent_block, "blocks") else None
                    )
                    if iter_exec:
                        # Get metadata from iter_exec or create default
                        if hasattr(iter_exec, "metadata") and iter_exec.metadata is not None:
                            metadata = iter_exec.metadata
                        else:
                            metadata = Metadata.create_leaf_success(
                                type=executor_type,
                                id=key,
                                duration_ms=0,
                                started_at=datetime.now(UTC).isoformat(),
                                wave=wave,
                                execution_order=completed_keys.index(key),
                                index=completed_keys.index(key),
                                depth=depth + 1,
                            )
                        iteration_results[key] = BlockExecution(
                            inputs=iter_exec.inputs if hasattr(iter_exec, "inputs") else {},
                            output=self._reconstruct_output(iter_exec, executor),
                            metadata=metadata,
                            paused=False,
                        )

        # Resume the paused iteration with user response
        # Re-resolve iteration inputs (needed for resume_block)
        each_context = {
            "key": current_key,
            "value": all_iterations[current_key],
            "index": current_idx,
            "count": iteration_count,
        }
        iteration_context_dict = context.model_dump()
        iteration_context_dict["each"] = each_context

        resolver = UnifiedVariableResolver(
            iteration_context_dict, secret_provider=self.secret_provider
        )
        resolved_inputs = await resolver.resolve_async(inputs_template)
        input_model = executor.input_type(**resolved_inputs)

        # Resume the paused iteration
        resumed_result = await self.resume_block(
            id=current_key,
            executor=executor,
            inputs=input_model,
            context=context,
            response=response,
            pause_metadata=paused_iter_checkpoint,
            wave=wave,
            execution_order=current_idx,
            depth=depth + 1,
        )
        iteration_results[current_key] = resumed_result

        # Execute remaining iterations (sequential execution)
        for idx_offset, key in enumerate(remaining_keys):
            idx = current_idx + 1 + idx_offset

            # Execute remaining iteration
            each_context = {
                "key": key,
                "value": all_iterations[key],
                "index": idx,
                "count": iteration_count,
            }
            iteration_context_dict = context.model_dump()
            iteration_context_dict["each"] = each_context

            resolver = UnifiedVariableResolver(
                iteration_context_dict, secret_provider=self.secret_provider
            )
            resolved_inputs = await resolver.resolve_async(inputs_template)
            input_model = executor.input_type(**resolved_inputs)

            result = await self.execute_block(
                id=key,
                executor=executor,
                inputs=input_model,
                context=context,
                wave=wave,
                execution_order=idx,
                depth=depth + 1,
            )

            iteration_results[key] = result

            # Check for pause (may pause again)
            if result.paused:
                # Validate pause_prompt is present
                if not result.pause_prompt:
                    raise RuntimeError(f"Block {key} marked as paused but pause_prompt is None")

                # Create new for_each checkpoint for this iteration
                # Track ALL completed iterations: previous + just-resumed + newly completed
                all_completed_keys = list(completed_keys)  # From checkpoint

                # Add the iteration we just resumed (now complete)
                if current_key not in all_completed_keys:
                    all_completed_keys.append(current_key)

                # Add any newly completed iterations from remaining_keys
                for k in remaining_keys[:idx_offset]:
                    if k not in all_completed_keys:
                        all_completed_keys.append(k)

                # Serialize completed iteration results for checkpoint (ADR-010)
                completed_iteration_results = {}
                for comp_key in all_completed_keys:
                    if comp_key in iteration_results:
                        exec_result = iteration_results[comp_key]
                        completed_iteration_results[comp_key] = {
                            "inputs": exec_result.inputs if hasattr(exec_result, "inputs") else {},
                            "output": (
                                exec_result.output.model_dump() if exec_result.output else None
                            ),
                            "metadata": exec_result.metadata.model_dump(),
                            "paused": exec_result.paused,
                        }

                for_each_checkpoint = {
                    "type": "for_each_iteration",
                    "for_each_block_id": block_id,
                    "current_iteration_key": key,
                    "current_iteration_index": idx,
                    "completed_iterations": all_completed_keys,
                    "completed_iteration_results": completed_iteration_results,
                    "remaining_iteration_keys": remaining_keys[idx_offset + 1 :],
                    "all_iterations": all_iterations,
                    "executor_type": executor_type,
                    "inputs_template": inputs_template,
                    "mode": mode,
                    "max_parallel": max_parallel,
                    "continue_on_error": continue_on_error,
                    "iteration_count": iteration_count,
                    "wave": wave,
                    "depth": depth,
                    "paused_iteration_checkpoint": result.pause_checkpoint_data,
                }
                raise ExecutionPaused(
                    prompt=result.pause_prompt,
                    checkpoint_data=for_each_checkpoint,
                    execution=context,
                )

            # Check for failure (fail-fast)
            if not continue_on_error and result.metadata.failed:
                # Mark remaining iterations as skipped
                for remaining_idx_offset in range(idx_offset + 1, len(remaining_keys)):
                    remaining_key = remaining_keys[remaining_idx_offset]
                    remaining_idx = current_idx + 1 + remaining_idx_offset
                    skipped_metadata = Metadata.create_leaf_skipped(
                        type=executor_type,
                        id=remaining_key,
                        started_at=datetime.now(UTC).isoformat(),
                        wave=wave,
                        execution_order=remaining_idx,
                        index=remaining_idx,
                        value=all_iterations[remaining_key],
                        depth=depth + 1,
                        message="Skipped due to previous iteration failure",
                    )
                    iteration_results[remaining_key] = BlockExecution(
                        output=None,
                        metadata=skipped_metadata,
                        paused=False,
                    )
                break

        # Aggregate results
        child_metas = [r.metadata for r in iteration_results.values()]
        parent_metadata = Metadata.create_for_each_parent(
            type=executor_type,
            id=block_id,
            iterations=all_iterations,
            child_metas=child_metas,
            mode=mode,
            depth=depth,
            index=0,
            value={},
        )

        return iteration_results, parent_metadata

    def _reconstruct_output(self, iter_exec: Any, executor: BlockExecutor) -> Any:
        """Reconstruct BlockOutput from stored Execution (ADR-010 helper)."""
        # Check if it's an Execution object (Workflow blocks use fractal pattern)
        if isinstance(iter_exec, Execution):
            # Return the full Execution (for Workflow blocks)
            return iter_exec

        # Check if iter_exec has outputs attribute (standard block execution)
        if hasattr(iter_exec, "outputs") and iter_exec.outputs:
            # Regular block - reconstruct output model
            try:
                return executor.output_type(**iter_exec.outputs)
            except Exception:
                # If output_type doesn't match, return None
                return None

        # No outputs available
        return None

    async def execute_for_each(
        self,
        id: str,
        executor: BlockExecutor,
        inputs_template: dict[str, Any],
        iterations: dict[str, Any],
        context: Execution,
        mode: Literal["parallel", "sequential"] = "parallel",
        max_parallel: int = 5,
        continue_on_error: bool = False,
        wave: int = 0,
        depth: int = 0,
    ) -> tuple[dict[str, BlockExecution], Metadata]:
        """
        Execute a for_each block with multiple iterations (ADR-009).

        Args:
            id: Block identifier
            executor: Block executor to run for each iteration
            inputs_template: Input template with {{each.*}} variables
            iterations: Dict mapping iteration keys to values
            context: Current execution context
            mode: Execution mode ("parallel" or "sequential")
            max_parallel: Maximum concurrent iterations (parallel mode only)
            continue_on_error: Continue on iteration failure
            wave: Wave number (for metadata)
            depth: Nesting depth (0 for root blocks, 1+ for nested iterations)

        Returns:
            Tuple of (iteration_results_dict, parent_metadata)
            - iteration_results_dict: Dict[iteration_key, BlockExecution]
            - parent_metadata: Aggregated Metadata for the for_each block

        Notes:
            - Creates iteration contexts with {{each.key}}, {{each.value}},
              {{each.index}}, {{each.count}}
            - Aggregates child Metadata into parent using
              Metadata.create_for_each_parent()
            - Handles continue_on_error: false (fail-fast) and true (resilient)
        """
        from .resolver import UnifiedVariableResolver

        # Validate iteration keys (ADR-009: security & stability)
        validate_iteration_keys(iterations)

        iteration_count = len(iterations)
        iteration_keys = list(iterations.keys())
        iteration_results: dict[str, BlockExecution] = {}

        # Helper to execute a single iteration
        async def execute_iteration(
            iteration_key: str, iteration_index: int, iteration_value: Any
        ) -> tuple[str, BlockExecution]:
            """Execute a single iteration with its own context."""
            # Create iteration context with 'each' namespace
            each_context = {
                "key": iteration_key,
                "value": iteration_value,
                "index": iteration_index,
                "count": iteration_count,
            }

            # Create augmented context with 'each' namespace
            iteration_context_dict = context.model_dump()
            iteration_context_dict["each"] = each_context

            # Resolve iteration inputs (replace {{each.*}} variables)
            resolver = UnifiedVariableResolver(
                iteration_context_dict, secret_provider=self.secret_provider
            )
            resolved_inputs = await resolver.resolve_async(inputs_template)

            # Validate and create input model
            input_model = executor.input_type(**resolved_inputs)

            # Execute this iteration (depth + 1 for nested tracking)
            result = await self.execute_block(
                id=iteration_key,
                executor=executor,
                inputs=input_model,
                context=context,
                wave=wave,
                execution_order=iteration_index,
                depth=depth + 1,
            )

            return iteration_key, result

        # Parallel mode: execute with concurrency control
        if mode == "parallel":
            semaphore = asyncio.Semaphore(max_parallel)

            async def execute_with_semaphore(
                key: str, index: int, value: Any
            ) -> tuple[str, BlockExecution]:
                async with semaphore:
                    return await execute_iteration(key, index, value)

            # Create tasks for all iterations
            tasks = [
                asyncio.create_task(execute_with_semaphore(key, idx, iterations[key]))
                for idx, key in enumerate(iteration_keys)
            ]

            # Catch ExecutionPaused in parallel mode (ADR-010: not supported)
            try:
                if continue_on_error:
                    # Resilient: run all iterations even if some fail
                    results = await asyncio.gather(*tasks, return_exceptions=False)
                    for key, result in results:
                        iteration_results[key] = result
                else:
                    # Fail-fast: cancel remaining on first failure
                    completed_keys = set()

                    for coro in asyncio.as_completed(tasks):
                        try:
                            key, result = await coro
                            iteration_results[key] = result
                            completed_keys.add(key)

                            # If iteration failed, cancel remaining
                            if result.metadata.failed:
                                # Cancel all pending tasks
                                for task in tasks:
                                    if not task.done():
                                        task.cancel()

                                # Mark remaining iterations as skipped
                                for remaining_key in iteration_keys:
                                    if remaining_key not in completed_keys:
                                        skipped_metadata = Metadata.create_leaf_skipped(
                                            type=executor.type_name,
                                            id=remaining_key,
                                            started_at=datetime.now(UTC).isoformat(),
                                            wave=wave,
                                            execution_order=iteration_keys.index(remaining_key),
                                            index=iteration_keys.index(remaining_key),
                                            value=iterations[remaining_key],
                                            depth=depth + 1,
                                            message="Skipped due to previous iteration failure",
                                        )
                                        iteration_results[remaining_key] = BlockExecution(
                                            output=None,
                                            metadata=skipped_metadata,
                                            paused=False,
                                        )
                                break

                        except asyncio.CancelledError:
                            # Task was cancelled, skip
                            continue
            except ExecutionPaused as e:
                # Parallel mode doesn't support pause/resume (ADR-010)
                raise NotImplementedError(
                    f"for_each block '{id}' paused in parallel mode. "
                    f"Pause/resume is only supported with for_each_mode: sequential. "
                    f"To use Prompt blocks in iterations, set for_each_mode: sequential."
                ) from e

            # Check for pauses in parallel mode (ADR-010: not supported)
            for key, result in iteration_results.items():
                if result.paused:
                    raise NotImplementedError(
                        f"for_each block '{id}' iteration '{key}' paused in parallel mode. "
                        f"Pause/resume is only supported with for_each_mode: sequential. "
                        f"To use Prompt blocks in iterations, set for_each_mode: sequential."
                    )

        # Sequential mode: execute in order
        else:
            for idx, key in enumerate(iteration_keys):
                result_key, result = await execute_iteration(key, idx, iterations[key])
                iteration_results[result_key] = result

                # Detect pause and bubble up with for_each context (ADR-010)
                if result.paused:
                    # Validate pause_prompt is present (should always be set when paused=True)
                    if not result.pause_prompt:
                        raise RuntimeError(
                            f"Block {result_key} marked as paused but pause_prompt is None"
                        )

                    # Create for_each checkpoint state
                    # completed_iterations should NOT include the currently paused iteration
                    completed_iteration_keys: list[str] = [
                        k for k in iteration_results.keys() if k != result_key
                    ]

                    # Serialize completed iteration results for checkpoint (ADR-010)
                    completed_iteration_results = {}
                    for key in completed_iteration_keys:
                        if key in iteration_results:
                            exec_result = iteration_results[key]
                            completed_iteration_results[key] = {
                                "inputs": exec_result.inputs
                                if hasattr(exec_result, "inputs")
                                else {},
                                "output": (
                                    exec_result.output.model_dump() if exec_result.output else None
                                ),
                                "metadata": exec_result.metadata.model_dump(),
                                "paused": exec_result.paused,
                            }

                    for_each_checkpoint = {
                        "type": "for_each_iteration",
                        "for_each_block_id": id,
                        "current_iteration_key": result_key,
                        "current_iteration_index": idx,
                        "completed_iterations": completed_iteration_keys,
                        "completed_iteration_results": completed_iteration_results,  # Store results
                        "remaining_iteration_keys": iteration_keys[idx + 1 :],
                        "all_iterations": iterations,
                        "executor_type": executor.type_name,
                        "inputs_template": inputs_template,
                        "mode": mode,
                        "max_parallel": max_parallel,
                        "continue_on_error": continue_on_error,
                        "iteration_count": iteration_count,
                        "wave": wave,
                        "depth": depth,
                        # Nested checkpoint from paused iteration
                        "paused_iteration_checkpoint": result.pause_checkpoint_data,
                    }

                    # Bubble up ExecutionPaused with for_each context
                    raise ExecutionPaused(
                        prompt=result.pause_prompt,
                        checkpoint_data=for_each_checkpoint,
                        execution=context,
                    )

                # Fail-fast: stop on first failure
                if not continue_on_error and result.metadata.failed:
                    # Mark remaining iterations as skipped
                    for remaining_idx in range(idx + 1, len(iteration_keys)):
                        remaining_key = iteration_keys[remaining_idx]
                        skipped_metadata = Metadata.create_leaf_skipped(
                            type=executor.type_name,
                            id=remaining_key,
                            started_at=datetime.now(UTC).isoformat(),
                            wave=wave,
                            execution_order=remaining_idx,
                            index=remaining_idx,
                            value=iterations[remaining_key],
                            depth=depth + 1,
                            message="Skipped due to previous iteration failure",
                        )
                        iteration_results[remaining_key] = BlockExecution(
                            output=None,
                            metadata=skipped_metadata,
                            paused=False,
                        )
                    break

        # Aggregate child metadata into parent Metadata
        child_metas = [result.metadata for result in iteration_results.values()]
        parent_metadata = Metadata.create_for_each_parent(
            type=executor.type_name,
            id=id,
            iterations=iterations,
            child_metas=child_metas,
            mode=mode,
            depth=depth,
            index=0,  # Root blocks have index 0
            value={},  # Root blocks have empty value
        )

        return iteration_results, parent_metadata
