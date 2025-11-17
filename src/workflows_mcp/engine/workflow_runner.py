"""
Stateless workflow executor (WorkflowRunner).

Separates execution behavior from workflow definition.
Uses ExecutionContext for dependency injection.
Returns ExecutionResult for consistent error handling.

Design Principles:
- Stateless (no instance state between executions)
- Uses ExecutionContext for shared resources
- Preserves partial execution on errors
- Supports fractal composition via Workflow
- Integrates secrets management (provider, redactor, audit log)
"""

import asyncio
import logging
import shutil
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from .context_vars import block_custom_outputs
from .exceptions import ExecutionPaused, RecursionDepthExceededError
from .execution import Execution
from .execution_context import ExecutionContext
from .execution_result import ExecutionResult, ExecutionState
from .metadata import Metadata
from .orchestrator import BlockOrchestrator
from .resolver import UnifiedVariableResolver
from .schema import BlockDefinition, DependencySpec, WorkflowSchema
from .secrets import EnvVarSecretProvider, SecretAuditLog, SecretRedactor

logger = logging.getLogger(__name__)


class WorkflowRunner:
    """
    Stateless workflow executor implementing clean execution model.

    Replaces the old WorkflowExecutor with:
    - ExecutionResult return type (instead of WorkflowResponse)
    - ExecutionContext dependency injection (instead of registries)
    - Preserved partial execution on errors (for debugging)
    - Cleaner separation of concerns

    Usage:
        runner = WorkflowRunner()
        result = await runner.execute(workflow, inputs, context)
        response_dict = result.to_response(response_format="detailed")
    """

    def __init__(self) -> None:
        """Initialize workflow runner."""
        # Initialize secret management components
        self.secret_provider = EnvVarSecretProvider()
        self.secret_redactor = SecretRedactor(self.secret_provider)
        self.secret_audit_log: SecretAuditLog | None = None  # Created per execution

        # Initialize orchestrator with secret management
        self.orchestrator = BlockOrchestrator(
            secret_provider=self.secret_provider,
            secret_redactor=self.secret_redactor,
        )

    async def execute(
        self,
        workflow: WorkflowSchema,
        runtime_inputs: dict[str, Any] | None = None,
        context: ExecutionContext | None = None,
        debug: bool = False,
    ) -> ExecutionResult:
        """
        Execute workflow and return ExecutionResult.

        ENHANCEMENT: Preserves partial execution on errors for debugging.

        Args:
            workflow: Workflow definition (Pydantic model with validated DAG)
            runtime_inputs: Runtime input overrides
            context: Execution context (REQUIRED - provides registries and dependencies)
            debug: If True, preserve scratch directory for debugging

        Returns:
            ExecutionResult (success/failure/paused) with full execution context

        Examples:
            # Success
            result = await runner.execute(workflow, {"project": "app"})
            # result.status == "success"
            # result.execution.outputs == {...}

            # Failure (partial execution preserved!)
            result = await runner.execute(workflow, {"invalid": "input"})
            # result.status == "failure"
            # result.error == "Missing required input: project"
            # result.execution.blocks == {"block1": {...}}  # Completed before error!
        """
        # Validate required context parameter
        if context is None:
            raise ValueError(
                "ExecutionContext is required. "
                "Create one via AppContext.create_execution_context() "
                "or ExecutionContext(...) with proper registries."
            )

        scratch_dir: Path | None = None

        try:
            # Execute workflow (may raise exceptions)
            execution = await self._execute_workflow_internal(workflow, runtime_inputs, context)
            scratch_dir = execution.scratch_dir

            # Success - wrap in ExecutionResult with secret redactor
            return ExecutionResult.success(execution, self.secret_redactor)

        except ExecutionPaused as e:
            # Workflow paused - extract execution state (unified Job architecture)
            execution_state = e.checkpoint_data.get("execution_state")
            if not execution_state:
                raise RuntimeError("ExecutionPaused missing execution_state - invalid pause")

            return ExecutionResult.paused(
                prompt=e.prompt,
                execution=e.execution,
                execution_state=execution_state,
                pause_metadata=e.checkpoint_data,
                secret_redactor=self.secret_redactor,
            )

        except Exception as e:
            logger.exception(f"Workflow execution failed: {e}")

            # Get partial execution from exception (attached in _execute_workflow_internal)
            partial_execution = getattr(e, "_partial_execution", None)

            if partial_execution is None:
                # Fallback: create minimal empty execution
                # Workflow-level Execution has no metadata - only blocks have metadata
                partial_execution = Execution(
                    inputs=runtime_inputs or {},
                    metadata=None,
                    blocks={},
                )
            else:
                scratch_dir = partial_execution.scratch_dir

            # Failure - wrap partial execution in ExecutionResult with secret redactor
            return ExecutionResult.failure(str(e), partial_execution, self.secret_redactor)

        finally:
            # Cleanup scratch directory unless debug mode
            if scratch_dir and scratch_dir.exists() and not debug:
                try:
                    shutil.rmtree(scratch_dir)
                    logger.debug(f"Cleaned up scratch directory: {scratch_dir}")
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to cleanup scratch directory {scratch_dir}: {cleanup_error}"
                    )
            elif scratch_dir and debug:
                logger.info(f"Debug mode: scratch directory preserved at {scratch_dir}")

    async def resume_from_state(
        self,
        execution_state: "ExecutionState",
        response: str,
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Resume workflow from ExecutionState (unified Job architecture).

        This is the new unified resume method that works with ExecutionState
        embedded in Job.result, replacing the checkpoint-based resume system.

        Args:
            execution_state: ExecutionState containing all runtime state
            response: LLM response for paused workflows
            context: Execution context

        Returns:
            ExecutionResult with resumed execution

        Raises:
            ValueError: If workflow not found or invalid state
        """
        try:
            # Get workflow from registry
            workflow = context.get_workflow(execution_state.workflow_name)
            if not workflow:
                raise ValueError(f"Workflow not found: {execution_state.workflow_name}")

            # Reconstruct workflow blocks from block_definitions
            from .schema import BlockDefinition

            blocks = [
                BlockDefinition.model_validate(bd)
                for bd in execution_state.block_definitions.values()
            ]
            # Update workflow schema with blocks (necessary if workflow was modified)
            workflow.blocks = blocks

            # Call internal resume with execution state
            execution = await self._resume_from_execution_state(
                workflow=workflow,
                execution_state=execution_state,
                response=response,
                context=context,
            )

            # Success - wrap in ExecutionResult
            return ExecutionResult.success(execution, self.secret_redactor)

        except ExecutionPaused as e:
            # Workflow paused again - extract execution state
            execution_state_new = e.checkpoint_data.get("execution_state")
            if not execution_state_new:
                raise RuntimeError("ExecutionPaused missing execution_state - invalid pause")

            return ExecutionResult.paused(
                prompt=e.prompt,
                execution=e.execution,
                execution_state=execution_state_new,
                pause_metadata=e.checkpoint_data,
                secret_redactor=self.secret_redactor,
            )

        except Exception as e:
            # Catch-all for unexpected exceptions
            logger.exception("Unexpected error during workflow resume from state: %s", e)
            minimal_execution = Execution(
                inputs={},
                metadata=None,
                blocks={},
            )
            return ExecutionResult.failure(
                f"Unexpected error: {e}", minimal_execution, self.secret_redactor
            )

    async def _resume_from_execution_state(
        self,
        workflow: WorkflowSchema,
        execution_state: "ExecutionState",
        response: str,
        context: ExecutionContext,
    ) -> Execution:
        """
        Resume workflow from ExecutionState (internal implementation).

        Restores the complete execution context from ExecutionState and continues
        execution from where the workflow paused.

        Args:
            workflow: Workflow schema
            execution_state: ExecutionState with runtime state
            response: LLM response
            context: Execution context

        Returns:
            Execution with resumed workflow state
        """
        start_time = time.time()

        # Initialize secret audit log
        self.secret_audit_log = SecretAuditLog()
        await self.secret_redactor.initialize()

        # Restore execution context from ExecutionState
        exec_context = execution_state.context
        completed_blocks = execution_state.completed_blocks.copy()
        current_wave_index = execution_state.current_wave_index
        execution_waves = execution_state.execution_waves
        paused_block_id = execution_state.paused_block_id

        # Inject ExecutionContext (not serialized - runtime dependency)
        exec_context.set_execution_context(context)

        # Resume paused block with response
        try:
            block_id = paused_block_id
            block_definition_dict = execution_state.block_definitions.get(block_id)
            if not block_definition_dict:
                raise ValueError(f"Block definition not found for paused block: {block_id}")

            from .schema import BlockDefinition

            block_definition = BlockDefinition.model_validate(block_definition_dict)

            # Get executor for this block type
            executor = context.executor_registry.get(block_definition.type)
            if not executor:
                raise ValueError(f"Executor not found for block type: {block_definition.type}")

            # Create variable resolver to resolve block inputs
            context_dict = self._execution_to_dict(exec_context)
            resolver = UnifiedVariableResolver(
                context_dict,
                secret_provider=self.secret_provider,
                audit_log=self.secret_audit_log,
            )

            # Resolve block inputs (from execution state context)
            resolved_inputs: dict[str, Any] = await resolver.resolve_async(block_definition.inputs)
            block_inputs = executor.input_type(**resolved_inputs)

            # Get pause metadata from execution state
            pause_metadata = execution_state.pause_metadata or {}

            # Execute resume on block
            block_execution = await self.orchestrator.resume_block(
                id=block_id,
                executor=executor,
                inputs=block_inputs,
                context=exec_context,
                response=response,
                pause_metadata=pause_metadata,
            )

            # Store result using set_block_result (like regular block execution)
            exec_context.set_block_result(
                block_id=block_id,
                inputs=resolved_inputs,
                outputs=block_execution.output.model_dump() if block_execution.output else {},
                metadata=block_execution.metadata,
            )

            # Mark block as completed
            if block_id not in completed_blocks:
                completed_blocks.append(block_id)

        except ExecutionPaused as e:
            # Block paused again - create execution state and re-raise
            new_execution_state = self._create_execution_state(
                workflow=workflow,
                runtime_inputs=execution_state.runtime_inputs,
                context=context,
                exec_context=exec_context,
                completed_blocks=completed_blocks,
                current_wave_index=current_wave_index,
                execution_waves=execution_waves,
                pause_exception=e,
            )

            raise ExecutionPaused(
                prompt=e.prompt,
                checkpoint_data={
                    **e.checkpoint_data,
                    "execution_state": new_execution_state,
                    "workflow_name": workflow.name,
                },
                execution=e.execution,
            )

        # Continue executing remaining waves
        try:
            await self._execute_waves_from(
                start_wave_index=current_wave_index + 1,
                execution_waves=execution_waves,
                workflow=workflow,
                runtime_inputs=execution_state.runtime_inputs,
                context=context,
                exec_context=exec_context,
                completed_blocks=completed_blocks,
            )
        except ExecutionPaused:
            # Pause bubbles up naturally
            raise
        except Exception:
            # Finalize partial execution before re-raising
            await self._finalize_execution_context(
                workflow, exec_context, completed_blocks, execution_waves, start_time
            )
            raise

        # Finalize execution
        await self._finalize_execution_context(
            workflow, exec_context, completed_blocks, execution_waves, start_time
        )

        return exec_context

    async def _execute_workflow_internal(
        self,
        workflow: WorkflowSchema,
        runtime_inputs: dict[str, Any] | None = None,
        context: ExecutionContext | None = None,
    ) -> Execution:
        """
        Execute workflow from start (fresh context, wave 0).

        ENHANCEMENT: Preserves partial execution on errors by attaching
        to exception before re-raising.

        Args:
            workflow: Workflow definition
            runtime_inputs: Runtime input overrides
            context: Execution context (optional)

        Returns:
            Execution with complete workflow state

        Raises:
            ExecutionPaused: If workflow pauses (execution in exception)
            Exception: On execution errors (partial execution attached)
        """
        start_time = time.time()

        # Initialize secret audit log for this execution
        self.secret_audit_log = SecretAuditLog()

        # Initialize secret redactor patterns
        await self.secret_redactor.initialize()

        # Create execution context
        exec_context = self._create_initial_execution_context(workflow, runtime_inputs, context)

        # Get pre-computed execution waves from workflow
        execution_waves = workflow.execution_waves

        # Execute all waves with error preservation
        completed_blocks: list[str] = []

        try:
            await self._execute_waves_from(
                start_wave_index=0,
                execution_waves=execution_waves,
                workflow=workflow,
                runtime_inputs=runtime_inputs or {},
                context=context,
                exec_context=exec_context,
                completed_blocks=completed_blocks,
            )
        except RecursionDepthExceededError:
            # Recursion depth exceeded - critical error, fail immediately without finalization
            raise
        except ExecutionPaused:
            # Pause bubbles up naturally (execution already in exception)
            raise
        except Exception as e:
            # ENHANCEMENT: Finalize partial execution before re-raising
            await self._finalize_execution_context(
                workflow, exec_context, completed_blocks, execution_waves, start_time
            )

            # Attach execution to exception for retrieval in error handler
            e._partial_execution = exec_context  # type: ignore[attr-defined]
            raise

        # Success - finalize normally
        await self._finalize_execution_context(
            workflow, exec_context, completed_blocks, execution_waves, start_time
        )

        return exec_context

    async def _execute_waves_from(
        self,
        start_wave_index: int,
        execution_waves: list[list[str]],
        workflow: WorkflowSchema,
        runtime_inputs: dict[str, Any],
        context: ExecutionContext | None,
        exec_context: Execution,
        completed_blocks: list[str],
    ) -> None:
        """
        Execute workflow waves starting from given index.

        Handles wave-by-wave parallel execution with checkpoint saving
        and ExecutionPaused exception handling.

        Args:
            start_wave_index: Wave index to start from
            execution_waves: All execution waves
            workflow: Workflow schema
            runtime_inputs: Runtime inputs
            context: Execution context (for checkpointing)
            exec_context: Execution state
            completed_blocks: List of completed block IDs (mutated)

        Raises:
            ExecutionPaused: If any block pauses (with checkpoint_id)
        """
        wave_idx = start_wave_index - 1  # For exception handler

        try:
            for wave_idx in range(start_wave_index, len(execution_waves)):
                wave = execution_waves[wave_idx]

                # Execute wave
                executed_blocks = await self._execute_wave(
                    wave=wave,
                    wave_idx=wave_idx,
                    workflow=workflow,
                    exec_context=exec_context,
                    completed_blocks=completed_blocks,
                )
                completed_blocks.extend(executed_blocks)

        except RecursionDepthExceededError:
            # Recursion depth exceeded - bubble up immediately without checkpointing
            raise

        except ExecutionPaused as e:
            # Workflow paused during wave execution
            # Create execution state (unified Job architecture - no checkpoint save)
            execution_state = self._create_execution_state(
                workflow=workflow,
                runtime_inputs=runtime_inputs,
                context=context,
                exec_context=exec_context,
                completed_blocks=completed_blocks,
                current_wave_index=wave_idx,
                execution_waves=execution_waves,
                pause_exception=e,
            )

            # Update exception with execution_state and workflow_name, then re-raise
            raise ExecutionPaused(
                prompt=e.prompt,
                checkpoint_data={
                    **e.checkpoint_data,
                    "execution_state": execution_state,
                    "workflow_name": workflow.name,
                },
                execution=e.execution,
            )

    async def _execute_wave(
        self,
        wave: list[str],
        wave_idx: int,
        workflow: WorkflowSchema,
        exec_context: Execution,
        completed_blocks: list[str],
    ) -> list[str]:
        """
        Execute a single wave of blocks in parallel.

        Args:
            wave: List of block IDs to execute
            wave_idx: Wave index
            workflow: Workflow schema
            exec_context: Execution context
            completed_blocks: Already completed block IDs

        Returns:
            List of block IDs executed in this wave

        Raises:
            ExecutionPaused: If any block pauses
        """
        # Create block lookup
        blocks_by_id = {block.id: block for block in workflow.blocks}

        # Prepare execution tasks
        tasks = []
        block_ids = []

        for block_id in wave:
            block_def = blocks_by_id[block_id]

            # Check if should skip due to dependencies
            if self._should_skip_block(block_id, block_def.depends_on, exec_context):
                self._mark_block_skipped(
                    block_id=block_id,
                    block_def=block_def,
                    exec_context=exec_context,
                    wave_idx=wave_idx,
                    execution_order=len(completed_blocks),
                    reason="Parent dependency did not complete successfully",
                )
                continue

            # Check condition
            if block_def.condition:
                should_execute = await self._evaluate_condition(block_def.condition, exec_context)
                if not should_execute:
                    self._mark_block_skipped(
                        block_id=block_id,
                        block_def=block_def,
                        exec_context=exec_context,
                        wave_idx=wave_idx,
                        execution_order=len(completed_blocks),
                        reason=f"Condition '{block_def.condition}' evaluated to False",
                    )
                    continue

            # Add to execution tasks
            execution_order = len(completed_blocks)
            task = self._execute_block(block_id, block_def, exec_context, wave_idx, execution_order)
            tasks.append(task)
            block_ids.append(block_id)

        # Execute blocks in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for block_id, result in zip(block_ids, results):
                if isinstance(result, ExecutionPaused):
                    # Pause bubbles up immediately
                    raise result
                elif isinstance(result, RecursionDepthExceededError):
                    # Recursion depth exceeded - critical error, bubble up immediately
                    raise result
                elif isinstance(result, Exception):
                    # Execution error - mark as failed but continue
                    self._mark_block_failed(
                        block_id=block_id,
                        block_def=blocks_by_id[block_id],
                        exec_context=exec_context,
                        wave_idx=wave_idx,
                        execution_order=len(completed_blocks),
                        error=str(result),
                    )

        return block_ids

    async def _execute_block(
        self,
        block_id: str,
        block_def: BlockDefinition,
        exec_context: Execution,
        wave_idx: int,
        execution_order: int,
    ) -> None:
        """
        Execute a single block using BlockOrchestrator.

        Supports both regular blocks and for_each blocks (ADR-009).

        Args:
            block_id: Block ID
            block_def: Block definition
            exec_context: Execution context
            wave_idx: Wave index
            execution_order: Execution order

        Raises:
            ExecutionPaused: If block pauses
            Exception: On execution errors
        """
        # Get execution context from internal state
        context = exec_context.execution_context
        if context is None:
            raise RuntimeError("ExecutionContext not found in Execution._internal")

        # Get executor from context
        executor = context.executor_registry.get(block_def.type)

        # ADR-009: Check if block has for_each
        if block_def.for_each:
            await self._execute_for_each_block(
                block_id=block_id,
                block_def=block_def,
                executor=executor,
                exec_context=exec_context,
                wave_idx=wave_idx,
            )
            return

        # Regular block execution (existing logic)
        # 1. Resolve variables in inputs (async for secrets support)
        resolved_inputs = await self._resolve_block_inputs(block_def.inputs, exec_context)

        # 2. Create input model
        input_model = executor.input_type(**resolved_inputs)

        # 3.5. Set custom outputs using contextvars (with resolved paths)
        if block_def.outputs:
            # Create variable resolver from execution context
            context_dict = self._execution_to_dict(exec_context)
            resolver = UnifiedVariableResolver(
                context_dict,
                secret_provider=self.secret_provider,
                audit_log=self.secret_audit_log,
            )

            # Resolve variables in output paths (async for secrets support)
            custom_outputs_dict = {}
            for name, output in block_def.outputs.items():
                output_dict = output.model_dump()
                # Resolve path variable substitution
                output_dict["path"] = await resolver.resolve_async(output_dict["path"])
                custom_outputs_dict[name] = output_dict

            block_custom_outputs.set(custom_outputs_dict)
        else:
            block_custom_outputs.set(None)

        # 4. Execute via orchestrator
        block_execution = await self.orchestrator.execute_block(
            id=block_id,  # Pass block ID to orchestrator
            executor=executor,
            inputs=input_model,
            context=exec_context,
            wave=wave_idx,
            execution_order=execution_order,
            depth=exec_context.depth,
        )

        # 5. Handle pause
        if block_execution.paused:
            # Add execution context to exception (ENHANCEMENT)
            pause_data = block_execution.pause_checkpoint_data or {}
            pause_data["paused_block_id"] = block_id

            raise ExecutionPaused(
                prompt=block_execution.pause_prompt or "Execution paused",
                checkpoint_data=pause_data,
                execution=exec_context,  # Include execution context
            )

        # 6. Store result
        if block_def.type == "Workflow":
            # Special case: Workflow block returns child Execution
            if block_execution.output is None:
                # Workflow failed
                exec_context.blocks[block_id] = Execution(
                    inputs=resolved_inputs,
                    outputs={},
                    metadata=block_execution.metadata,
                    blocks={},
                )
            else:
                # Wrap child execution with block-level metadata
                assert isinstance(block_execution.output, Execution)
                child_exec = block_execution.output
                exec_context.blocks[block_id] = Execution(
                    inputs=resolved_inputs,
                    outputs=child_exec.outputs,
                    metadata=block_execution.metadata,
                    blocks=child_exec.blocks,
                    depth=child_exec.depth,  # Preserve recursion depth
                )
        else:
            # Regular block
            exec_context.set_block_result(
                block_id=block_id,
                inputs=resolved_inputs,
                outputs=block_execution.output.model_dump() if block_execution.output else {},
                metadata=block_execution.metadata,  # Metadata from orchestrator
            )

    async def _execute_for_each_block(
        self,
        block_id: str,
        block_def: BlockDefinition,
        executor: Any,
        exec_context: Execution,
        wave_idx: int,
    ) -> None:
        """
        Execute a for_each block with multiple iterations (ADR-009).

        Args:
            block_id: Block ID
            block_def: Block definition with for_each field
            executor: Block executor
            exec_context: Execution context
            wave_idx: Wave index

        Raises:
            ExecutionPaused: If any iteration pauses (not yet supported for for_each)
            Exception: On execution errors
        """
        # 1. Resolve for_each expression to get iterations
        context_dict = self._execution_to_dict(exec_context)

        # Get workflow name safely from internal metadata
        workflow_metadata = exec_context.workflow_metadata
        workflow_name = workflow_metadata.get("workflow_name", "")

        resolver = UnifiedVariableResolver(
            context_dict,
            secret_provider=self.secret_provider,
            audit_log=self.secret_audit_log,
            workflow_name=workflow_name,
            block_id=block_id,
        )
        for_each_value = await resolver.resolve_async(block_def.for_each)

        # 2. Convert to dict format (ADR-009: iterations are always dicts)
        if isinstance(for_each_value, list):
            # Convert list to dict with numeric string keys: ["a", "b"] → {"0": "a", "1": "b"}
            iterations = {str(i): value for i, value in enumerate(for_each_value)}
        elif isinstance(for_each_value, dict):
            # Already a dict, use as-is
            iterations = for_each_value
        else:
            raise ValueError(
                f"for_each expression must evaluate to dict or list, "
                f"got {type(for_each_value).__name__}: {block_def.for_each}"
            )

        # 3. Handle empty collection - mark block as skipped
        if not iterations:
            # Empty for_each is valid - mark block as skipped (like conditional execution)
            self._mark_block_skipped(
                block_id=block_id,
                block_def=block_def,
                exec_context=exec_context,
                wave_idx=wave_idx,
                execution_order=0,
                reason=f"for_each expression resulted in empty collection: {block_def.for_each}",
            )
            return

        # 4. Execute via orchestrator.execute_for_each()
        # Cast mode to Literal type for type safety
        mode = cast(Literal["parallel", "sequential"], block_def.for_each_mode)

        iteration_results, parent_meta = await self.orchestrator.execute_for_each(
            id=block_id,
            executor=executor,
            inputs_template=block_def.inputs,
            iterations=iterations,
            context=exec_context,
            mode=mode,
            max_parallel=block_def.max_parallel,
            continue_on_error=block_def.continue_on_error,
            wave=wave_idx,
            depth=exec_context.depth,
        )

        # 5. Store results in execution context using fractal structure
        exec_context.set_for_each_result(
            block_id=block_id,
            parent_meta=parent_meta,
            iteration_results=iteration_results,
        )

        # Note: Pause handling for for_each blocks is not yet implemented.
        # If any iteration pauses, the entire for_each block would need to pause,
        # storing iteration state in checkpoint. This is Phase 2+ enhancement.

    def _should_skip_block(
        self,
        _block_id: str,  # Reserved for future logging/debugging
        depends_on: list[DependencySpec],
        exec_context: Execution,
    ) -> bool:
        """Check if block should skip due to required dependencies."""
        if not depends_on:
            return False

        for dep_spec in depends_on:
            dep_metadata = exec_context.get_block_metadata(dep_spec.block)
            if dep_metadata:
                if dep_metadata.requires_dependent_skip(required=dep_spec.required):
                    return True

        return False

    def _mark_block_skipped(
        self,
        block_id: str,
        block_def: BlockDefinition,
        exec_context: Execution,
        wave_idx: int,
        execution_order: int,
        reason: str,
    ) -> None:
        """Mark a block as skipped."""
        skip_time = datetime.now(UTC).isoformat()
        metadata = Metadata.create_leaf_skipped(
            type=block_def.type,
            id=block_id,
            started_at=skip_time,
            wave=wave_idx,
            execution_order=execution_order,
            index=execution_order,
            depth=exec_context.depth,
            message=reason,
        )

        default_outputs = self._create_default_outputs(block_def.type, exec_context)

        exec_context.set_block_result(
            block_id=block_id,
            inputs={},
            outputs=default_outputs,
            metadata=metadata,
        )

    def _mark_block_failed(
        self,
        block_id: str,
        block_def: BlockDefinition,
        exec_context: Execution,
        wave_idx: int,
        execution_order: int,
        error: str,
    ) -> None:
        """Mark a block as failed due to execution error."""
        fail_time = datetime.now(UTC).isoformat()
        metadata = Metadata.create_leaf_failure(
            type=block_def.type,
            id=block_id,
            duration_ms=0,
            started_at=fail_time,
            wave=wave_idx,
            execution_order=execution_order,
            index=execution_order,
            depth=exec_context.depth,
            message=error,
        )

        default_outputs = self._create_default_outputs(block_def.type, exec_context)

        exec_context.set_block_result(
            block_id=block_id,
            inputs={},
            outputs=default_outputs,
            metadata=metadata,
        )

    def _create_default_outputs(self, block_type: str, exec_context: Execution) -> dict[str, Any]:
        """Create default outputs for skipped/failed blocks.

        All executor output models define Pydantic field defaults, so we can simply
        instantiate the model with no arguments to get a semantically meaningful
        "failed state" (e.g., status_code=0, success=False).

        This approach is consistent with orchestrator's crash handling.
        """
        context = exec_context.execution_context
        if context is None:
            return {}

        try:
            executor = context.executor_registry.get(block_type)
            # Instantiate output model with defaults, then convert to dict
            return executor.output_type().model_dump()

        except Exception:
            return {}

    def _create_initial_execution_context(
        self,
        workflow: WorkflowSchema,
        runtime_inputs: dict[str, Any] | None,
        context: ExecutionContext | None,
    ) -> Execution:
        """Create fresh Execution context for workflow start."""
        # Create workflow-scoped scratch directory
        execution_id = f"{workflow.name}-{uuid.uuid4().hex[:8]}"
        scratch_dir = Path(tempfile.gettempdir()) / f"workflows-{execution_id}"
        scratch_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        # Create execution context (workflow-level has no metadata - only blocks have metadata)
        workflow_start_time = datetime.now(UTC).isoformat()
        exec_context = Execution(
            inputs=self._merge_workflow_inputs(workflow, runtime_inputs),
            metadata=None,  # Workflow-level Execution has no block metadata
            blocks={},
            depth=len(context.workflow_stack) if context else 0,
        )

        # Set scratch directory
        exec_context.scratch_dir = scratch_dir

        # Store workflow metadata and ExecutionContext
        workflow_metadata_dict = {
            "workflow_name": workflow.name,
            "started_at": workflow_start_time,
            "execution_id": execution_id,
            "scratch_dir": str(scratch_dir),
        }

        if context:
            exec_context.set_execution_context(context)
            exec_context.workflow_stack = context.workflow_stack + [workflow.name]
            exec_context.workflow_metadata = workflow_metadata_dict
        else:
            exec_context.workflow_stack = [workflow.name]
            exec_context.workflow_metadata = workflow_metadata_dict

        logger.debug(f"Created scratch directory for workflow '{workflow.name}': {scratch_dir}")

        return exec_context

    async def _finalize_execution_context(
        self,
        workflow: WorkflowSchema,
        exec_context: Execution,
        completed_blocks: list[str],
        execution_waves: list[list[str]],
        start_time: float | None = None,
    ) -> None:
        """Finalize execution context with outputs and metadata (async for secrets support).

        CRITICAL: This method MUST NOT raise exceptions to ensure partial execution
        is preserved even if output evaluation fails.
        """
        # Evaluate workflow outputs (wrapped to catch resolution errors)
        try:
            workflow_outputs = await self._evaluate_workflow_outputs(workflow, exec_context)
            exec_context.outputs = workflow_outputs
        except Exception as output_error:
            # Output evaluation failed (e.g., variable resolution error)
            # Set outputs to empty dict with error info
            logger.warning(f"Failed to evaluate workflow outputs: {output_error}")
            exec_context.outputs = {
                "_error": f"Output evaluation failed: {output_error}",
                "_note": "Check debug log for partial execution details",
            }

        # Update metadata (safe operations, should not fail)
        metadata_updates: dict[str, Any] = {
            "total_blocks": len(completed_blocks),
            "execution_waves": len(execution_waves),
            "completed_at": datetime.now(UTC).isoformat(),
        }

        if start_time is not None:
            metadata_updates["execution_time_seconds"] = time.time() - start_time

        # Include secret audit summary if available
        if self.secret_audit_log:
            metadata_updates["secret_access_count"] = len(self.secret_audit_log.events)

        # Update workflow metadata in _internal storage
        exec_context.update_workflow_metadata(**metadata_updates)

    def _execution_to_dict(self, exec_context: Execution) -> dict[str, Any]:
        """Convert Execution model to dict for variable resolution."""
        # Get workflow metadata from _internal storage
        metadata = exec_context.workflow_metadata
        blocks = {
            block_id: (block_exec.model_dump() if isinstance(block_exec, Execution) else block_exec)
            for block_id, block_exec in exec_context.blocks.items()
        }
        return {
            "inputs": exec_context.inputs,
            "metadata": metadata,
            "blocks": blocks,
            "tmp": str(exec_context.scratch_dir) if exec_context.scratch_dir else "",
        }

    async def _evaluate_condition(self, condition: str, exec_context: Execution) -> bool:
        """Evaluate block condition using Jinja2 expression evaluation."""
        try:
            context_dict = self._execution_to_dict(exec_context)
            resolver = UnifiedVariableResolver(
                context_dict,
                secret_provider=self.secret_provider,
                audit_log=self.secret_audit_log,
            )
            # Resolve the condition expression - Jinja2 evaluates boolean expressions
            result = await resolver.resolve_async(condition)
            if not isinstance(result, bool):
                raise ValueError(f"Condition must evaluate to boolean, got {type(result).__name__}")
            return result
        except Exception as e:
            raise ValueError(f"Condition evaluation failed: {e}") from e

    async def _resolve_block_inputs(
        self, inputs: dict[str, Any], exec_context: Execution
    ) -> dict[str, Any]:
        """Resolve variables in block inputs (async for secrets support)."""
        context_dict = self._execution_to_dict(exec_context)
        resolver = UnifiedVariableResolver(
            context_dict,
            secret_provider=self.secret_provider,
            audit_log=self.secret_audit_log,
        )
        resolved: dict[str, Any] = await resolver.resolve_async(inputs)
        return resolved

    def _merge_workflow_inputs(
        self,
        workflow: WorkflowSchema,
        runtime_inputs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge default and runtime inputs."""
        merged = {}

        # Apply defaults
        for input_name, input_decl in workflow.inputs.items():
            if input_decl.default is not None:
                merged[input_name] = input_decl.default

        # Override with runtime inputs
        if runtime_inputs:
            merged.update(runtime_inputs)

        # Validate required inputs
        missing_inputs = []
        empty_string_inputs = []
        for input_name, input_decl in workflow.inputs.items():
            is_required = getattr(input_decl, "required", False)
            if is_required:
                if input_name not in merged:
                    missing_inputs.append(input_name)
                elif input_decl.type.value == "str" and merged[input_name] == "":
                    empty_string_inputs.append(input_name)

        errors = []
        if missing_inputs:
            errors.append(f"Missing required inputs: {', '.join(missing_inputs)}")
        if empty_string_inputs:
            errors.append(
                f"Required string inputs cannot be empty: {', '.join(empty_string_inputs)}"
            )

        if errors:
            raise ValueError("; ".join(errors))

        return merged

    async def _evaluate_workflow_outputs(
        self,
        workflow: WorkflowSchema,
        exec_context: Execution,
    ) -> dict[str, Any]:
        """
        Evaluate workflow-level outputs with type coercion (async for secrets support).

        All outputs use WorkflowOutputSchema with:
        - value: Expression referencing block outputs
        - type: Output type (defaults to str)
        - description: Optional documentation

        Type coercion applies based on declared type, converting resolved values to:
        str, int, float, bool, json, list, or dict.
        """
        if not workflow.outputs:
            return {}

        outputs = {}
        context_dict = self._execution_to_dict(exec_context)
        resolver = UnifiedVariableResolver(
            context_dict,
            secret_provider=self.secret_provider,
            audit_log=self.secret_audit_log,
        )

        for output_name, output_schema in workflow.outputs.items():
            # Extract expression and type from WorkflowOutputSchema
            output_expr = output_schema.value
            output_type = (
                output_schema.type.value
                if hasattr(output_schema.type, "value")
                else output_schema.type
            )

            # Resolve variables and expressions (async for secrets support)
            # UnifiedVariableResolver handles boolean expressions automatically via Jinja2
            resolved_value = await resolver.resolve_async(output_expr)

            # Apply type coercion
            from .executors_core import coerce_value_type

            try:
                resolved_value = coerce_value_type(resolved_value, output_type)
            except ValueError as e:
                # Type coercion failed - log error and keep original value
                logger.error(
                    f"Failed to coerce output '{output_name}' to type '{output_type}': {e}. "
                    f"Using uncoerced value: {resolved_value}"
                )

            outputs[output_name] = resolved_value

        return outputs

    def _create_execution_state(
        self,
        workflow: WorkflowSchema,
        runtime_inputs: dict[str, Any],
        context: ExecutionContext | None,
        exec_context: Execution,
        completed_blocks: list[str],
        current_wave_index: int,
        execution_waves: list[list[str]],
        pause_exception: ExecutionPaused | None = None,
    ) -> "ExecutionState":
        """
        Package runtime execution state for resume (unified Job architecture).

        This creates an ExecutionState containing all information needed to
        resume workflow execution from a pause point. Replaces checkpoint saving.

        Args:
            workflow: Workflow schema
            runtime_inputs: Original workflow inputs
            context: Execution context (None for top-level workflows)
            exec_context: Current execution state
            completed_blocks: List of completed block IDs
            current_wave_index: Current wave index in DAG
            execution_waves: Full DAG wave structure
            pause_exception: Optional pause exception with metadata

        Returns:
            ExecutionState ready for serialization in Job.result
        """
        # Extract workflow stack (empty for top-level workflows)
        workflow_stack = context.workflow_stack if context else []
        stack_list = [{"name": wf} for wf in workflow_stack]

        # Convert workflow blocks to dict
        block_definitions = {block.id: block.model_dump() for block in workflow.blocks}

        # Extract paused block ID and metadata from exception
        paused_block_id = ""
        pause_metadata = None
        if pause_exception:
            paused_block_id = pause_exception.checkpoint_data.get("paused_block_id", "")
            pause_metadata = pause_exception.checkpoint_data

        return ExecutionState(
            context=exec_context,
            completed_blocks=completed_blocks.copy(),
            current_wave_index=current_wave_index,
            execution_waves=execution_waves,
            block_definitions=block_definitions,
            workflow_stack=stack_list,
            paused_block_id=paused_block_id,
            workflow_name=workflow.name,
            runtime_inputs=runtime_inputs,
            pause_metadata=pause_metadata,
        )

    @staticmethod
    def _extract_execution_state(job_result: dict[str, Any]) -> "ExecutionState":
        """
        Extract ExecutionState from Job.result dict (unified Job architecture).

        Reconstructs ExecutionState from the serialized format stored in Job.result
        when a workflow pauses. This is the inverse of ExecutionState serialization
        in ExecutionResult._build_debug_data().

        Args:
            job_result: Job.result dict containing execution_state

        Returns:
            ExecutionState ready for resume

        Raises:
            ValueError: If execution_state missing or invalid format
        """
        # Extract execution_state from job result
        execution_state_dict = job_result.get("execution_state")
        if not execution_state_dict:
            raise ValueError(
                "Job result missing 'execution_state' field - cannot resume paused workflow"
            )

        # Reconstruct Execution context from dict
        from .execution import Execution

        context_dict = execution_state_dict.get("context")
        if not context_dict:
            raise ValueError("Execution state missing 'context' field")

        exec_context = Execution.model_validate(context_dict)

        # Reconstruct ExecutionState
        return ExecutionState(
            context=exec_context,
            completed_blocks=execution_state_dict.get("completed_blocks", []),
            current_wave_index=execution_state_dict.get("current_wave_index", 0),
            execution_waves=execution_state_dict.get("execution_waves", []),
            block_definitions=execution_state_dict.get("block_definitions", {}),
            workflow_stack=execution_state_dict.get("workflow_stack", []),
            paused_block_id=execution_state_dict.get("paused_block_id", ""),
            workflow_name=execution_state_dict.get("workflow_name", ""),
            runtime_inputs=execution_state_dict.get("runtime_inputs", {}),
        )


__all__ = ["WorkflowRunner"]
