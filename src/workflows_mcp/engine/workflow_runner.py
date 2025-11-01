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
"""

import asyncio
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from .checkpoint import CheckpointConfig, CheckpointState
from .context_vars import block_custom_outputs
from .exceptions import ExecutionPaused, RecursionDepthExceededError
from .execution import Execution
from .execution_context import ExecutionContext
from .execution_result import ExecutionResult
from .metadata import Metadata
from .orchestrator import BlockOrchestrator
from .schema import BlockDefinition, DependencySpec, WorkflowSchema
from .variables import ConditionEvaluator, InvalidConditionError, VariableResolver

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
        runner = WorkflowRunner(checkpoint_config)
        result = await runner.execute(workflow, inputs, context)
        response_dict = result.to_response(response_format="detailed")
    """

    def __init__(
        self,
        checkpoint_config: CheckpointConfig | None = None,
    ) -> None:
        """
        Initialize workflow runner.

        Args:
            checkpoint_config: Optional checkpoint configuration.
                If None, checkpointing is disabled (default for nested workflows and MCP tools).
                Pass CheckpointConfig() explicitly to enable checkpointing.
        """
        # Fix: When None is passed, disable checkpointing instead of using defaults
        # This ensures nested workflows don't create checkpoints (executors_workflow.py:145)
        self.checkpoint_config = (
            checkpoint_config if checkpoint_config is not None else CheckpointConfig(enabled=False)
        )
        self.orchestrator = BlockOrchestrator()

    async def execute(
        self,
        workflow: WorkflowSchema,
        runtime_inputs: dict[str, Any] | None = None,
        context: ExecutionContext | None = None,
    ) -> ExecutionResult:
        """
        Execute workflow and return ExecutionResult.

        ENHANCEMENT: Preserves partial execution on errors for debugging.

        Args:
            workflow: Workflow definition (Pydantic model with validated DAG)
            runtime_inputs: Runtime input overrides
            context: Execution context (optional, created if None)

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
        try:
            # Execute workflow (may raise exceptions)
            execution = await self._execute_workflow_internal(workflow, runtime_inputs, context)

            # Success - wrap in ExecutionResult
            return ExecutionResult.success(execution)

        except ExecutionPaused as e:
            # Workflow paused - execution context in exception
            return ExecutionResult.paused(
                checkpoint_id=e.checkpoint_data.get("checkpoint_id", ""),
                prompt=e.prompt,
                execution=e.execution,
                pause_metadata=e.checkpoint_data,
            )

        except Exception as e:
            logger.exception(f"Workflow execution failed: {e}")

            # Get partial execution from exception (attached in _execute_workflow_internal)
            partial_execution = getattr(e, "_partial_execution", None)

            if partial_execution is None:
                # Fallback: create minimal empty execution
                partial_execution = Execution(
                    inputs=runtime_inputs or {},
                    metadata={
                        "workflow_name": workflow.name,
                        "error": "Execution failed before context creation",
                    },
                    blocks={},
                )

            # Failure - wrap partial execution in ExecutionResult
            return ExecutionResult.failure(str(e), partial_execution)

    async def resume(
        self,
        checkpoint_id: str,
        response: str | None = None,
        context: ExecutionContext | None = None,
    ) -> ExecutionResult:
        """
        Resume workflow from checkpoint.

        Args:
            checkpoint_id: Checkpoint ID
            response: LLM response for paused workflows
            context: Execution context (must contain checkpoint_store)

        Returns:
            ExecutionResult with resumed execution
        """
        if context is None:
            raise ValueError("ExecutionContext required for resume operations")

        try:
            # Call internal resume method (returns Execution)
            execution = await self._resume_workflow_internal(checkpoint_id, response or "", context)

            # Success - wrap in ExecutionResult
            return ExecutionResult.success(execution)

        except ValueError as e:
            # Checkpoint or workflow not found
            minimal_execution = Execution(
                inputs={},
                metadata={"error": "Checkpoint not found", "checkpoint_id": checkpoint_id},
                blocks={},
            )
            return ExecutionResult.failure(str(e), minimal_execution)

        except ExecutionPaused as e:
            # Workflow paused again
            return ExecutionResult.paused(
                checkpoint_id=e.checkpoint_data.get("checkpoint_id", ""),
                prompt=e.prompt,
                execution=e.execution,
                pause_metadata=e.checkpoint_data,
            )

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
            self._finalize_execution_context(
                workflow, exec_context, completed_blocks, execution_waves, start_time
            )

            # Attach execution to exception for retrieval in error handler
            e._partial_execution = exec_context  # type: ignore[attr-defined]
            raise

        # Success - finalize normally
        self._finalize_execution_context(
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

                # Checkpoint after wave (crash recovery)
                if (
                    context
                    and self.checkpoint_config.enabled
                    and self.checkpoint_config.checkpoint_every_wave
                ):
                    await self._save_checkpoint(
                        workflow=workflow,
                        runtime_inputs=runtime_inputs,
                        context=context,
                        exec_context=exec_context,
                        completed_blocks=completed_blocks,
                        current_wave_index=wave_idx,
                        execution_waves=execution_waves,
                    )

        except RecursionDepthExceededError:
            # Recursion depth exceeded - bubble up immediately without checkpointing
            raise

        except ExecutionPaused as e:
            # Workflow paused during wave execution
            # Save ONE checkpoint with pause metadata
            if context:
                checkpoint_id = await self._save_checkpoint(
                    workflow=workflow,
                    runtime_inputs=runtime_inputs,
                    context=context,
                    exec_context=exec_context,
                    completed_blocks=completed_blocks,
                    current_wave_index=wave_idx,
                    execution_waves=execution_waves,
                    pause_exception=e,
                )

                # Update exception with saved checkpoint_id and workflow_name, then re-raise
                raise ExecutionPaused(
                    prompt=e.prompt,
                    checkpoint_data={
                        **e.checkpoint_data,
                        "checkpoint_id": checkpoint_id,
                        "workflow_name": workflow.name,
                    },
                    execution=e.execution,
                )
            else:
                # No checkpoint store available - re-raise without checkpoint
                raise

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
                should_execute = self._evaluate_condition(block_def.condition, exec_context)
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
        context = exec_context._internal.get("execution_context")
        if context is None:
            raise RuntimeError("ExecutionContext not found in Execution._internal")

        # 1. Resolve variables in inputs
        resolved_inputs = self._resolve_block_inputs(block_def.inputs, exec_context)

        # 2. Get executor from context
        executor = context.executor_registry.get(block_def.type)

        # 3. Create input model
        input_model = executor.input_type(**resolved_inputs)

        # 3.5. Set custom outputs using contextvars (with resolved paths)
        if block_def.outputs:
            # Create variable resolver from execution context
            context_dict = self._execution_to_dict(exec_context)
            resolver = VariableResolver(context_dict)

            # Resolve variables in output paths
            custom_outputs_dict = {}
            for name, output in block_def.outputs.items():
                output_dict = output.model_dump()
                # Resolve path variable substitution
                output_dict["path"] = resolver.resolve(output_dict["path"])
                custom_outputs_dict[name] = output_dict

            block_custom_outputs.set(custom_outputs_dict)
        else:
            block_custom_outputs.set(None)

        # 4. Execute via orchestrator
        block_execution = await self.orchestrator.execute_block(
            executor=executor,
            inputs=input_model,
            context=exec_context,
            wave=wave_idx,
            execution_order=execution_order,
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
                metadata=block_execution.metadata,
            )

    def _should_skip_block(
        self,
        block_id: str,
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
        metadata = Metadata.from_skipped(
            message=reason,
            timestamp=skip_time,
            wave=wave_idx,
            execution_order=execution_order,
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
        metadata = Metadata.from_execution_failure(
            message=error,
            execution_time_ms=0.0,
            started_at=fail_time,
            completed_at=fail_time,
            wave=wave_idx,
            execution_order=execution_order,
        )

        default_outputs = self._create_default_outputs(block_def.type, exec_context)

        exec_context.set_block_result(
            block_id=block_id,
            inputs={},
            outputs=default_outputs,
            metadata=metadata,
        )

    def _create_default_outputs(self, block_type: str, exec_context: Execution) -> dict[str, Any]:
        """Create default outputs for skipped/failed blocks."""
        context = exec_context._internal.get("execution_context")
        if context is None:
            return {}

        try:
            executor = context.executor_registry.get(block_type)
            output_model_class = executor.output_type

            defaults: dict[str, Any] = {}
            for field_name, field_info in output_model_class.model_fields.items():
                field_type = field_info.annotation

                # Type-based defaults
                if "str" in str(field_type):
                    defaults[field_name] = ""
                elif "int" in str(field_type):
                    defaults[field_name] = 0
                elif "float" in str(field_type):
                    defaults[field_name] = 0.0
                elif "bool" in str(field_type):
                    defaults[field_name] = False
                elif "dict" in str(field_type):
                    defaults[field_name] = {}
                elif "list" in str(field_type):
                    defaults[field_name] = []
                else:
                    defaults[field_name] = None

            return defaults

        except Exception:
            return {}

    def _create_initial_execution_context(
        self,
        workflow: WorkflowSchema,
        runtime_inputs: dict[str, Any] | None,
        context: ExecutionContext | None,
    ) -> Execution:
        """Create fresh Execution context for workflow start."""
        exec_context = Execution(
            inputs=self._merge_workflow_inputs(workflow, runtime_inputs),
            metadata={
                "workflow_name": workflow.name,
                "start_time": datetime.now(UTC).isoformat(),
            },
            blocks={},
            depth=len(context.workflow_stack) if context else 0,
        )

        # Store ExecutionContext in _internal (if provided)
        if context:
            exec_context._internal = {
                "execution_context": context,
                "workflow_stack": context.workflow_stack + [workflow.name],
            }
        else:
            exec_context._internal = {
                "execution_context": None,
                "workflow_stack": [workflow.name],
            }

        return exec_context

    def _finalize_execution_context(
        self,
        workflow: WorkflowSchema,
        exec_context: Execution,
        completed_blocks: list[str],
        execution_waves: list[list[str]],
        start_time: float | None = None,
    ) -> None:
        """Finalize execution context with outputs and metadata."""
        # Evaluate workflow outputs
        workflow_outputs = self._evaluate_workflow_outputs(workflow, exec_context)
        exec_context.outputs = workflow_outputs

        # Update metadata
        metadata_updates: dict[str, Any] = {
            "total_blocks": len(completed_blocks),
            "execution_waves": len(execution_waves),
            "completed_at": datetime.now(UTC).isoformat(),
        }

        if start_time is not None:
            metadata_updates["execution_time_seconds"] = time.time() - start_time

        if isinstance(exec_context.metadata, dict):
            existing_metadata = exec_context.metadata.copy()
            existing_metadata.update(metadata_updates)
            exec_context.metadata = existing_metadata
        else:
            workflow_name_value = getattr(exec_context.metadata, "workflow_name", "")
            exec_context.metadata = {
                "workflow_name": workflow_name_value,
                **metadata_updates,
            }

    def _execution_to_dict(self, exec_context: Execution) -> dict[str, Any]:
        """Convert Execution model to dict for variable resolution."""
        metadata = (
            exec_context.metadata
            if isinstance(exec_context.metadata, dict)
            else exec_context.metadata.model_dump()
        )
        blocks = {
            block_id: (block_exec.model_dump() if isinstance(block_exec, Execution) else block_exec)
            for block_id, block_exec in exec_context.blocks.items()
        }
        return {
            "inputs": exec_context.inputs,
            "metadata": metadata,
            "blocks": blocks,
        }

    def _evaluate_condition(self, condition: str, exec_context: Execution) -> bool:
        """Evaluate block condition."""
        try:
            context_dict = self._execution_to_dict(exec_context)
            evaluator = ConditionEvaluator()
            return evaluator.evaluate(condition, context_dict)
        except InvalidConditionError as e:
            raise ValueError(f"Condition evaluation failed: {e}")

    def _resolve_block_inputs(
        self, inputs: dict[str, Any], exec_context: Execution
    ) -> dict[str, Any]:
        """Resolve variables in block inputs."""
        context_dict = self._execution_to_dict(exec_context)
        resolver = VariableResolver(context_dict)
        resolved: dict[str, Any] = resolver.resolve(inputs)
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

    def _evaluate_workflow_outputs(
        self,
        workflow: WorkflowSchema,
        exec_context: Execution,
    ) -> dict[str, Any]:
        """Evaluate workflow-level outputs."""
        if not workflow.outputs:
            return {}

        outputs = {}
        context_dict = self._execution_to_dict(exec_context)
        resolver = VariableResolver(context_dict)
        evaluator = ConditionEvaluator()

        comparison_ops = ["==", "!=", ">=", "<=", ">", "<", " and ", " or ", " not "]

        for output_name, output_value in workflow.outputs.items():
            output_expr = output_value if isinstance(output_value, str) else output_value.value

            # Resolve variables
            resolved_value = resolver.resolve(output_expr)

            # Evaluate boolean expressions if present
            is_string = isinstance(resolved_value, str)
            has_operator = (
                any(op in resolved_value for op in comparison_ops) if is_string else False
            )
            if is_string and has_operator:
                try:
                    resolved_value = evaluator.evaluate(resolved_value, context_dict)
                except InvalidConditionError:
                    pass

            outputs[output_name] = resolved_value

        return outputs

    async def _save_checkpoint(
        self,
        workflow: WorkflowSchema,
        runtime_inputs: dict[str, Any],
        context: ExecutionContext,
        exec_context: Execution,
        completed_blocks: list[str],
        current_wave_index: int,
        execution_waves: list[list[str]],
        pause_exception: ExecutionPaused | None = None,
    ) -> str:
        """Save workflow checkpoint for crash recovery or pause/resume."""
        # Generate checkpoint ID
        prefix = "pause" if pause_exception else "chk"
        timestamp = int(time.time() * 1000)
        checkpoint_id = f"{prefix}_{workflow.name}_{timestamp}_{uuid.uuid4().hex[:8]}"

        # Extract workflow stack
        workflow_stack = context.workflow_stack

        # Convert workflow blocks to dict
        block_definitions = {block.id: block.model_dump() for block in workflow.blocks}

        # Extract pause metadata
        paused_block_id = None
        pause_prompt = None
        pause_metadata = None

        if pause_exception:
            paused_block_id = pause_exception.checkpoint_data.get("paused_block_id")
            pause_prompt = pause_exception.prompt
            pause_metadata = pause_exception.checkpoint_data

        # Create checkpoint state
        stack_list = [{"name": wf} for wf in workflow_stack]

        checkpoint_state = CheckpointState(
            checkpoint_id=checkpoint_id,
            workflow_name=workflow.name,
            created_at=time.time(),
            runtime_inputs=runtime_inputs,
            context=exec_context,
            completed_blocks=completed_blocks.copy(),
            current_wave_index=current_wave_index,
            execution_waves=execution_waves,
            block_definitions=block_definitions,
            workflow_stack=stack_list,
            paused_block_id=paused_block_id,
            pause_prompt=pause_prompt,
            pause_metadata=pause_metadata,
        )

        # Save checkpoint
        await context.checkpoint_store.save_checkpoint(checkpoint_state)

        logger.info(
            f"Saved {'pause ' if pause_exception else ''}checkpoint '{checkpoint_id}' "
            f"for workflow '{workflow.name}' at wave {current_wave_index}"
        )

        return checkpoint_id

    async def _resume_workflow_internal(
        self,
        checkpoint_id: str,
        response: str,
        context: ExecutionContext,
    ) -> Execution:
        """Resume workflow from checkpoint."""
        # Load checkpoint
        checkpoint = await context.checkpoint_store.load_checkpoint(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        # Get workflow
        workflow = context.get_workflow(checkpoint.workflow_name)
        if workflow is None:
            raise ValueError(f"Workflow not found: {checkpoint.workflow_name}")

        # Restore execution context
        exec_context = checkpoint.context
        workflow_stack = [ws.get("name", "") for ws in checkpoint.workflow_stack]
        exec_context._internal = {
            "execution_context": context,
            "workflow_stack": workflow_stack,
        }

        # Resume paused block if needed
        completed_blocks = checkpoint.completed_blocks.copy()
        if checkpoint.paused_block_id:
            paused_block_id = checkpoint.paused_block_id

            # Get block definition
            if paused_block_id not in checkpoint.block_definitions:
                raise ValueError(f"Paused block not found: {paused_block_id}")
            block_def = BlockDefinition(**checkpoint.block_definitions[paused_block_id])

            try:
                # Resume paused block
                await self._resume_paused_block(
                    block_id=paused_block_id,
                    block_def=block_def,
                    exec_context=exec_context,
                    response=response,
                    pause_metadata=checkpoint.pause_metadata or {},
                    wave_idx=checkpoint.current_wave_index,
                    execution_order=len(completed_blocks),
                )

                # Verify resumed block succeeded
                resumed_block_metadata = exec_context.get_block_metadata(paused_block_id)
                if resumed_block_metadata and resumed_block_metadata.status.is_failed():
                    error_msg = f"Failed to resume block '{paused_block_id}'"
                    error_detail = resumed_block_metadata.message
                    raise ValueError(f"{error_msg}: {error_detail}")

                completed_blocks.append(paused_block_id)

            except ExecutionPaused as e:
                # Block paused again during resume - save checkpoint and re-raise
                if context:
                    checkpoint_id = await self._save_checkpoint(
                        workflow=workflow,
                        runtime_inputs=checkpoint.runtime_inputs,
                        context=context,
                        exec_context=exec_context,
                        completed_blocks=completed_blocks,
                        current_wave_index=checkpoint.current_wave_index,
                        execution_waves=checkpoint.execution_waves,
                        pause_exception=e,
                    )

                    # Update exception with saved checkpoint_id and workflow_name, then re-raise
                    raise ExecutionPaused(
                        prompt=e.prompt,
                        checkpoint_data={
                            **e.checkpoint_data,
                            "checkpoint_id": checkpoint_id,
                            "workflow_name": workflow.name,
                        },
                        execution=e.execution,
                    )
                else:
                    # No checkpoint store available - re-raise without checkpoint
                    raise

        # Execute remaining waves
        await self._execute_waves_from(
            start_wave_index=checkpoint.current_wave_index + 1,
            execution_waves=checkpoint.execution_waves,
            workflow=workflow,
            runtime_inputs=checkpoint.runtime_inputs,
            context=context,
            exec_context=exec_context,
            completed_blocks=completed_blocks,
        )

        # Finalize context
        self._finalize_execution_context(
            workflow, exec_context, completed_blocks, checkpoint.execution_waves
        )

        return exec_context

    async def _resume_paused_block(
        self,
        block_id: str,
        block_def: BlockDefinition,
        exec_context: Execution,
        response: str,
        pause_metadata: dict[str, Any],
        wave_idx: int,
        execution_order: int,
    ) -> None:
        """Resume a paused block execution."""
        # Get execution context
        context = exec_context._internal.get("execution_context")
        if context is None:
            raise RuntimeError("ExecutionContext not found in Execution._internal")

        # Resolve inputs
        resolved_inputs = self._resolve_block_inputs(block_def.inputs, exec_context)

        # Get executor
        executor = context.executor_registry.get(block_def.type)

        # Create input model
        input_model = executor.input_type(**resolved_inputs)

        # Resume via orchestrator
        block_execution = await self.orchestrator.resume_block(
            executor=executor,
            inputs=input_model,
            context=exec_context,
            response=response,
            pause_metadata=pause_metadata,
            wave=wave_idx,
            execution_order=execution_order,
        )

        # Handle pause (block paused again)
        if block_execution.paused:
            pause_data = block_execution.pause_checkpoint_data or {}
            pause_data["paused_block_id"] = block_id

            raise ExecutionPaused(
                prompt=block_execution.pause_prompt or "Execution paused",
                checkpoint_data=pause_data,
                execution=exec_context,
            )

        # Store result
        if block_def.type == "Workflow":
            if block_execution.output is None:
                exec_context.blocks[block_id] = Execution(
                    inputs=resolved_inputs,
                    outputs={},
                    metadata=block_execution.metadata,
                    blocks={},
                )
            else:
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
            exec_context.set_block_result(
                block_id=block_id,
                inputs=resolved_inputs,
                outputs=block_execution.output.model_dump() if block_execution.output else {},
                metadata=block_execution.metadata,
            )


__all__ = ["WorkflowRunner"]
