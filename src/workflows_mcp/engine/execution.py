"""Unified execution model for fractal architecture."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .metadata import Metadata

if TYPE_CHECKING:
    from .execution_context import ExecutionContext


class Execution(BaseModel):
    """
    Universal execution model (fractal/recursive).

    This is THE ONLY execution structure. Used for:
    - Workflows (top-level executions)
    - Blocks (child executions within workflows)
    - Nested workflows (via Workflow)

    Every execution has the same structure, enabling fractal composition.
    """

    model_config = {"arbitrary_types_allowed": True}

    # Namespaces (ADR-009 unified fractal architecture)
    inputs: dict[str, Any] = Field(default_factory=dict)
    """Execution inputs (parameters)."""

    outputs: dict[str, Any] = Field(default_factory=dict)
    """Execution outputs (results)."""

    # Execution metadata (ADR-009 fractal architecture)
    # Type is Metadata only - dicts are converted via model_validator during deserialization
    # No default - metadata must be explicitly set
    metadata: Metadata | None = Field(default=None)
    """
    Universal node metadata (ADR-009 fractal for_each design).

    Unified structure supporting:
    - Regular blocks (count=1, mode=None, iterations={})
    - For_each iterations (count>1, mode="parallel"|"sequential", iterations={...})
    - Recursive nesting (iterations can contain iterations)

    See metadata.py for Metadata documentation.
    """

    # Recursive structure - executions contain executions!
    blocks: dict[str, Execution] = Field(default_factory=dict)
    """
    Child executions (for workflows/composite blocks).

    Note: In workflow YAML definition, 'blocks' is a list of block specs.
          Here in execution context, 'blocks' is a dict of execution results
          keyed by block_id for efficient lookup during variable resolution.
    """

    # Recursion tracking (for workflow composition)
    depth: int = Field(default=0, description="Current recursion depth (0 for top-level workflows)")
    """
    Recursion depth tracking for nested workflow execution.

    Incremented when a Workflow block executes a child workflow, enabling
    recursive workflow calls with configurable depth limits. Top-level
    workflows start at depth 0, first-level nested workflows at depth 1, etc.

    Used to prevent infinite recursion by checking against ExecutionContext.max_recursion_depth.
    """

    # Serializable execution state (automatically serialized by Pydantic)
    workflow_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow-level metadata (name, start_time, etc.)",
    )
    """Workflow-level metadata (automatically serialized)."""

    workflow_stack: list[str] = Field(
        default_factory=list,
        description="Stack of workflow names for recursion tracking",
    )
    """Workflow composition stack (automatically serialized)."""

    scratch_dir: Path | None = Field(
        default=None,
        description="Workflow-scoped scratch directory (absolute path in system temp)",
    )
    """Workflow scratch directory (automatically serialized)."""

    # Non-serializable runtime dependency (PrivateAttr - not serialized by Pydantic)
    _execution_context: ExecutionContext | None = PrivateAttr(default=None)
    """Runtime execution context (registries, stores, queues) - injected on execution/resume."""

    # Accessor for execution context (runtime dependency injection)
    @property
    def execution_context(self) -> ExecutionContext | None:
        """Get the execution context (dependency injection)."""
        return self._execution_context

    def set_execution_context(self, ctx: ExecutionContext) -> None:
        """Set the execution context (dependency injection)."""
        self._execution_context = ctx

    # Convenience methods for workflow metadata
    def update_workflow_metadata(self, **kwargs: Any) -> None:
        """Update workflow metadata with new key-value pairs."""
        self.workflow_metadata.update(kwargs)

    # Convenience methods for workflow stack
    def push_workflow(self, workflow_name: str) -> None:
        """Push workflow onto stack (for recursion tracking)."""
        self.workflow_stack.append(workflow_name)

    @property
    def parent_workflow(self) -> str | None:
        """Get parent workflow name (for composition)."""
        return self.workflow_stack[-1] if self.workflow_stack else None

    @model_validator(mode="before")
    @classmethod
    def validate_metadata_field(cls, data: Any) -> Any:
        """
        Convert dict metadata to Metadata if needed (for checkpoint deserialization).

        This validator ensures that:
        1. When loading from checkpoint (dict with metadata key), convert to Metadata
        2. When creating programmatically (Metadata object), keep it as is
        """
        if isinstance(data, dict):
            metadata = data.get("metadata")
            if metadata:
                if isinstance(metadata, Metadata):
                    # Already a Metadata object, keep it
                    data["metadata"] = metadata
                elif isinstance(metadata, dict):
                    # Dict from checkpoint deserialization, convert to Metadata
                    try:
                        data["metadata"] = Metadata(**metadata)
                    except Exception:
                        # If conversion fails, leave as None
                        pass
        return data

    def set_block_result(
        self,
        block_id: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        metadata: Metadata,
    ) -> None:
        """
        Store a block execution result (leaf block).

        Args:
            block_id: Block identifier
            inputs: Block inputs
            outputs: Block outputs
            metadata: Block execution metadata
        """
        execution = Execution(inputs=inputs, outputs=outputs, blocks={}, metadata=metadata)
        self.blocks[block_id] = execution

    def set_for_each_result(
        self,
        block_id: str,
        parent_meta: Metadata,
        iteration_results: dict[str, Any],
    ) -> None:
        """
        Store a for_each block execution result with nested iterations (ADR-009).

        Creates fractal structure where iterations are stored in the parent block's
        'blocks' field, enabling bracket notation access: blocks.id["key"].outputs.field

        Args:
            block_id: Block identifier
            parent_meta: Aggregated parent metadata (from Metadata.create_for_each_parent)
            iteration_results: Dict mapping iteration_key to BlockExecution or
                             tuple of (inputs, outputs, meta)

        Example:
            # From execute_for_each() return value
            iteration_results, parent_meta = await orchestrator.execute_for_each(...)
            execution.set_for_each_result(
                "analyze_files",
                parent_meta,
                iteration_results  # Dict[str, BlockExecution]
            )

            # Access: blocks.analyze_files["file1"].outputs.response
        """
        # Create child Execution objects for each iteration
        iteration_executions: dict[str, Execution] = {}

        for iteration_key, result in iteration_results.items():
            # Handle both BlockExecution and tuple formats
            if hasattr(result, "output") and hasattr(result, "metadata"):
                # BlockExecution format (from execute_for_each)
                metadata = result.metadata
                inputs = result.inputs

                # Special handling for Workflow executors (fractal pattern)
                if isinstance(result.output, Execution):
                    # Workflow block - extract outputs and blocks separately
                    child_exec = result.output
                    iter_exec = Execution(
                        inputs=inputs,
                        outputs=child_exec.outputs,
                        blocks=child_exec.blocks,
                        metadata=metadata,
                    )
                else:
                    # Regular block - dump output model
                    outputs = result.output.model_dump() if result.output else {}
                    iter_exec = Execution(
                        inputs=inputs, outputs=outputs, blocks={}, metadata=metadata
                    )
            else:
                # Tuple format: (inputs, outputs, metadata)
                inputs, outputs, metadata = result
                iter_exec = Execution(inputs=inputs, outputs=outputs, blocks={}, metadata=metadata)

            iteration_executions[iteration_key] = iter_exec

        # Store parent with iterations as child blocks
        parent_exec = Execution(
            inputs={}, outputs={}, blocks=iteration_executions, metadata=parent_meta
        )
        self.blocks[block_id] = parent_exec

    def get_block_metadata(self, block_id: str) -> Metadata | None:
        """
        Get metadata for a specific block.

        Args:
            block_id: Block identifier

        Returns:
            Metadata object if block exists, None otherwise
        """
        block = self.blocks.get(block_id)
        if block and isinstance(block.metadata, Metadata):
            return block.metadata
        return None

    def get_block_output(self, block_id: str, output_key: str) -> Any:
        """
        Get a specific output from a block.

        Args:
            block_id: Block identifier
            output_key: Output field name

        Returns:
            Output value if exists, None otherwise
        """
        block = self.blocks.get(block_id)
        if block:
            return block.outputs.get(output_key)
        return None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """
        Override model_dump to ensure Metadata computed properties are included.

        When Pydantic serializes nested models, it doesn't automatically use
        custom model_dump() methods. We need to explicitly handle Metadata
        serialization to include ADR-007 shortcut accessors (succeeded, failed, skipped, status).
        """
        # Get base serialization
        data = super().model_dump(**kwargs)

        # If metadata is a Metadata object, ensure it's properly serialized with computed properties
        if isinstance(self.metadata, Metadata):
            data["metadata"] = self.metadata.model_dump(**kwargs)

        # Recursively handle nested Execution objects in blocks
        if "blocks" in data:
            blocks_dict = {}
            for block_id, block_exec in self.blocks.items():
                if isinstance(block_exec, Execution):
                    blocks_dict[block_id] = block_exec.model_dump(**kwargs)
                else:
                    blocks_dict[block_id] = block_exec
            data["blocks"] = blocks_dict

        return data
