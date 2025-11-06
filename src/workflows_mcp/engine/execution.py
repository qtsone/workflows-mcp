"""Unified execution model for fractal architecture."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .node_meta import NodeMeta


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

    # Use alias to support __meta__ field name (Pydantic handles the double-underscore)
    meta: NodeMeta | dict[str, Any] = Field(
        default_factory=dict,
        alias="__meta__",
        serialization_alias="__meta__",
    )
    """
    Universal node metadata (ADR-009 fractal for_each design).

    Unified structure supporting:
    - Regular blocks (count=1, mode=None, iterations={})
    - For_each iterations (count>1, mode="parallel"|"sequential", iterations={...})
    - Recursive nesting (iterations can contain iterations)

    See node_meta.py for NodeMeta documentation.
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

    # Internal namespace (hidden from variable resolution)
    _internal: dict[str, Any] = PrivateAttr(default_factory=dict)
    """Internal state not accessible in variable resolution."""

    @model_validator(mode="before")
    @classmethod
    def validate_meta_field(cls, data: Any) -> Any:
        """Convert dict meta to NodeMeta if needed (for checkpoint deserialization)."""
        if isinstance(data, dict):
            # Handle both 'meta' and '__meta__' keys
            meta_value = data.get("meta") or data.get("__meta__")
            if meta_value and isinstance(meta_value, dict):
                try:
                    data["meta"] = NodeMeta(**meta_value)
                    if "__meta__" in data:
                        data["__meta__"] = data["meta"]
                except Exception:
                    # If conversion fails, leave as dict
                    pass
        return data

    def set_block_result(
        self,
        block_id: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        meta: NodeMeta,
    ) -> None:
        """
        Store a block execution result (leaf block).

        Args:
            block_id: Block identifier
            inputs: Block inputs
            outputs: Block outputs
            meta: Block execution metadata (__meta__)
        """
        execution = Execution(inputs=inputs, outputs=outputs, blocks={})
        execution.meta = meta
        self.blocks[block_id] = execution

    def set_for_each_result(
        self,
        block_id: str,
        parent_meta: NodeMeta,
        iteration_results: dict[str, Any],
    ) -> None:
        """
        Store a for_each block execution result with nested iterations (ADR-009).

        Creates fractal structure where iterations are stored in the parent block's
        'blocks' field, enabling bracket notation access: blocks.id["key"].outputs.field

        Args:
            block_id: Block identifier
            parent_meta: Aggregated parent metadata (from NodeMeta.create_for_each_parent)
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
                outputs = result.output.model_dump() if result.output else {}
                meta = result.metadata
                inputs = {}  # Inputs not stored in BlockExecution
            else:
                # Tuple format: (inputs, outputs, meta)
                inputs, outputs, meta = result

            iter_exec = Execution(inputs=inputs, outputs=outputs, blocks={})
            iter_exec.meta = meta
            iteration_executions[iteration_key] = iter_exec

        # Store parent with iterations as child blocks
        parent_exec = Execution(inputs={}, outputs={}, blocks=iteration_executions)
        parent_exec.meta = parent_meta
        self.blocks[block_id] = parent_exec

    def get_block_metadata(self, block_id: str) -> NodeMeta | None:
        """
        Get metadata for a specific block.

        Args:
            block_id: Block identifier

        Returns:
            NodeMeta object if block exists, None otherwise
        """
        block = self.blocks.get(block_id)
        if block and isinstance(block.meta, NodeMeta):
            return block.meta
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
        Override model_dump to ensure NodeMeta computed properties are included.

        When Pydantic serializes nested models, it doesn't automatically use
        custom model_dump() methods. We need to explicitly handle NodeMeta
        serialization to include ADR-007 shortcut accessors (succeeded, failed, skipped, status).
        """
        # Get base serialization (includes __meta__ via alias)
        data = super().model_dump(**kwargs)

        # If __meta__ is a NodeMeta object, ensure it's properly serialized with computed properties
        if isinstance(self.meta, NodeMeta):
            data["__meta__"] = self.meta.model_dump(**kwargs)

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
