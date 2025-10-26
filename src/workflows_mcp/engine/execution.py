"""Unified execution model for fractal architecture."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from .metadata import Metadata


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

    # Namespaces (ADR-005 + unified)
    inputs: dict[str, Any] = Field(default_factory=dict)
    """Execution inputs (parameters)."""

    outputs: dict[str, Any] = Field(default_factory=dict)
    """Execution outputs (results)."""

    metadata: Metadata | dict[str, Any] = Field(default_factory=dict)
    """Execution metadata (state, timing, outcome)."""

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

    def set_block_result(
        self,
        block_id: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        metadata: Metadata,
    ) -> None:
        """
        Store a block execution result.

        Args:
            block_id: Block identifier
            inputs: Block inputs
            outputs: Block outputs
            metadata: Block execution metadata
        """
        self.blocks[block_id] = Execution(
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            blocks={},  # Leaf blocks have no sub-blocks
        )

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
        serialization to include ADR-005 shortcut accessors.
        """
        # Get base serialization
        data = super().model_dump(**kwargs)

        # If metadata is a Metadata object, ensure it's properly serialized
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
