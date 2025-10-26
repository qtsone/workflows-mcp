"""Checkpoint state management for workflow execution.

Provides immutable checkpoint state representation for pause/resume functionality.
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from .execution import Execution


class CheckpointState(BaseModel):
    """Immutable representation of workflow execution state at checkpoint.

    This captures all necessary information to resume workflow execution
    from a specific point in time.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint
        workflow_name: Name of the workflow being executed
        created_at: Unix timestamp when checkpoint was created
        runtime_inputs: Original inputs provided to workflow
        context: Current execution context (Execution model, not dict)
        completed_blocks: List of block IDs that have finished execution
        current_wave_index: Index of current execution wave
        execution_waves: List of execution waves (each wave is list of block IDs)
        block_definitions: Block configuration data
        workflow_stack: Stack for workflow composition (parent workflows)
        paused_block_id: ID of block that triggered pause (None for normal checkpoints)
        pause_prompt: Prompt for LLM when paused (None for normal checkpoints)
        pause_metadata: Additional metadata for pause (None for normal checkpoints)
    """

    model_config = {"arbitrary_types_allowed": True}

    checkpoint_id: str
    workflow_name: str
    created_at: float
    runtime_inputs: dict[str, Any]
    context: Execution  # Pydantic model field
    completed_blocks: list[str]
    current_wave_index: int
    execution_waves: list[list[str]]
    block_definitions: dict[str, Any]
    workflow_stack: list[dict[str, Any]] = Field(default_factory=list)
    paused_block_id: str | None = None
    pause_prompt: str | None = None
    pause_metadata: dict[str, Any] | None = None


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint behavior.

    Attributes:
        enabled: Whether checkpointing is enabled (default: True)
        checkpoint_every_wave: Create checkpoint after each wave completion (default: True)
    """

    enabled: bool = True
    checkpoint_every_wave: bool = True


@dataclass
class PauseData:
    """Data associated with a paused workflow execution.

    Attributes:
        prompt: Human-readable message explaining why workflow is paused
        checkpoint_id: Reference to the checkpoint where execution paused
        pause_metadata: Additional metadata about the pause (custom fields)
    """

    prompt: str
    checkpoint_id: str
    pause_metadata: dict[str, Any] = field(default_factory=dict)
