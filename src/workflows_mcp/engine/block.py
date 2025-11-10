"""
Pydantic base models for executor input/output validation.

This module provides type-safe I/O validation for workflow block executors
using Pydantic v2.

Post ADR-006 Architecture:
- BlockInput: Strict input validation (extra='forbid')
- BlockOutput: Flexible output model (extra='allow')
- Executors return BaseModel directly (no Block wrapper class)

Key Benefits:
- Type safety (Pydantic v2 validation)
- Clear I/O contracts for each executor
- Automatic schema generation for MCP tools
"""

from typing import Any

from pydantic import BaseModel, Field


class BlockInput(BaseModel):
    """Base class for block input validation using Pydantic v2."""

    model_config = {"extra": "forbid"}  # Pydantic v2 config - reject unknown fields


class BlockOutput(BaseModel):
    """Base class for block output validation using Pydantic v2.

    Allows extra fields to support dynamic outputs from custom block configurations
    and child workflow outputs in Workflow blocks.

    Executors can populate meta with executor-specific metadata fields
    (e.g., exit_code for Shell, tokens_used for LLMCall). The orchestrator
    will merge these with core timing/execution fields when creating Metadata.
    """

    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Executor-specific metadata fields (exit_code, tokens_used, etc.)",
        exclude=True,
    )

    model_config = {"extra": "allow"}
