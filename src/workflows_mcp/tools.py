"""MCP tool implementations for workflow execution.

This module contains all MCP tool function implementations that expose
workflow execution functionality to Claude Code via the MCP protocol.

Following official Anthropic MCP Python SDK patterns:
- Tool functions decorated with @mcp.tool()
- Flat parameter signatures with Annotated types for validation
- Type hints for automatic schema generation
- Async functions for all tools
- Clear docstrings (become tool descriptions)
"""

import json
from datetime import datetime
from typing import Annotated, Any, Literal

from mcp.types import ToolAnnotations
from pydantic import Field

from .context import AppContextType
from .engine import WorkflowRunner, load_workflow_from_yaml
from .engine.checkpoint import CheckpointConfig
from .formatting import (
    format_checkpoint_info_markdown,
    format_checkpoint_list_markdown,
    format_checkpoint_not_found_error,
    format_workflow_info_markdown,
    format_workflow_list_markdown,
    format_workflow_not_found_error,
)
from .server import mcp

# =============================================================================
# MCP Tools (following official SDK decorator pattern)
# =============================================================================


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,  # Execution creates side effects
        openWorldHint=True,  # Interacts with external systems via Shell blocks
    )
)
async def execute_workflow(
    workflow: Annotated[
        str,
        Field(
            description=(
                "Workflow name to execute (e.g., 'python-ci-pipeline', 'sequential-echo', "
                "'git-git-checkout-branch'). Use list_workflows() to see all available workflows."
            ),
            min_length=1,
            max_length=200,
        ),
    ],
    inputs: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Runtime inputs as key-value pairs for block variable substitution. "
                "Example: {'project_name': 'my-app', 'branch_name': 'feature/new-ui'}. "
                "Use get_workflow_info(workflow) to see required inputs for a workflow."
            )
        ),
    ] = None,
    response_format: Annotated[
        Literal["minimal", "detailed"],
        Field(
            description=(
                "Output verbosity control (default: 'minimal' for normal execution):\n"
                "- 'minimal': Returns status, outputs, and errors.\n"
                "- 'detailed': Includes full block execution details. Use when:\n"
                "  * Debugging workflow failures\n"
                "  * Investigating unexpected behavior\n"
                "  * Analyzing block-level timing\n"
                "WARNING: 'detailed' mode increases token usage significantly."
            )
        ),
    ] = "minimal",
    *,
    ctx: AppContextType,
) -> dict[str, Any]:
    """Execute a DAG-based workflow with inputs.

    Supports git operations, bash commands, templates, and workflow composition.

    Returns:
        Workflow execution result with structure:
        {
            "status": "success" | "failure" | "paused",
            "outputs": {...},  # Workflow outputs
            "blocks": {...},   # Block execution details (detailed mode only)
            "metadata": {...}, # Execution metadata (detailed mode only)
            "error": "...",    # Error message (if status is "failure")
        }

    Examples:
        - Execute python-ci-pipeline: workflow="python-ci-pipeline"
        - Run sequential-echo: workflow="sequential-echo", inputs={"message": "Hello"}
    """
    # Validate context availability
    if ctx is None:
        return {
            "status": "failure",
            "error": "Server context not available. Tool requires context to access resources.",
        }

    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context
    registry = app_ctx.registry

    # Validate workflow exists
    if workflow not in registry:
        available = registry.list_names()
        return {
            "status": "failure",
            "error": (
                f"Workflow '{workflow}' not found. "
                f"Available workflows: {', '.join(available[:5])}"
                f"{' (and more)' if len(available) > 5 else ''}. "
                "Use list_workflows() to see all workflows or filter by tags."
            ),
            "available_workflows": available,
        }

    # Get workflow schema
    workflow_schema = registry.get(workflow)
    if workflow_schema is None:
        return {
            "status": "failure",
            "error": f"Failed to load workflow '{workflow}' from registry.",
        }

    # Create execution context
    exec_context = app_ctx.create_execution_context()

    # Create WorkflowRunner and execute
    # Enable checkpointing for top-level MCP tool execution (None only for nested workflows)
    runner = WorkflowRunner(checkpoint_config=CheckpointConfig())
    result = await runner.execute(
        workflow=workflow_schema,
        runtime_inputs=inputs,
        context=exec_context,
    )

    # Format response using ExecutionResult.to_response()
    return result.to_response(response_format)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,  # Execution creates side effects
        openWorldHint=True,  # Interacts with external systems via Shell blocks
    )
)
async def execute_inline_workflow(
    workflow_yaml: Annotated[
        str,
        Field(
            description=(
                "Complete workflow definition as YAML string including name, description, blocks. "
                "Must be valid YAML syntax following the workflow schema. "
                "Use validate_workflow_yaml() to check YAML before execution."
            ),
            min_length=10,
            max_length=100000,
        ),
    ],
    inputs: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Runtime inputs as key-value pairs for block variable substitution. "
                "Example: {'source_path': 'src/', 'output_dir': 'dist/'}."
            )
        ),
    ] = None,
    response_format: Annotated[
        Literal["minimal", "detailed"],
        Field(
            description=(
                "Output verbosity control (default: 'minimal' for normal execution):\n"
                "- 'minimal': Returns status, outputs, and errors.\n"
                "- 'detailed': Includes full block execution details. Use when:\n"
                "  * Debugging workflow failures\n"
                "  * Investigating unexpected behavior\n"
                "  * Analyzing block-level timing\n"
                "WARNING: 'detailed' mode increases token usage significantly."
            )
        ),
    ] = "minimal",
    *,
    ctx: AppContextType,
) -> dict[str, Any]:
    """Execute a workflow provided as YAML string without registering it.

    Enables dynamic workflow execution without file system modifications.
    Useful for ad-hoc workflows, one-off tasks, or testing workflow definitions.

    Returns:
        Workflow execution result with structure similar to execute_workflow.
    """
    # Validate context availability
    if ctx is None:
        return {
            "status": "failure",
            "error": "Server context not available. Tool requires context to access resources.",
        }

    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context

    # Parse YAML string to WorkflowSchema
    load_result = load_workflow_from_yaml(workflow_yaml, source="<inline-workflow>")

    if not load_result.is_success:
        return {
            "status": "failure",
            "error": (
                f"Failed to parse workflow YAML: {load_result.error}. "
                "Ensure your YAML is valid and includes required fields: "
                "'name', 'description', and 'blocks'. "
                "Use validate_workflow_yaml() to check YAML syntax before execution."
            ),
        }

    workflow_schema = load_result.value
    if workflow_schema is None:
        return {
            "status": "failure",
            "error": (
                "Workflow definition parsing returned None. "
                "The YAML structure may be invalid. "
                "Required fields: 'name' (string), 'description' (string), "
                "'blocks' (list of block definitions). "
                "Use validate_workflow_yaml() to validate your workflow YAML."
            ),
        }

    # Create execution context
    exec_context = app_ctx.create_execution_context()

    # Create WorkflowRunner and execute (no registration needed for inline workflows)
    # Enable checkpointing for top-level MCP tool execution (None only for nested workflows)
    runner = WorkflowRunner(checkpoint_config=CheckpointConfig())
    result = await runner.execute(
        workflow=workflow_schema,
        runtime_inputs=inputs,
        context=exec_context,
    )

    # Format response using ExecutionResult.to_response()
    return result.to_response(response_format)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        idempotentHint=True,
    )
)
async def list_workflows(
    tags: Annotated[
        list[str],
        Field(
            description=(
                "Optional list of tags to filter workflows (e.g. ['git', 'ci']). "
                "Workflows matching ALL tags are returned (AND logic). "
                "Empty list returns all workflows."
            ),
            max_length=20,
        ),
    ] = [],
    format: Annotated[  # noqa: A002
        Literal["json", "markdown"],
        Field(
            description=(
                "Response format:\n"
                "  - json: JSON string with list of workflow names\n"
                "  - markdown: Human-readable formatted list with headers"
            )
        ),
    ] = "json",
    *,
    ctx: AppContextType,
) -> str:
    """
    List all available workflows, optionally filtered by tags.

    Discover workflows by name or filter by tags to find workflows for specific tasks.
    Returns workflow names only - use `get_workflow_info()` for detailed information.

    Returns:
        JSON format: JSON string with list of workflow names
        Markdown format: Formatted list with headers
    """
    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context
    registry = app_ctx.registry

    workflows = registry.list_names(tags=tags or [])

    if format == "markdown":
        return format_workflow_list_markdown(workflows, tags or None)
    else:
        # Return JSON string for programmatic access
        return json.dumps(workflows)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        idempotentHint=True,
    )
)
async def get_workflow_info(
    workflow: Annotated[
        str,
        Field(
            description=(
                "Workflow name to retrieve information about (e.g., 'python-ci-pipeline'). "
                "Use list_workflows() to see all available workflows."
            ),
            min_length=1,
            max_length=200,
        ),
    ],
    format: Annotated[  # noqa: A002
        Literal["json", "markdown"],
        Field(
            description=(
                "Response format:\n"
                "- 'json': Structured data with workflow metadata, blocks, inputs, outputs\n"
                "- 'markdown': Human-readable formatted description with sections"
            )
        ),
    ] = "json",
    *,
    ctx: AppContextType,
) -> dict[str, Any] | str:
    """Get detailed information about a specific workflow.

    Retrieve comprehensive metadata including description, blocks, inputs, outputs,
    and dependencies. Use before executing a workflow to understand its requirements.

    Returns:
        JSON format: Structured workflow metadata
        Markdown format: Human-readable formatted description
    """
    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context
    registry = app_ctx.registry

    # Check if workflow exists
    if workflow not in registry:
        return format_workflow_not_found_error(workflow, registry.list_names(), format)

    # Get metadata from registry
    metadata = registry.get_workflow_metadata(workflow)

    # Get workflow definition for block details
    workflow_def = registry.get(workflow)

    # Build comprehensive info dictionary
    info: dict[str, Any] = {
        "name": metadata["name"],
        "description": metadata["description"],
        "version": metadata.get("version", "1.0"),
        "total_blocks": len(workflow_def.blocks),
        "blocks": [
            {
                "id": block.id,
                "type": block.type,
                "depends_on": [dep.block for dep in block.depends_on],
            }
            for block in workflow_def.blocks
        ],
    }

    # Add optional metadata fields
    if "author" in metadata:
        info["author"] = metadata["author"]
    if "tags" in metadata:
        info["tags"] = metadata["tags"]

    # Add input/output schema if available
    if workflow_def:
        # Convert input declarations to simple type mapping
        if workflow_def.inputs:
            info["inputs"] = {
                name: {"type": decl.type.value, "description": decl.description}
                for name, decl in workflow_def.inputs.items()
            }

        # Add output mappings if available
        if workflow_def.outputs:
            info["outputs"] = workflow_def.outputs

    # Format as markdown if requested
    if format == "markdown":
        return format_workflow_info_markdown(info)

    return info


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        idempotentHint=True,
    )
)
async def get_workflow_schema() -> dict[str, Any]:
    """Get complete JSON Schema for workflow validation.

    Returns the auto-generated JSON Schema that describes the structure of
    workflow YAML files, including all registered block types and their inputs.

    This schema can be used for:
    - Pre-execution validation
    - Editor autocomplete (VS Code YAML extension)
    - Documentation generation
    - Client-side validation

    Returns:
        Complete JSON Schema for workflow definitions
    """
    # Schema can be generated from executor registry without context
    from .engine.executor_base import create_default_registry

    # Create registry with all built-in executors and generate schema
    registry = create_default_registry()
    schema: dict[str, Any] = registry.generate_workflow_schema()
    return schema


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        idempotentHint=True,
    )
)
async def validate_workflow_yaml(
    yaml_content: Annotated[
        str,
        Field(
            description=(
                "YAML workflow definition to validate. Must be complete workflow YAML "
                "including name, description, and blocks sections."
            ),
            min_length=10,
            max_length=100000,
        ),
    ],
) -> dict[str, Any]:
    """Validate workflow YAML against schema before execution.

    Performs comprehensive validation without executing the workflow. Use before
    execute_inline_workflow to catch errors early and get clear validation feedback.

    Returns:
        Validation result with detailed feedback:
        {
            "valid": bool,                    # True if all validation passes
            "errors": list[str],              # List of validation errors (empty if valid)
            "warnings": list[str],            # List of warnings (non-blocking issues)
            "block_types_used": list[str]     # List of block types found in workflow
        }
    """
    # Parse workflow YAML
    load_result = load_workflow_from_yaml(yaml_content, source="<validation>")

    if not load_result.is_success:
        return {
            "valid": False,
            "errors": [
                f"YAML parsing error: {load_result.error}",
                "Common issues: Invalid YAML syntax, missing required fields "
                "('name', 'description', 'blocks'), or incorrect indentation. "
                "Check your YAML syntax with a YAML validator.",
            ],
            "warnings": [],
            "block_types_used": [],
        }

    workflow_def = load_result.value
    if workflow_def is None:
        return {
            "valid": False,
            "errors": [
                "Workflow definition parsing returned None - YAML structure is invalid.",
                "Required fields: 'name' (string), 'description' (string), "
                "'blocks' (list of block definitions with 'id', 'type', and 'inputs').",
                "Each block must have: id (unique identifier), type (executor type), "
                "and inputs (parameters for the executor).",
            ],
            "warnings": [],
            "block_types_used": [],
        }

    # Extract block types used
    block_types_used = list({block.type for block in workflow_def.blocks})

    # Validate block types against executor registry
    from .engine.executor_base import create_default_registry

    errors: list[str] = []
    warnings: list[str] = []

    registry = create_default_registry()
    registered_types = registry.list_types()

    for block in workflow_def.blocks:
        block_type = block.type
        if block_type not in registered_types:
            errors.append(
                f"Unknown block type '{block_type}' in block '{block.id}'. "
                f"Available block types: {', '.join(sorted(registered_types))}. "
                "Check for typos or use get_workflow_schema() to see all valid block types."
            )

    # If no errors, workflow is valid
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "block_types_used": block_types_used,
    }


# =============================================================================
# Checkpoint Management Tools
# =============================================================================


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
async def resume_workflow(
    checkpoint_id: Annotated[
        str,
        Field(
            description=(
                "Checkpoint token from pause or list_checkpoints "
                "(e.g., 'pause_abc123', 'auto_def456'). "
                "Use list_checkpoints() to see all available checkpoints."
            ),
            min_length=1,
            max_length=100,
        ),
    ],
    response: Annotated[
        str,
        Field(
            description=(
                "Your response to the pause prompt (required for paused workflows). "
                "For Prompt blocks, provide appropriate text based on the prompt. "
                "The workflow will interpret your response using conditions or subsequent blocks."
            ),
            max_length=10000,
        ),
    ] = "",
    response_format: Annotated[
        Literal["minimal", "detailed"],
        Field(
            description=(
                "Output verbosity control (default: 'minimal' for normal execution):\n"
                "- 'minimal': Returns status, outputs, and errors.\n"
                "- 'detailed': Includes full block execution details. Use when:\n"
                "  * Debugging workflow failures\n"
                "  * Investigating unexpected behavior\n"
                "  * Analyzing block-level timing\n"
                "WARNING: 'detailed' mode increases token usage significantly."
            )
        ),
    ] = "minimal",
    *,
    ctx: AppContextType,
) -> dict[str, Any]:
    """Resume a paused or checkpointed workflow.

    Continue workflow execution from a checkpoint created during pause (interactive blocks)
    or automatic checkpointing. Provides crash recovery and interactive workflow support.

    Returns:
        Workflow execution result similar to execute_workflow.
    """
    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context

    # Create execution context
    exec_context = app_ctx.create_execution_context()

    # Create WorkflowRunner and resume
    # Enable checkpointing for top-level MCP tool execution (None only for nested workflows)
    runner = WorkflowRunner(checkpoint_config=CheckpointConfig())
    result = await runner.resume(
        checkpoint_id=checkpoint_id,
        response=response,
        context=exec_context,
    )

    # Format response using ExecutionResult.to_response()
    return result.to_response(response_format)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        idempotentHint=True,
    )
)
async def list_checkpoints(
    workflow_name: Annotated[
        str,
        Field(
            description=(
                "Filter checkpoints by workflow name (e.g., 'python-ci-pipeline'). "
                "Empty string returns checkpoints for all workflows."
            ),
            max_length=200,
        ),
    ] = "",
    format: Annotated[  # noqa: A002
        Literal["json", "markdown"],
        Field(
            description=(
                "Response format:\n"
                "- 'json': Structured data with checkpoints list and metadata\n"
                "- 'markdown': Human-readable formatted list with checkpoint details"
            )
        ),
    ] = "json",
    *,
    ctx: AppContextType,
) -> dict[str, Any] | str:
    """List available workflow checkpoints.

    Shows all checkpoints, including both automatic checkpoints (for crash recovery)
    and pause checkpoints (for interactive workflows).

    Returns:
        JSON format: Dictionary with checkpoints list and total count
        Markdown format: Formatted string with headers and checkpoint details
    """
    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context

    filter_name = workflow_name if workflow_name else None
    checkpoints = await app_ctx.checkpoint_store.list_checkpoints(filter_name)

    checkpoint_data = [
        {
            "checkpoint_id": c.checkpoint_id,
            "workflow": c.workflow_name,
            "created_at": c.created_at,
            "created_at_iso": datetime.fromtimestamp(c.created_at).isoformat(),
            "is_paused": c.paused_block_id is not None,
            "pause_prompt": c.pause_prompt,
            "type": "pause" if c.paused_block_id is not None else "automatic",
        }
        for c in checkpoints
    ]

    if format == "markdown":
        return format_checkpoint_list_markdown(checkpoint_data, workflow_name or None)
    else:
        return {
            "checkpoints": checkpoint_data,
            "total": len(checkpoints),
        }


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        idempotentHint=True,
    )
)
async def get_checkpoint_info(
    checkpoint_id: Annotated[
        str,
        Field(
            description=(
                "Checkpoint token to retrieve information about (e.g., 'pause_abc123'). "
                "Use list_checkpoints() to see all available checkpoints."
            ),
            min_length=1,
            max_length=100,
        ),
    ],
    format: Annotated[  # noqa: A002
        Literal["json", "markdown"],
        Field(
            description=(
                "Response format:\n"
                "- 'json': Structured data with detailed checkpoint information\n"
                "- 'markdown': Human-readable formatted details with progress sections"
            )
        ),
    ] = "json",
    *,
    ctx: AppContextType,
) -> dict[str, Any] | str:
    """Get detailed information about a specific checkpoint.

    Useful for inspecting checkpoint state before resuming.

    Returns:
        JSON format: Dictionary with detailed checkpoint information
        Markdown format: Formatted string with sections and progress details
    """
    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context

    state = await app_ctx.checkpoint_store.load_checkpoint(checkpoint_id)
    if state is None:
        return format_checkpoint_not_found_error(checkpoint_id, format)

    # Calculate progress percentage
    total_blocks = sum(len(wave) for wave in state.execution_waves)
    if total_blocks > 0:
        progress_percentage = len(state.completed_blocks) / total_blocks * 100
    else:
        progress_percentage = 0

    info = {
        "found": True,
        "checkpoint_id": state.checkpoint_id,
        "workflow_name": state.workflow_name,
        "created_at": state.created_at,
        "created_at_iso": datetime.fromtimestamp(state.created_at).isoformat(),
        "is_paused": state.paused_block_id is not None,
        "paused_block_id": state.paused_block_id,
        "pause_prompt": state.pause_prompt,
        "completed_blocks": state.completed_blocks,
        "current_wave": state.current_wave_index,
        "total_waves": len(state.execution_waves),
        "progress_percentage": round(progress_percentage, 1),
    }

    if format == "markdown":
        return format_checkpoint_info_markdown(state)

    return info


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,  # Deletes checkpoint
        idempotentHint=True,  # Same result if called multiple times
    )
)
async def delete_checkpoint(
    checkpoint_id: Annotated[
        str,
        Field(
            description=(
                "Checkpoint token to delete (e.g., 'pause_abc123'). "
                "Use list_checkpoints() to see all available checkpoints. "
                "Useful for cleaning up paused workflows that are no longer needed."
            ),
            min_length=1,
            max_length=100,
        ),
    ],
    *,
    ctx: AppContextType,
) -> dict[str, Any]:
    """Delete a checkpoint.

    Useful for cleaning up paused workflows that are no longer needed.

    Returns:
        Deletion status
    """
    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context

    deleted = await app_ctx.checkpoint_store.delete_checkpoint(checkpoint_id)

    return {
        "deleted": deleted,
        "checkpoint_id": checkpoint_id,
        "message": "Checkpoint deleted successfully" if deleted else "Checkpoint not found",
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Tool functions (all MCP tools)
    "execute_workflow",
    "execute_inline_workflow",
    "list_workflows",
    "get_workflow_info",
    "get_workflow_schema",
    "validate_workflow_yaml",
    "resume_workflow",
    "list_checkpoints",
    "get_checkpoint_info",
    "delete_checkpoint",
]
