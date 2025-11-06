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
                "Runtime inputs for variable substitution ({{inputs.*}} in workflow). "
                "Examples: {'project_path': './app'}, {'branch': 'main', 'force': True}, "
                "{'python_version': '3.12'}. Optional: get_workflow_info() shows expected inputs."
            )
        ),
    ] = None,
    debug: Annotated[
        bool,
        Field(
            description=(
                "Enable debug logging (default: False). Writes detailed execution log to "
                "/tmp/<workflow>-<timestamp>.json with block details, variable resolution, "
                "DAG waves. Use for: debugging failures, analyzing execution order."
            )
        ),
    ] = False,
    *,
    ctx: AppContextType,
) -> dict[str, Any]:
    """Execute a registered workflow template.

    Runs pre-registered workflows from the template library. Workflows execute as DAGs
    with parallel execution of independent blocks. Supports shell commands, git operations,
    file operations, HTTP calls, LLM calls, and workflow composition.

    Use execute_workflow for: Pre-built templates used repeatedly
    Use execute_inline_workflow for: Ad-hoc YAML definitions, testing, one-off tasks

    Returns:
        {
            "status": "success" | "failure" | "paused",
            "outputs": {...},        # Workflow outputs (if defined)
            "error": "...",          # Error details (if failed)
            "checkpoint_id": "...",  # Resume token (if paused)
            "prompt": "..."          # Prompt text (if paused)
        }

    Tool Call Examples:
        execute_workflow(workflow="python-ci-pipeline", inputs={"project_path": "./app"})
        execute_workflow(workflow="git-checkout-branch", inputs={"branch": "main"})
        execute_workflow(workflow="node-build", inputs={"workspace": "./frontend"}, debug=True)
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
    return result.to_response(debug)


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
                "Complete workflow YAML definition. Must include: name, description, blocks. "
                "Recommend: validate_workflow_yaml() first to catch errors. "
                "Quote {{...}} variables: use 'value: \"{{var}}\"' or block scalars (| or >)."
            ),
            min_length=10,
            max_length=100000,
        ),
    ],
    inputs: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Runtime inputs for variable substitution ({{inputs.*}} in YAML). "
                "Examples: {'source': 'src/'}, {'dir': '/tmp', 'force': True}."
            )
        ),
    ] = None,
    debug: Annotated[
        bool,
        Field(
            description=(
                "Enable debug logging (default: False). Writes detailed execution log to "
                "/tmp/<workflow>-<timestamp>.json with block details, variable resolution, "
                "DAG waves. Use for: debugging failures, analyzing execution order."
            )
        ),
    ] = False,
    *,
    ctx: AppContextType,
) -> dict[str, Any]:
    """Execute a workflow from YAML string without registering it.

        Enables ad-hoc workflow execution without file system modifications. Useful for testing
        workflow definitions, one-off tasks, or dynamically generated workflows.

        Use execute_inline_workflow for: Testing, prototyping, LLM-generated workflows
        Use execute_workflow for: Pre-built templates, production workflows

        Returns:
            Same structure as execute_workflow (status, outputs, error, checkpoint_id, prompt).

        Tool Call Example:
            execute_inline_workflow(
                workflow_yaml='''
    name: test-workflow
    description: Test workflow
    blocks:
      - id: echo
        type: Shell
        inputs:
          command: echo "{{inputs.message}}"
                ''',
                inputs={"message": "Hello"}
            )
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
    return result.to_response(debug)


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
                "Filter by tags (AND logic - all tags must match). "
                "Examples: ['git'], ['python', 'ci'], ['node', 'test']. "
                "Empty list returns all workflows."
            ),
            max_length=20,
        ),
    ] = [],
    format: Annotated[  # noqa: A002
        Literal["json", "markdown"],
        Field(
            description=(
                "Response format: 'json' (workflow names array) or "
                "'markdown' (formatted list with headers)."
            )
        ),
    ] = "json",
    *,
    ctx: AppContextType,
) -> str:
    """List available workflow templates, optionally filtered by tags.

    Discover workflows by name or filter by domain-specific tags (git, python, node, ci, test).
    Returns workflow names only - use get_workflow_info() for detailed information.

    Returns:
        JSON: Array of workflow names ["workflow1", "workflow2"]
        Markdown: Formatted list with headers and descriptions

    Tool Call Examples:
        list_workflows()  # All workflows
        list_workflows(tags=["git"])  # Git workflows only
        list_workflows(tags=["python", "ci"], format="markdown")  # Python CI, formatted
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
                "Workflow name to inspect. Examples: 'python-ci-pipeline', 'git-checkout-branch'."
            ),
            min_length=1,
            max_length=200,
        ),
    ],
    format: Annotated[  # noqa: A002
        Literal["json", "markdown"],
        Field(
            description=(
                "Response format: 'json' (structured metadata) or "
                "'markdown' (formatted description)."
            )
        ),
    ] = "json",
    *,
    ctx: AppContextType,
) -> dict[str, Any] | str:
    """Get detailed information about a workflow template.

    Returns metadata including: description, block structure, required/optional inputs,
    output definitions, and dependencies. Useful for understanding workflow requirements
    before execution.

    Returns:
        JSON: {name, description, blocks: [{id, type, depends_on}], inputs, outputs}
        Markdown: Formatted sections with block details and input/output schemas

    Tool Call Examples:
        get_workflow_info(workflow="python-ci-pipeline")
        get_workflow_info(workflow="git-checkout-branch", format="markdown")
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
    """Get JSON Schema for workflow validation and generation.

    Returns complete schema describing workflow YAML structure, including all registered
    block types (Shell, Workflow, CreateFile, etc.) and their input schemas.

    Use for: Generating workflows, validating YAML, discovering block types and parameters.

    Returns:
        JSON Schema with workflow structure, block type definitions, and parameter schemas

    Tool Call Example:
        get_workflow_schema()  # No parameters required
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
                "Complete workflow YAML to validate. Must include: name, description, blocks. "
                "Use before execute_inline_workflow to catch syntax/schema errors."
            ),
            min_length=10,
            max_length=100000,
        ),
    ],
) -> dict[str, Any]:
    """Validate workflow YAML without execution.

        Checks: YAML syntax, required fields (name, description, blocks), block type existence,
        and schema compliance. Use before execute_inline_workflow to catch errors early.

        Returns:
            {
                "valid": bool,               # True if validation passes
                "errors": list[str],         # Validation errors (empty if valid)
                "warnings": list[str],       # Non-blocking warnings
                "block_types_used": list[str]  # Block types found in workflow
            }

            Common errors: Invalid YAML syntax, missing required fields, unknown block types,
            incorrect indentation. Errors include actionable fix suggestions.

        Tool Call Example:
            validate_workflow_yaml(yaml_content='''
    name: test
    description: Test workflow
    blocks:
      - id: test
        type: Shell
        inputs:
          command: echo "test"
            ''')
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
                "Checkpoint ID from paused workflow. Get from: execute_workflow response "
                "(if paused) or list_checkpoints(). Format: 'pause_abc123'."
            ),
            min_length=1,
            max_length=100,
        ),
    ],
    response: Annotated[
        str,
        Field(
            description=(
                "Response to pause prompt. For Prompt blocks, provide text based on the prompt. "
                "Workflow continues execution using this value in subsequent blocks or conditions."
            ),
            max_length=10000,
        ),
    ] = "",
    debug: Annotated[
        bool,
        Field(
            description=(
                "Enable debug logging (default: False). Writes detailed execution log to "
                "/tmp/<workflow>-<timestamp>.json with block details, variable resolution, "
                "DAG waves. Use for: debugging failures, analyzing execution order."
            )
        ),
    ] = False,
    *,
    ctx: AppContextType,
) -> dict[str, Any]:
    """Resume a paused workflow from checkpoint.

    Continues workflow execution from where it paused (Prompt blocks pause for LLM input).
    Use when execute_workflow returns status="paused" with checkpoint_id.

    Returns:
        Same structure as execute_workflow (status, outputs, error, checkpoint_id, prompt)

    Tool Call Examples:
        resume_workflow(checkpoint_id="pause_abc123", response="yes")
        resume_workflow(checkpoint_id="pause_xyz789", response="proceed with deployment")
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
    return result.to_response(debug)


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
                "Filter by workflow name. Examples: 'python-ci-pipeline', ''. "
                "Empty string returns all checkpoints."
            ),
            max_length=200,
        ),
    ] = "",
    format: Annotated[  # noqa: A002
        Literal["json", "markdown"],
        Field(description=("Response format: 'json' (structured list) or 'markdown' (formatted).")),
    ] = "json",
    *,
    ctx: AppContextType,
) -> dict[str, Any] | str:
    """List workflow checkpoints.

    Shows paused workflows awaiting resume. Each checkpoint includes: ID, workflow name,
    creation time, pause prompt, type (pause vs automatic).

    Returns:
        JSON: {checkpoints: [{checkpoint_id, workflow, created_at, ...}], total: N}
        Markdown: Formatted table with checkpoint details

    Tool Call Examples:
        list_checkpoints()  # All checkpoints
        list_checkpoints(workflow_name="python-ci-pipeline")  # Filter by workflow
        list_checkpoints(format="markdown")  # Formatted output
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
                "Checkpoint ID to inspect. Get from list_checkpoints() or execute response."
            ),
            min_length=1,
            max_length=100,
        ),
    ],
    format: Annotated[  # noqa: A002
        Literal["json", "markdown"],
        Field(
            description=("Response format: 'json' (detailed metadata) or 'markdown' (formatted).")
        ),
    ] = "json",
    *,
    ctx: AppContextType,
) -> dict[str, Any] | str:
    """Get checkpoint details.

    Returns detailed checkpoint state including: workflow name, creation time, pause prompt,
    paused block ID, completed blocks, execution progress percentage.

    Returns:
        JSON: {checkpoint_id, workflow_name, paused_block_id, progress_percentage, ...}
        Markdown: Formatted sections with progress details

    Tool Call Examples:
        get_checkpoint_info(checkpoint_id="pause_abc123")
        get_checkpoint_info(checkpoint_id="pause_abc123", format="markdown")
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
                "Checkpoint ID to delete. Get from list_checkpoints(). "
                "Use for cleaning up abandoned paused workflows."
            ),
            min_length=1,
            max_length=100,
        ),
    ],
    *,
    ctx: AppContextType,
) -> dict[str, Any]:
    """Delete a checkpoint.

    Removes checkpoint permanently. Use for cleaning up paused workflows that won't be resumed.
    Cannot be undone.

    Returns:
        {deleted: bool, checkpoint_id: str, message: str}

    Tool Call Example:
        delete_checkpoint(checkpoint_id="pause_abc123")
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
