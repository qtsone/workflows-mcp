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
from .formatting import (
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
        title="Execute Workflow",
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
            description="Workflow name (use list_workflows() to discover)",
            min_length=1,
            max_length=200,
        ),
    ],
    inputs: Annotated[
        dict[str, Any] | None,
        Field(description="Runtime inputs for {{inputs.*}} substitution"),
    ] = None,
    debug: Annotated[
        bool,
        Field(description="Write debug log to /tmp/"),
    ] = False,
    mode: Annotated[
        Literal["sync", "async"],
        Field(description="sync=wait for result, async=return job_id"),
    ] = "sync",
    timeout: Annotated[
        int | None,
        Field(
            description="Timeout in seconds (async mode only)",
            ge=1,
            le=86400,
        ),
    ] = None,
    *,
    ctx: AppContextType,
) -> dict[str, Any]:
    """Run a workflow by name. Required: workflow. Optional: inputs, mode (sync|async), timeout."""
    # Validate context availability
    if ctx is None:
        return {
            "status": "failure",
            "error": "Server context not available. Tool requires context to access resources.",
        }

    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context

    # Handle async mode - submit to job queue and return immediately
    if mode == "async":
        if not app_ctx.job_queue:
            return {
                "status": "failure",
                "error": "Async execution not enabled",
                "message": (
                    "Job queue not available. Use mode='sync' or enable job queue "
                    "with WORKFLOWS_JOB_QUEUE_ENABLED=true."
                ),
            }

        # Submit job with optional timeout
        job_id = await app_ctx.job_queue.submit_job(workflow, inputs, timeout=timeout)
        # Get effective timeout for response
        effective_timeout = timeout if timeout else app_ctx.job_queue._default_job_timeout
        return {
            "job_id": job_id,
            "workflow": workflow,
            "status": "queued",
            "timeout": effective_timeout,
            "message": "Job submitted successfully. Use get_job_status() to check progress.",
        }

    # Synchronous mode - execute workflow and wait for completion
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
    runner = WorkflowRunner()
    result = await runner.execute(
        workflow=workflow_schema,
        runtime_inputs=inputs,
        context=exec_context,
        debug=debug,
    )

    # Handle paused workflows (unified Job architecture)
    if result.status == "paused":
        # Paused workflows require job_queue for resume
        if not app_ctx.job_queue:
            return {
                "status": "failure",
                "error": "Workflow paused but job queue not enabled",
                "message": (
                    "Interactive workflows (Prompt blocks) require job queue for pause/resume. "
                    "Enable with WORKFLOWS_JOB_QUEUE_ENABLED=true or use mode='async'."
                ),
            }

        # Create Job with PAUSED status for resume (unified architecture)
        from uuid import uuid4

        from .engine.job_queue import Job, WorkflowStatus

        # Generate unique job ID
        job_id = f"job_{uuid4().hex[:8]}"

        # Create Job with execution state embedded in result
        job = Job(
            id=job_id,
            workflow=workflow,
            inputs=inputs or {},
            status=WorkflowStatus.PAUSED,
            result=result._build_debug_data(),  # Contains execution_state for resume
            created_at=datetime.now(),
            started_at=datetime.now(),  # Started immediately in sync mode
        )

        # Save to JobStore for resume
        await app_ctx.job_queue._store.save_job(job)

        # Return response with job_id for resume
        response = result.to_response(debug)
        response["job_id"] = job_id
        response["message"] = (
            f"Workflow paused waiting for input. "
            f"Use resume_workflow(job_id='{job_id}', response='your_answer') to continue."
        )
        return response

    # Format response using ExecutionResult.to_response()
    return result.to_response(debug)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Execute Inline Workflow",
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
            description="Complete workflow YAML with name, description, and blocks",
            min_length=10,
            max_length=100000,
        ),
    ],
    inputs: Annotated[
        dict[str, Any] | None,
        Field(description="Runtime inputs for {{inputs.*}} variable substitution"),
    ] = None,
    debug: Annotated[
        bool,
        Field(description="Write execution log to /tmp/<workflow>-<timestamp>.json"),
    ] = False,
    *,
    ctx: AppContextType,
) -> dict[str, Any]:
    """Run inline YAML workflow. Required: workflow_yaml. Optional: inputs, debug."""
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
    runner = WorkflowRunner()
    result = await runner.execute(
        workflow=workflow_schema,
        runtime_inputs=inputs,
        context=exec_context,
        debug=debug,
    )

    # Format response using ExecutionResult.to_response()
    return result.to_response(debug)


@mcp.tool(
    annotations=ToolAnnotations(
        title="List Workflows",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def list_workflows(
    tags: Annotated[
        list[str],
        Field(
            description="Filter by tags (AND logic). Empty list returns all workflows.",
            max_length=20,
        ),
    ] = [],
    format: Annotated[  # noqa: A002
        Literal["json", "markdown"],
        Field(description="Output format"),
    ] = "json",
    *,
    ctx: AppContextType,
) -> str:
    """List workflow templates. Optional: tags (filter), format (json|markdown)."""
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
        title="Get Workflow Info",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def get_workflow_info(
    workflow: Annotated[
        str,
        Field(description="Workflow name to inspect", min_length=1, max_length=200),
    ],
    format: Annotated[  # noqa: A002
        Literal["json", "markdown"],
        Field(description="Output format"),
    ] = "json",
    *,
    ctx: AppContextType,
) -> dict[str, Any] | str:
    """Get workflow details (blocks, inputs, outputs). Required: workflow. Optional: format."""
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
        title="Get Workflow Schema",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def get_workflow_schema() -> dict[str, Any]:
    """Get JSON Schema for workflow YAML structure and block types. No parameters."""
    # Schema can be generated from executor registry without context
    from .engine.executor_base import create_default_registry

    # Create registry with all built-in executors and generate schema
    registry = create_default_registry()
    schema: dict[str, Any] = registry.generate_workflow_schema()
    return schema


@mcp.tool(
    annotations=ToolAnnotations(
        title="Validate Workflow YAML",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def validate_workflow_yaml(
    yaml_content: Annotated[
        str,
        Field(
            description="Complete workflow YAML to validate",
            min_length=10,
            max_length=100000,
        ),
    ],
) -> dict[str, Any]:
    """Validate workflow YAML without execution. Required: yaml_content."""
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
        title="Resume Workflow",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
async def resume_workflow(
    job_id: Annotated[
        str,
        Field(description="Job ID from paused workflow", min_length=1, max_length=100),
    ],
    response: Annotated[
        str,
        Field(description="Response to the pause prompt", max_length=10000),
    ] = "",
    debug: Annotated[
        bool,
        Field(description="Write execution log to /tmp/<workflow>-<timestamp>.json"),
    ] = False,
    *,
    ctx: AppContextType,
) -> dict[str, Any]:
    """Resume a paused workflow. Required: job_id. Optional: response, debug."""
    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context

    # Require job_queue for unified architecture
    if not app_ctx.job_queue:
        return {
            "status": "failure",
            "error": "Job queue not enabled",
            "message": (
                "Resume functionality requires job queue for unified pause/resume architecture. "
                "Enable with WORKFLOWS_JOB_QUEUE_ENABLED=true."
            ),
        }

    # Load Job from JobStore
    try:
        job_data = await app_ctx.job_queue._store.load_job(job_id)
    except KeyError:
        return {
            "status": "failure",
            "error": f"Job not found: {job_id}",
            "message": "Use list_jobs(status='paused') to see available paused workflows.",
        }

    # Validate job status
    from .engine.job_queue import Job, WorkflowStatus

    job = Job.model_validate(job_data)
    if job.status != WorkflowStatus.PAUSED:
        return {
            "status": "failure",
            "error": f"Job not paused: {job_id} (status={job.status.value})",
            "message": (
                "Only paused workflows can be resumed. Use get_job_status() to check status."
            ),
        }

    # Extract ExecutionState from Job.result
    if not job.result:
        return {
            "status": "failure",
            "error": f"Job missing result data: {job_id}",
            "message": "Paused job corrupted - cannot resume.",
        }

    # Use WorkflowRunner helper to extract execution state
    try:
        execution_state = WorkflowRunner._extract_execution_state(job.result)
    except ValueError as e:
        return {
            "status": "failure",
            "error": str(e),
            "message": "Failed to extract execution state from paused job.",
        }

    # Create execution context
    exec_context = app_ctx.create_execution_context()

    # Create WorkflowRunner and resume from state
    runner = WorkflowRunner()
    result = await runner.resume_from_state(
        execution_state=execution_state,
        response=response,
        context=exec_context,
    )

    # Update job based on result status
    from datetime import datetime

    if result.status == "success":
        # Workflow completed successfully
        job.status = WorkflowStatus.COMPLETED
        job.result = result._build_debug_data()
        job.completed_at = datetime.now()
        job.updated_at = datetime.now()
        await app_ctx.job_queue._store.save_job(job)
        await app_ctx.job_queue._store.increment_stat("completed_jobs")

    elif result.status == "failure":
        # Workflow failed during resume
        job.status = WorkflowStatus.FAILED
        job.result = result._build_debug_data()
        job.error = result.error
        job.completed_at = datetime.now()
        job.updated_at = datetime.now()
        await app_ctx.job_queue._store.save_job(job)
        await app_ctx.job_queue._store.increment_stat("failed_jobs")

    elif result.status == "paused":
        # Workflow paused again - update job with new execution state
        job.result = result._build_debug_data()
        job.updated_at = datetime.now()
        await app_ctx.job_queue._store.save_job(job)

        # Return response with same job_id
        response_dict = result.to_response(debug)
        response_dict["job_id"] = job_id
        response_dict["message"] = (
            f"Workflow paused again. Use resume_workflow(job_id='{job_id}') to continue."
        )
        return response_dict

    # Format response using ExecutionResult.to_response()
    return result.to_response(debug)


# =============================================================================
# Async Execution Tools (Job Queue)
# =============================================================================


@mcp.tool(
    annotations=ToolAnnotations(
        title="Get Job Status",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def get_job_status(
    job_id: str,
    *,
    ctx: AppContextType,
) -> dict[str, Any]:
    """Get job status and outputs. Required: job_id."""
    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context

    # Check if job queue is available
    if not app_ctx.job_queue:
        return {
            "error": "Job queue not available",
            "message": "Async execution is not enabled",
        }

    # Get job status
    try:
        return await app_ctx.job_queue.get_status(job_id)
    except KeyError:
        return {
            "error": "Job not found",
            "job_id": job_id,
            "message": f"No job found with ID: {job_id}",
        }


@mcp.tool(
    annotations=ToolAnnotations(
        title="Cancel Job",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def cancel_job(
    job_id: str,
    *,
    ctx: AppContextType,
) -> dict[str, Any]:
    """Cancel a pending or running job. Required: job_id."""
    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context

    # Check if job queue is available
    if not app_ctx.job_queue:
        return {
            "error": "Job queue not available",
            "message": "Async execution is not enabled",
        }

    # Cancel job
    try:
        cancelled = await app_ctx.job_queue.cancel_job(job_id)
        return {
            "job_id": job_id,
            "cancelled": cancelled,
            "message": (
                "Job cancelled successfully" if cancelled else "Job already completed or failed"
            ),
        }
    except KeyError:
        return {
            "error": "Job not found",
            "job_id": job_id,
            "message": f"No job found with ID: {job_id}",
        }


@mcp.tool(
    annotations=ToolAnnotations(
        title="List Jobs",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def list_jobs(
    status: str | None = None,
    limit: int = 100,
    *,
    ctx: AppContextType,
) -> dict[str, Any]:
    """List workflow jobs. Optional: status (filter), limit (default 100)."""
    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context

    # Check if job queue is available
    if not app_ctx.job_queue:
        return {
            "error": "Job queue not available",
            "message": "Async execution is not enabled",
            "jobs": [],
            "total": 0,
        }

    # Parse status filter
    from .engine.job_queue import WorkflowStatus

    status_filter = None
    if status:
        try:
            status_filter = WorkflowStatus(status.lower())
        except ValueError:
            return {
                "error": "Invalid status",
                "message": f"Invalid status: {status}. "
                f"Valid values: queued, running, paused, completed, failed, cancelled",
                "jobs": [],
            }

    # List jobs
    jobs = await app_ctx.job_queue.list_jobs(status=status_filter, limit=limit)

    # Get total from stats (now async)
    stats = await app_ctx.job_queue.get_stats()

    return {
        "jobs": jobs,
        "total": stats.get("total_jobs", 0),
        "filtered": len(jobs),
    }


@mcp.tool(
    annotations=ToolAnnotations(
        title="Get Queue Statistics",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def get_queue_stats(
    *,
    ctx: AppContextType,
) -> dict[str, Any]:
    """Get queue statistics (IO and job queues). No parameters."""
    app_ctx = ctx.request_context.lifespan_context

    stats: dict[str, Any] = {}

    if app_ctx.io_queue:
        stats["io_queue"] = app_ctx.io_queue.get_stats()

    if app_ctx.job_queue:
        stats["job_queue"] = await app_ctx.job_queue.get_stats()

    if not stats:
        return {
            "error": "No queues enabled",
            "message": "Both IO queue and Job queue are disabled",
        }

    return stats


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
    # Async execution tools
    "get_job_status",
    "cancel_job",
    "list_jobs",
    "get_queue_stats",
]
