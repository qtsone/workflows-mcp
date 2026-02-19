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

from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

from .context import AppContextType
from .engine import WorkflowRunner, load_workflow_from_yaml
from .formatting import (
    format_workflow_info_markdown,
    format_workflow_list_markdown,
    format_workflow_not_found_error,
)
from .server import load_workflows, mcp

# =============================================================================
# Response Helpers
# =============================================================================


def _json_response(data: dict[str, Any]) -> CallToolResult:
    """Build a CallToolResult with both compact text and structured content.

    Returns a CallToolResult that the SDK passes through unchanged:
    - content: compact JSON string in TextContent (preserves TASK-057 fix)
    - structuredContent: raw dict for clients that support it (TASK-062)

    This avoids the SDK's default indent=2 serialization while also providing
    parsed JSON objects via structuredContent for modern MCP clients.
    """
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(data, separators=(",", ":")))],
        structuredContent=data,
    )


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
            description=(
                "Name of the workflow to execute. Use list_workflows to discover available options"
            ),
            examples=["build-project", "deploy-app"],
        ),
    ],
    inputs: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Key-value pairs passed to the workflow, accessible via {{inputs.key}} in templates"
            ),
            default=None,
        ),
    ],
    debug: Annotated[
        bool,
        Field(
            description=(
                "Write detailed execution trace to "
                "/tmp/<workflow>-<timestamp>.json for troubleshooting"
            ),
            default=False,
        ),
    ],
    mode: Annotated[
        Literal["sync", "async"],
        Field(
            description=(
                "Execution mode: 'sync' waits for completion, "
                "'async' returns a job_id immediately for tracking"
            ),
            default="sync",
            examples=["sync", "async"],
        ),
    ],
    timeout: Annotated[
        int | None,
        Field(
            description=(
                "Maximum execution time in seconds for async jobs. Only applies to async mode"
            ),
            ge=1,
            le=86400,
            default=None,
        ),
    ],
    *,
    ctx: AppContextType,
) -> CallToolResult:
    """
    Run a registered workflow by name to automate a task.

    WHEN TO USE: After calling list_workflows to find available workflow names.

    PARAMETERS:
    - workflow: The exact workflow name (e.g., "build-project", "deploy-app")
    - inputs: Key-value pairs the workflow needs (e.g., {"branch": "main"})
    - mode: "sync" waits for completion, "async" returns job_id immediately
    - debug: Set True to write execution trace to /tmp for troubleshooting

    RETURNS: {status: "success"|"failure"|"paused", outputs: {...}, ...}

    SEE ALSO: list_workflows (discover names), get_workflow_info (see required inputs)
    """
    # Validate context availability
    if ctx is None:
        return _json_response(
            {
                "status": "failure",
                "error": "Server context not available. Tool requires context to access resources.",
            }
        )

    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context

    # Handle async mode - submit to job queue and return immediately
    if mode == "async":
        if not app_ctx.job_queue:
            return _json_response(
                {
                    "status": "failure",
                    "error": "Async execution not enabled",
                    "message": (
                        "Job queue not available. Use mode='sync' or enable job queue "
                        "with WORKFLOWS_JOB_QUEUE_ENABLED=true."
                    ),
                }
            )

        # Submit job with optional timeout
        job_id = await app_ctx.job_queue.submit_job(workflow, inputs, timeout=timeout)
        # Get effective timeout for response
        effective_timeout = timeout if timeout else app_ctx.job_queue._default_job_timeout
        return _json_response(
            {
                "job_id": job_id,
                "workflow": workflow,
                "status": "queued",
                "timeout": effective_timeout,
                "message": (
                    f"Job submitted successfully. "
                    f"Use get_job_status(job_id='{job_id}') to check progress."
                ),
            }
        )

    # Synchronous mode - execute workflow and wait for completion
    registry = app_ctx.registry

    # Validate workflow exists
    if workflow not in registry:
        available = registry.list_names()
        return _json_response(
            {
                "status": "failure",
                "error": (
                    f"Workflow '{workflow}' not found. "
                    f"Available workflows: {', '.join(available[:5])}"
                    f"{' (and more)' if len(available) > 5 else ''}. "
                    "Use list_workflows() to see all workflows or filter by tags."
                ),
                "available_workflows": available,
            }
        )

    # Get workflow schema
    workflow_schema = registry.get(workflow)
    if workflow_schema is None:
        return _json_response(
            {
                "status": "failure",
                "error": f"Failed to load workflow '{workflow}' from registry.",
            }
        )

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
            return _json_response(
                {
                    "status": "failure",
                    "error": "Workflow paused but job queue not enabled",
                    "message": (
                        "Interactive workflows (Prompt blocks) require job queue for pause/resume. "
                        "Enable with WORKFLOWS_JOB_QUEUE_ENABLED=true or use mode='async'."
                    ),
                }
            )

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
        return _json_response(response)

    # Format response using ExecutionResult.to_response()
    return _json_response(result.to_response(debug))


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
            description=(
                "Complete workflow definition in YAML including name, description, and blocks array"
            ),
        ),
    ],
    inputs: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Key-value pairs passed to the workflow, accessible via {{inputs.key}} in templates"
            ),
            default=None,
        ),
    ],
    debug: Annotated[
        bool,
        Field(
            description=(
                "Write detailed execution trace to "
                "/tmp/<workflow>-<timestamp>.json for troubleshooting"
            ),
            default=False,
        ),
    ],
    *,
    ctx: AppContextType,
) -> CallToolResult:
    """
    Run a workflow directly from YAML without registering it.

    WHEN TO USE: For testing new workflow definitions or one-off automation.
    Call validate_workflow_yaml first to check for syntax errors.

    PARAMETERS:
    - workflow_yaml: Complete YAML with name, description, and blocks
    - inputs: Runtime values accessible via {{inputs.key}} in templates
    - debug: Set True to write execution trace for troubleshooting

    EXAMPLE workflow_yaml:
        name: my-workflow
        description: Example workflow
        blocks:
          - id: step1
            type: Shell
            inputs:
              command: echo "Hello"

    SEE ALSO: validate_workflow_yaml, get_workflow_schema
    """
    # Validate context availability
    if ctx is None:
        return _json_response(
            {
                "status": "failure",
                "error": "Server context not available. Tool requires context to access resources.",
            }
        )

    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context

    # Parse YAML string to WorkflowSchema
    load_result = load_workflow_from_yaml(workflow_yaml, source="<inline-workflow>")

    if not load_result.is_success:
        return _json_response(
            {
                "status": "failure",
                "error": (
                    f"Failed to parse workflow YAML: {load_result.error}. "
                    "Ensure your YAML is valid and includes required fields: "
                    "'name', 'description', and 'blocks'. "
                    "Use validate_workflow_yaml() to check YAML syntax before execution."
                ),
            }
        )

    workflow_schema = load_result.value
    if workflow_schema is None:
        return _json_response(
            {
                "status": "failure",
                "error": (
                    "Workflow definition parsing returned None. "
                    "The YAML structure may be invalid. "
                    "Required fields: 'name' (string), 'description' (string), "
                    "'blocks' (list of block definitions). "
                    "Use validate_workflow_yaml() to validate your workflow YAML."
                ),
            }
        )

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
    return _json_response(result.to_response(debug))


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
            description=(
                "Filter by tags using AND logic. Example: ['build', 'ci']. Empty returns all"
            ),
            default_factory=list,
            examples=[["build"], ["ci", "deploy"]],
        ),
    ],
    format: Annotated[  # noqa: A002
        Literal["json", "markdown"],
        Field(
            description=(
                "Return format: 'json' for programmatic use, 'markdown' for human-readable display"
            ),
            default="json",
            examples=["json", "markdown"],
        ),
    ],
    *,
    ctx: AppContextType,
) -> CallToolResult:
    """
    Discover available workflows in the registry.

    WHEN TO USE: Call this FIRST before execute_workflow to find valid workflow names.

    PARAMETERS:
    - tags: Filter by tags (e.g., ["build", "ci"]). Empty list returns all.
    - format: "json" for programmatic use, "markdown" for display

    RETURNS: List of workflow names like ["build-project", "deploy-app", ...]

    SEE ALSO: get_workflow_info (details about a specific workflow)
    """
    # Validate context availability
    if ctx is None:
        return _json_response({"status": "failure", "error": "Server context not available"})

    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context
    registry = app_ctx.registry

    workflows = registry.list_names(tags=tags or [])

    if format == "markdown":
        return CallToolResult(
            content=[
                TextContent(
                    type="text", text=format_workflow_list_markdown(workflows, tags or None)
                )
            ],
        )
    else:
        # Return compact JSON array for programmatic access
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(workflows, separators=(",", ":")))],
            structuredContent={"workflows": workflows},
        )


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
        Field(
            description="Name of the workflow to inspect",
            examples=["build-project", "deploy-app"],
        ),
    ],
    format: Annotated[  # noqa: A002
        Literal["json", "markdown"],
        Field(
            description=(
                "Return format: 'json' for structured data, 'markdown' for readable documentation"
            ),
            default="json",
            examples=["json", "markdown"],
        ),
    ],
    *,
    ctx: AppContextType,
) -> CallToolResult:
    """
    Inspect a workflow's structure including required inputs and block sequence.

    WHEN TO USE: After list_workflows, before execute_workflow - to understand
    what inputs a workflow expects and what outputs it produces.

    PARAMETERS:
    - workflow: Exact workflow name from list_workflows
    - format: "json" for parsing, "markdown" for readable documentation

    RETURNS: {name, description, inputs: {...}, outputs: {...}, blocks: [...]}

    SEE ALSO: list_workflows (find names), execute_workflow (run it)
    """
    # Validate context availability
    if ctx is None:
        return _json_response({"status": "failure", "error": "Server context not available"})

    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context
    registry = app_ctx.registry

    # Check if workflow exists
    if workflow not in registry:
        error_result = format_workflow_not_found_error(workflow, registry.list_names(), format)
        # Wrap dict result (for json format) with _json_response
        if isinstance(error_result, dict):
            return _json_response(error_result)
        return CallToolResult(
            content=[TextContent(type="text", text=error_result)],
        )

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

        # Add output mappings if available - convert to JSON-serializable format
        if workflow_def.outputs:
            info["outputs"] = {
                name: schema.model_dump() if hasattr(schema, "model_dump") else schema
                for name, schema in workflow_def.outputs.items()
            }

    # Format as markdown if requested
    if format == "markdown":
        return CallToolResult(
            content=[TextContent(type="text", text=format_workflow_info_markdown(info))],
        )

    return _json_response(info)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Get Workflow Schema",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def get_workflow_schema(
    *,
    ctx: AppContextType,
) -> CallToolResult:
    """
    Get the complete JSON Schema for workflow YAML authoring.

    WHEN TO USE: When writing new workflow YAML to understand:
    - Available block types: Shell, LLM, SQL, ReadFiles, EditFile, Workflow, etc.
    - Required fields for each block type
    - Valid workflow structure

    RETURNS: JSON Schema object with $schema, properties, definitions

    SEE ALSO: validate_workflow_yaml (check your YAML), execute_inline_workflow (run it)
    """
    # Validate context availability
    if ctx is None:
        return _json_response({"status": "failure", "error": "Server context not available"})

    # Use executor registry from lifespan context (efficient, no recreation)
    app_ctx = ctx.request_context.lifespan_context
    registry = app_ctx.executor_registry
    schema: dict[str, Any] = registry.generate_workflow_schema()
    return _json_response(schema)


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
            description=(
                "Workflow YAML to validate. Must include name, description, and blocks fields"
            ),
        ),
    ],
    *,
    ctx: AppContextType,
) -> CallToolResult:
    """
    Check workflow YAML for syntax errors and unknown block types.

    WHEN TO USE: BEFORE execute_inline_workflow to catch errors early.

    PARAMETERS:
    - yaml_content: Complete workflow YAML string

    RETURNS: {valid: true|false, errors: [...], warnings: [...], block_types_used: [...]}

    SEE ALSO: get_workflow_schema (see valid syntax), execute_inline_workflow (run it)
    """
    # Validate context availability
    if ctx is None:
        return _json_response({"status": "failure", "error": "Server context not available"})

    # Parse workflow YAML
    load_result = load_workflow_from_yaml(yaml_content, source="<validation>")

    if not load_result.is_success:
        return _json_response(
            {
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
        )

    workflow_def = load_result.value
    if workflow_def is None:
        return _json_response(
            {
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
        )

    # Extract block types used
    block_types_used = list({block.type for block in workflow_def.blocks})

    # Use executor registry from lifespan context (efficient, no recreation)
    app_ctx = ctx.request_context.lifespan_context
    registry = app_ctx.executor_registry

    errors: list[str] = []
    warnings: list[str] = []

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
    return _json_response(
        {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "block_types_used": block_types_used,
        }
    )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Reload Workflows",
        readOnlyHint=False,  # Modifies server state (registry)
        destructiveHint=False,
        idempotentHint=True,  # Repeated calls have same effect
        openWorldHint=True,  # Reads from filesystem
    )
)
async def reload_workflows(
    *,
    ctx: AppContextType,
) -> CallToolResult:
    """
    Refresh the workflow registry from disk to pick up file changes.

    WHEN TO USE: After modifying workflow YAML files on disk.
    The registry caches workflows at startup, so call this to reload.

    RETURNS: {status: "success", total: <number of workflows loaded>}

    SEE ALSO: list_workflows (see what's loaded)
    """
    # Access shared resources from lifespan context
    if ctx is None:
        return _json_response(
            {
                "status": "failure",
                "message": "Server context not available.",
            }
        )

    app_ctx = ctx.request_context.lifespan_context
    registry = app_ctx.registry

    try:
        load_workflows(registry)
    except Exception as e:
        return _json_response(
            {
                "status": "failure",
                "message": f"Failed to reload workflows: {str(e)}",
            }
        )

    return _json_response(
        {
            "status": "success",
            "message": "Successfully reloaded workflows",
            "total": len(registry.list_names()),
        }
    )


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
        Field(description="ID of the paused job (returned when workflow pauses)"),
    ],
    response: Annotated[
        str,
        Field(
            description="Your answer to the question the Prompt block displayed",
            default="",
        ),
    ],
    debug: Annotated[
        bool,
        Field(
            description=(
                "Write execution trace to /tmp/<workflow>-<timestamp>.json for troubleshooting"
            ),
            default=False,
        ),
    ],
    *,
    ctx: AppContextType,
) -> CallToolResult:
    """
    Continue a workflow that paused waiting for user input (Prompt block).

    WHEN TO USE: After execute_workflow returns status="paused" with a job_id.
    The workflow paused because a Prompt block needs your answer.

    PARAMETERS:
    - job_id: The job_id returned when workflow paused (e.g., "job_a1b2c3d4")
    - response: Your answer to the prompt question
    - debug: Set True for execution trace

    RETURNS: Workflow continues and returns final {status, outputs, ...}

    SEE ALSO: list_jobs (find paused jobs), get_job_status (check job state)
    """
    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context

    # Require job_queue for unified architecture
    if not app_ctx.job_queue:
        return _json_response(
            {
                "status": "failure",
                "error": "Job queue not enabled",
                "message": (
                    "Resume functionality requires job queue for unified pause/resume architecture."
                    "Enable with WORKFLOWS_JOB_QUEUE_ENABLED=true."
                ),
            }
        )

    # Load Job from JobStore
    try:
        job_data = await app_ctx.job_queue._store.load_job(job_id)
    except KeyError:
        return _json_response(
            {
                "status": "failure",
                "error": f"Job not found: {job_id}",
                "message": "Use list_jobs(status='paused') to see available paused workflows.",
            }
        )

    # Validate job status
    from .engine.job_queue import Job, WorkflowStatus

    job = Job.model_validate(job_data)
    if job.status != WorkflowStatus.PAUSED:
        return _json_response(
            {
                "status": "failure",
                "error": f"Job not paused: {job_id} (status={job.status.value})",
                "message": (
                    "Only paused workflows can be resumed. Use get_job_status() to check status."
                ),
            }
        )

    # Extract ExecutionState from Job.result
    if not job.result:
        return _json_response(
            {
                "status": "failure",
                "error": f"Job missing result data: {job_id}",
                "message": "Paused job corrupted - cannot resume.",
            }
        )

    # Use WorkflowRunner helper to extract execution state
    try:
        execution_state = WorkflowRunner._extract_execution_state(job.result)
    except ValueError as e:
        return _json_response(
            {
                "status": "failure",
                "error": str(e),
                "message": "Failed to extract execution state from paused job.",
            }
        )

    # Extract workflow_stack from execution state for proper depth tracking on resume
    # workflow_stack is saved as [{"name": "wf1"}, {"name": "wf2"}] format
    saved_stack = execution_state.workflow_stack or []
    workflow_stack = [item["name"] if isinstance(item, dict) else item for item in saved_stack]

    # Create execution context with restored workflow_stack
    exec_context = app_ctx.create_execution_context(workflow_stack=workflow_stack)

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
            f"Workflow paused again. "
            f"Use resume_workflow(job_id='{job_id}', response='your_answer') to continue."
        )
        return _json_response(response_dict)

    # Format response using ExecutionResult.to_response()
    return _json_response(result.to_response(debug))


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
    job_id: Annotated[
        str,
        Field(description="Job ID returned from execute_workflow in async mode"),
    ],
    *,
    ctx: AppContextType,
) -> CallToolResult:
    """
    Check the status and results of an async workflow job.

    WHEN TO USE: After execute_workflow with mode="async" returns a job_id.
    Poll this to track progress until status is "completed" or "failed".

    PARAMETERS:
    - job_id: Job ID from execute_workflow async response

    RETURNS: {status: "queued"|"running"|"paused"|"completed"|"failed", outputs: {...}, error: ...}

    SEE ALSO: list_jobs (see all jobs), cancel_job (stop a job)
    """
    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context

    # Check if job queue is available
    if not app_ctx.job_queue:
        return _json_response(
            {
                "status": "failure",
                "error": "Job queue not available",
                "message": "Async execution is not enabled",
            }
        )

    # Get job status
    try:
        result = await app_ctx.job_queue.get_status(job_id)
        return _json_response(result)
    except KeyError:
        return _json_response(
            {
                "status": "failure",
                "error": "Job not found",
                "job_id": job_id,
                "message": f"No job found with ID: {job_id}",
            }
        )


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
    job_id: Annotated[
        str,
        Field(description="ID of the job to cancel"),
    ],
    *,
    ctx: AppContextType,
) -> CallToolResult:
    """
    Stop an async job that is queued or running.

    WHEN TO USE: To abort a long-running or stuck workflow.
    Note: Cancelled jobs cannot be resumed.

    PARAMETERS:
    - job_id: Job ID to cancel

    RETURNS: {cancelled: true|false, message: ...}

    SEE ALSO: list_jobs (find job IDs), get_job_status (check before cancelling)
    """
    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context

    # Check if job queue is available
    if not app_ctx.job_queue:
        return _json_response(
            {
                "status": "failure",
                "error": "Job queue not available",
                "message": "Async execution is not enabled",
            }
        )

    # Cancel job
    try:
        cancelled = await app_ctx.job_queue.cancel_job(job_id)
        return _json_response(
            {
                "job_id": job_id,
                "cancelled": cancelled,
                "message": (
                    "Job cancelled successfully" if cancelled else "Job already completed or failed"
                ),
            }
        )
    except KeyError:
        return _json_response(
            {
                "status": "failure",
                "error": "Job not found",
                "job_id": job_id,
                "message": f"No job found with ID: {job_id}",
            }
        )


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
    status: Annotated[
        Literal["queued", "running", "paused", "completed", "failed", "cancelled"] | None,
        Field(
            description="Filter by status: queued, running, paused, completed, failed, cancelled",
            default=None,
            examples=["paused", "completed"],
        ),
    ],
    limit: Annotated[
        int,
        Field(
            description="Maximum number of jobs to return",
            ge=1,
            le=1000,
            default=100,
        ),
    ],
    *,
    ctx: AppContextType,
) -> CallToolResult:
    """
    List async workflow jobs with optional status filter.

    WHEN TO USE: To find jobs by status, especially "paused" jobs waiting for input.

    PARAMETERS:
    - status: Filter by "queued", "running", "paused", "completed", "failed", "cancelled"
    - limit: Maximum jobs to return (default 100)

    RETURNS: {jobs: [...], total: <count>, filtered: <count>}

    SEE ALSO: get_job_status (details on one job), resume_workflow (continue paused jobs)
    """
    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context

    # Check if job queue is available
    if not app_ctx.job_queue:
        return _json_response(
            {
                "status": "failure",
                "error": "Job queue not available",
                "message": "Async execution is not enabled",
                "jobs": [],
                "total": 0,
            }
        )

    # Parse status filter
    from .engine.job_queue import WorkflowStatus

    status_filter = None
    if status:
        try:
            status_filter = WorkflowStatus(status.lower())
        except ValueError:
            return _json_response(
                {
                    "status": "failure",
                    "error": "Invalid status",
                    "message": f"Invalid status: {status}. "
                    f"Valid values: queued, running, paused, completed, failed, cancelled",
                    "jobs": [],
                }
            )

    # List jobs
    jobs = await app_ctx.job_queue.list_jobs(status=status_filter, limit=limit)

    # Get total from stats (now async)
    stats = await app_ctx.job_queue.get_stats()

    return _json_response(
        {
            "jobs": jobs,
            "total": stats.get("total_jobs", 0),
            "filtered": len(jobs),
        }
    )


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
) -> CallToolResult:
    """
    Get metrics about job and IO queue health.

    WHEN TO USE: For monitoring system health and capacity.

    RETURNS: {job_queue: {total_jobs, queued, running, ...}, io_queue: {...}}

    SEE ALSO: list_jobs (see actual jobs)
    """
    app_ctx = ctx.request_context.lifespan_context

    stats: dict[str, Any] = {}

    if app_ctx.io_queue:
        stats["io_queue"] = app_ctx.io_queue.get_stats()

    if app_ctx.job_queue:
        stats["job_queue"] = await app_ctx.job_queue.get_stats()

    if not stats:
        return _json_response(
            {
                "status": "failure",
                "error": "No queues enabled",
                "message": "Both IO queue and Job queue are disabled",
            }
        )

    return _json_response(stats)


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
    "reload_workflows",
    "resume_workflow",
    # Async execution tools
    "get_job_status",
    "cancel_job",
    "list_jobs",
    "get_queue_stats",
]
