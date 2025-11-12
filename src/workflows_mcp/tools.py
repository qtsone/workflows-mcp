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
    mode: Annotated[
        Literal["sync", "async"],
        Field(
            description=(
                "Execution mode (default: 'sync'). "
                "'sync': Blocking execution, returns result immediately. "
                "'async': Non-blocking execution, returns job_id for tracking via get_job_status()."
            )
        ),
    ] = "sync",
    timeout: Annotated[
        int | None,
        Field(
            description=(
                "Job timeout in seconds (async mode only). "
                "Default: 3600 (1 hour). Maximum: 86400 (24 hours). "
                "Use lower values for quick workflows, higher for long-running tasks. "
                "Ignored in sync mode."
            ),
            ge=1,
            le=86400,
        ),
    ] = None,
    *,
    ctx: AppContextType,
) -> dict[str, Any]:
    """Execute a registered workflow template.

    Runs pre-registered workflows from the template library. Workflows execute as DAGs
    with parallel execution of independent blocks. Supports shell commands, git operations,
    file operations, HTTP calls, LLM calls, and workflow composition.

    Use execute_workflow for: Pre-built templates used repeatedly
    Use execute_inline_workflow for: Ad-hoc YAML definitions, testing, one-off tasks

    Execution Modes:
        - sync (default): Blocks until workflow completes, returns full result
        - async: Returns immediately with job_id, check status with get_job_status()

    Returns:
        Sync mode: {
            "status": "success" | "failure" | "paused",
            "outputs": {...},        # Workflow outputs (if defined)
            "error": "...",          # Error details (if failed)
            "job_id": "...",         # Job ID for resume (if paused)
            "prompt": "..."          # Prompt text (if paused)
        }

        Async mode: {
            "job_id": "job_abc123",
            "workflow": "workflow-name",
            "status": "queued",
            "message": "Job submitted. Use get_job_status() to check progress."
        }

    Tool Call Examples:
        # Synchronous execution (default)
        execute_workflow(workflow="python-ci-pipeline", inputs={"project_path": "./app"})

        # Asynchronous execution with default timeout (1 hour)
        execute_workflow(workflow="long-task", inputs={...}, mode="async")

        # Asynchronous execution with custom timeout (5 minutes)
        execute_workflow(workflow="quick-check", inputs={...}, mode="async", timeout=300)

        # Asynchronous execution with extended timeout (4 hours)
        execute_workflow(workflow="ml-training", inputs={...}, mode="async", timeout=14400)

        # Debug mode
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
            f"Workflow paused. Use resume_workflow(job_id='{job_id}') to continue."
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
        title="Get Workflow Schema",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
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
        Field(
            description=(
                "Job ID from paused workflow. Get from: execute_workflow response "
                "(if paused) or list_jobs(status='paused'). Format: 'job_abc123'."
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
    Use when execute_workflow returns status="paused" with job_id.

    Returns:
        Same structure as execute_workflow (status, outputs, error, job_id, prompt)

    Tool Call Examples:
        resume_workflow(job_id="job_abc123", response="yes")
        resume_workflow(job_id="job_xyz789", response="proceed with deployment")
    """
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
    """Get job status.

    Args:
        job_id: Job ID from execute_workflow (async mode)
        ctx: MCP context with access to shared resources

    Returns:
        {
            "id": str,
            "workflow": str,
            "status": "queued" | "running" | "completed" | "failed" | "cancelled" | "paused",
            "outputs": dict | None,  # Workflow outputs only
            "error": str | None,
            "prompt": str | None,  # Pause prompt (only when status="paused")
            "created_at": str,
            "started_at": str | None,
            "completed_at": str | None,
            "result_file": str  # Full job data location for debugging
        }
    """
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
    """Cancel pending or running job.

    Args:
        job_id: Job ID from submit_workflow
        ctx: MCP context with access to shared resources

    Returns:
        Cancellation result

    Example (success):
        {
            "job_id": "job_a1b2c3d4",
            "cancelled": true,
            "message": "Job cancelled successfully"
        }

    Example (already completed):
        {
            "job_id": "job_a1b2c3d4",
            "cancelled": false,
            "message": "Job already completed or failed"
        }
    """
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
    """List jobs with optional status filter.

    Args:
        status: Filter by status (queued/running/completed/failed/cancelled) or None for all
        limit: Maximum number of jobs to return (default: 100)
        ctx: MCP context with access to shared resources

    Returns:
        List of jobs (most recent first)

    Example:
        {
            "jobs": [
                {
                    "id": "job_a1b2c3d4",
                    "workflow": "python-ci-pipeline",
                    "status": "completed",
                    "created_at": "2025-11-10T15:30:00",
                    "started_at": "2025-11-10T15:30:01",
                    "completed_at": "2025-11-10T15:30:45"
                },
                ...
            ],
            "total": 42,
            "filtered": 15
        }
    """
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
    """Get queue statistics for monitoring.

    Returns statistics for both IO queue and Job queue including
    operation counts, queue sizes, and worker status.

    Args:
        ctx: MCP context with access to shared resources

    Returns:
        Statistics for available queues

    Example:
        {
            "io_queue": {
                "total_operations": 1234,
                "successful_operations": 1200,
                "failed_operations": 34,
                "queue_size": 0
            },
            "job_queue": {
                "total_jobs": 56,
                "completed_jobs": 45,
                "failed_jobs": 8,
                "cancelled_jobs": 3,
                "queue_size": 2,
                "active_workers": 3
            }
        }
    """
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
