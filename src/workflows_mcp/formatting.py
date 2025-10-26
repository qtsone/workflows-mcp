"""Shared formatting utilities for MCP tool responses.

This module provides reusable formatting functions for tool responses,
following the DRY (Don't Repeat Yourself) principle. All formatting logic
for markdown and JSON responses is centralized here.

Following MCP best practices:
- Markdown format: Human-readable with headers, lists, and formatting
- JSON format: Machine-readable structured data for programmatic access
"""

from datetime import datetime
from typing import Any

# =============================================================================
# Markdown Formatting Utilities
# =============================================================================


def format_workflow_list_markdown(workflows: list[str], tags: list[str] | None = None) -> str:
    """Format workflow list as markdown.

    Args:
        workflows: List of workflow names
        tags: Optional tags used for filtering (for display)

    Returns:
        Markdown-formatted workflow list with headers
    """
    if not workflows:
        tag_msg = f" with tags: {', '.join(tags)}" if tags else ""
        return f"No workflows found{tag_msg}"

    header = f"## Available Workflows ({len(workflows)})"
    if tags:
        header += f"\n**Filtered by tags**: {', '.join(tags)}"

    workflow_list = "\n".join(f"- {name}" for name in workflows)
    return f"{header}\n\n{workflow_list}"


def format_workflow_info_markdown(info: dict[str, Any]) -> str:
    """Format workflow info as markdown.

    Args:
        info: Workflow information dictionary

    Returns:
        Markdown-formatted workflow information with sections
    """
    lines = [
        f"# Workflow: {info['name']}",
        "",
        info["description"],
        "",
        "## Configuration",
        f"- **Version**: {info.get('version', '1.0')}",
        f"- **Total Blocks**: {info['total_blocks']}",
    ]

    if "tags" in info and info["tags"]:
        lines.append(f"- **Tags**: {', '.join(info['tags'])}")

    if "author" in info:
        lines.append(f"- **Author**: {info['author']}")

    lines.append("")
    lines.append("## Blocks")
    for block in info["blocks"]:
        block_line = f"- **{block['id']}** ({block['type']})"
        if block.get("depends_on"):
            block_line += f" - depends on: {', '.join(block['depends_on'])}"
        lines.append(block_line)

    if "inputs" in info and info["inputs"]:
        lines.append("")
        lines.append("## Inputs")
        for name, spec in info["inputs"].items():
            desc = spec.get("description", "No description")
            lines.append(f"- **{name}** ({spec['type']}): {desc}")

    if "outputs" in info and info["outputs"]:
        lines.append("")
        lines.append("## Outputs")
        for name, expr in info["outputs"].items():
            lines.append(f"- **{name}**: `{expr}`")

    return "\n".join(lines)


def format_checkpoint_list_markdown(
    checkpoints: list[dict[str, Any]], workflow_filter: str | None = None
) -> str:
    """Format checkpoint list as markdown.

    Args:
        checkpoints: List of checkpoint data dictionaries
        workflow_filter: Optional workflow name used for filtering

    Returns:
        Markdown-formatted checkpoint list
    """
    if not checkpoints:
        filter_msg = f" for workflow: {workflow_filter}" if workflow_filter else ""
        return f"No checkpoints found{filter_msg}"

    lines = [f"## Available Checkpoints ({len(checkpoints)})"]
    if workflow_filter:
        lines.append(f"**Filtered by workflow**: {workflow_filter}")
    lines.append("")

    for cp in checkpoints:
        checkpoint_type = str(cp["type"]).capitalize()
        created_at_value = cp["created_at"]
        if isinstance(created_at_value, (int, float)):
            created_at_ts = float(created_at_value)
        else:
            created_at_ts = 0.0
        created_dt = datetime.fromtimestamp(created_at_ts).strftime("%Y-%m-%d %H:%M:%S")

        lines.append(f"### {cp['checkpoint_id']}")
        lines.append(f"- **Workflow**: {cp['workflow']}")
        lines.append(f"- **Type**: {checkpoint_type}")
        lines.append(f"- **Created**: {created_dt}")
        if cp["is_paused"] and cp["pause_prompt"]:
            lines.append(f"- **Pause Prompt**: {cp['pause_prompt']}")
        lines.append("")

    return "\n".join(lines)


def format_checkpoint_info_markdown(state: Any) -> str:
    """Format checkpoint info as markdown.

    Args:
        state: CheckpointState object

    Returns:
        Markdown-formatted checkpoint information
    """
    created_dt = datetime.fromtimestamp(state.created_at).strftime("%Y-%m-%d %H:%M:%S")
    checkpoint_type = "Pause" if state.paused_block_id else "Automatic"

    # Calculate progress
    total_blocks = sum(len(wave) for wave in state.execution_waves)
    if total_blocks > 0:
        progress_percentage = len(state.completed_blocks) / total_blocks * 100
    else:
        progress_percentage = 0

    lines = [
        f"# Checkpoint: {state.checkpoint_id}",
        "",
        f"**Workflow**: {state.workflow_name}",
        f"**Type**: {checkpoint_type}",
        f"**Created**: {created_dt}",
        "",
        "## Progress",
        f"- **Current Wave**: {state.current_wave_index} / {len(state.execution_waves)}",
        (
            f"- **Completed Blocks**: {len(state.completed_blocks)} / {total_blocks} "
            f"({round(progress_percentage, 1)}%)"
        ),
    ]

    if state.paused_block_id:
        lines.append("")
        lines.append("## Pause Information")
        lines.append(f"- **Paused Block ID**: {state.paused_block_id}")
        if state.pause_prompt:
            lines.append(f"- **Prompt**: {state.pause_prompt}")

    if state.completed_blocks:
        lines.append("")
        lines.append("## Completed Blocks")
        for block_id in state.completed_blocks:
            lines.append(f"- {block_id}")

    return "\n".join(lines)


# =============================================================================
# Error Formatting Utilities
# =============================================================================


def format_workflow_not_found_error(
    workflow_name: str, available: list[str], format_type: str = "json"
) -> dict[str, Any] | str:
    """Format workflow not found error with helpful guidance.

    Args:
        workflow_name: The workflow name that was not found
        available: List of available workflow names
        format_type: Response format ("json" or "markdown")

    Returns:
        Error message in requested format with available workflows
    """
    if format_type == "markdown":
        workflow_list = "\n".join(f"- {name}" for name in available)
        return (
            f"**Error**: Workflow not found: `{workflow_name}`\n\n"
            f"**Available workflows:**\n{workflow_list}"
        )
    else:
        return {
            "error": f"Workflow not found: {workflow_name}",
            "available_workflows": available,
        }


def format_checkpoint_not_found_error(
    checkpoint_id: str, format_type: str = "json"
) -> dict[str, Any] | str:
    """Format checkpoint not found error.

    Args:
        checkpoint_id: The checkpoint ID that was not found
        format_type: Response format ("json" or "markdown")

    Returns:
        Error message in requested format
    """
    if format_type == "markdown":
        return f"**Error**: Checkpoint `{checkpoint_id}` not found or expired"
    else:
        return {
            "found": False,
            "error": f"Checkpoint {checkpoint_id} not found or expired",
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Markdown formatters
    "format_workflow_list_markdown",
    "format_workflow_info_markdown",
    "format_checkpoint_list_markdown",
    "format_checkpoint_info_markdown",
    # Error formatters
    "format_workflow_not_found_error",
    "format_checkpoint_not_found_error",
]
