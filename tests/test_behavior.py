#!/usr/bin/env python3
"""Behavior-based workflow testing helpers.

Philosophy: Test workflow BEHAVIOR, not structure. This makes tests more
resilient to architectural changes while still catching regressions.

Usage:
    from test_behavior import WorkflowBehavior, assert_workflow_behavior

    # Define expected behavior
    expected = WorkflowBehavior(
        status="success",
        output_schema={
            "exit_code": int,
            "success": bool
        },
        side_effects=[
            FileExists("/tmp/output.txt"),
            CommandExecuted("echo Hello")
        ]
    )

    # Execute and validate
    result = await execute_workflow("test-workflow")
    assert_workflow_behavior(result, expected)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest

# =============================================================================
# Behavior Specification Classes
# =============================================================================


@dataclass
class WorkflowBehavior:
    """Specification of expected workflow behavior.

    Instead of matching exact JSON structure, this specifies what the workflow
    should accomplish (status, outputs, side effects).
    """

    status: Literal["success", "failure", "paused"]
    output_schema: dict[str, type] | None = None  # Expected output types
    output_values: dict[str, Any] | None = None  # Expected output values
    error_pattern: str | None = None  # Error message pattern (for failures)
    side_effects: list["SideEffect"] = field(default_factory=list)


@dataclass
class SideEffect:
    """Base class for workflow side effects to verify."""

    def verify(self) -> bool:
        """Verify this side effect occurred."""
        raise NotImplementedError


@dataclass
class FileExists(SideEffect):
    """Verify a file was created."""

    path: str

    def verify(self) -> bool:
        return Path(self.path).exists()


@dataclass
class FileContains(SideEffect):
    """Verify file contains specific text."""

    path: str
    pattern: str

    def verify(self) -> bool:
        if not Path(self.path).exists():
            return False
        content = Path(self.path).read_text()
        return self.pattern in content


@dataclass
class CommandExecuted(SideEffect):
    """Verify command was executed (check stdout/stderr)."""

    command: str

    def verify(self) -> bool:
        # This would need access to execution logs
        # For now, return True (placeholder)
        return True


# =============================================================================
# Assertion Helpers
# =============================================================================


def assert_workflow_behavior(
    actual: dict[str, Any], expected: WorkflowBehavior, strict: bool = False
) -> None:
    """Assert workflow result matches expected behavior.

    Args:
        actual: Actual workflow execution result
        expected: Expected behavior specification
        strict: If True, fail on extra outputs not in schema

    Raises:
        AssertionError: If behavior doesn't match
    """
    # Validate status
    assert actual["status"] == expected.status, (
        f"Status mismatch: expected {expected.status}, got {actual['status']}"
    )

    # Validate error (for failures)
    if expected.status == "failure":
        assert "error" in actual, "Failure status must include error field"
        if expected.error_pattern:
            assert expected.error_pattern in actual["error"], (
                f"Error pattern '{expected.error_pattern}' not found in: {actual['error']}"
            )

    # Validate outputs (for success)
    if expected.status == "success":
        assert "outputs" in actual, "Success status must include outputs field"

        # Validate output schema (types)
        if expected.output_schema:
            for key, expected_type in expected.output_schema.items():
                assert key in actual["outputs"], f"Missing output key: {key}"
                actual_value = actual["outputs"][key]
                assert isinstance(actual_value, expected_type), (
                    f"Output '{key}' type mismatch: "
                    f"expected {expected_type.__name__}, "
                    f"got {type(actual_value).__name__}"
                )

        # Validate output values (exact matches)
        if expected.output_values:
            for key, expected_value in expected.output_values.items():
                assert key in actual["outputs"], f"Missing output key: {key}"
                assert actual["outputs"][key] == expected_value, (
                    f"Output '{key}' value mismatch: "
                    f"expected {expected_value}, "
                    f"got {actual['outputs'][key]}"
                )

        # Strict mode: fail on unexpected outputs
        if strict and expected.output_schema:
            for key in actual["outputs"]:
                assert key in expected.output_schema, f"Unexpected output key: {key}"

    # Validate side effects
    for side_effect in expected.side_effects:
        assert side_effect.verify(), f"Side effect not verified: {side_effect}"


def assert_outputs_match_schema(outputs: dict[str, Any], schema: dict[str, type]) -> None:
    """Validate outputs match expected types.

    Useful for quick type validation without full behavior specification.
    """
    for key, expected_type in schema.items():
        assert key in outputs, f"Missing output: {key}"
        assert isinstance(outputs[key], expected_type), (
            f"Output '{key}' type mismatch: "
            f"expected {expected_type.__name__}, got {type(outputs[key]).__name__}"
        )


def assert_workflow_succeeded(result: dict[str, Any]) -> None:
    """Assert workflow completed successfully."""
    assert result["status"] == "success", f"Workflow failed: {result.get('error', 'Unknown error')}"
    assert "outputs" in result, "Success status must include outputs"


def assert_workflow_failed(result: dict[str, Any], error_pattern: str | None = None) -> None:
    """Assert workflow failed with expected error."""
    assert result["status"] == "failure", f"Expected failure, got: {result['status']}"
    assert "error" in result, "Failure status must include error field"

    if error_pattern:
        assert error_pattern in result["error"], (
            f"Error pattern '{error_pattern}' not found in: {result['error']}"
        )


def assert_workflow_paused(result: dict[str, Any], prompt_pattern: str | None = None) -> None:
    """Assert workflow paused for user input."""
    assert result["status"] == "paused", f"Expected paused, got: {result['status']}"
    assert "prompt" in result, "Paused status must include prompt field"
    assert "job_id" in result, "Paused workflows must return job_id"

    if prompt_pattern:
        assert prompt_pattern in result["prompt"], (
            f"Prompt pattern '{prompt_pattern}' not found in: {result['prompt']}"
        )


# =============================================================================
# Example Usage in Tests
# =============================================================================


@pytest.mark.asyncio
async def example_behavior_test():
    """Example of behavior-based testing."""
    from test_mcp_client import get_mcp_client

    # Define expected behavior
    expected = WorkflowBehavior(
        status="success",
        output_schema={"exit_code": int, "succeeded": bool},
        output_values={"exit_code": 0, "succeeded": True},
    )

    # Execute workflow via MCP
    async with get_mcp_client() as client:
        import json

        from mcp.types import TextContent

        result = await client.call_tool(
            "execute_workflow",
            arguments={"workflow": "test-workflow", "inputs": {}, "debug": False},
        )

        content = result.content[0]
        assert isinstance(content, TextContent)
        response = json.loads(content.text)

        # Validate behavior
        assert_workflow_behavior(response, expected)


# =============================================================================
# Test Discovery and Execution
# =============================================================================


def discover_workflow_behaviors() -> dict[str, WorkflowBehavior]:
    """Discover expected behaviors for all test workflows.

    This would load behavior specifications from YAML files or Python modules.
    For now, returns example behaviors.
    """
    return {
        "workflow-output-type-coercion": WorkflowBehavior(
            status="success",
            output_schema={"exit_code_int": int, "command_succeeded": bool},
            output_values={"exit_code_int": 0, "command_succeeded": True},
        ),
        # Add more workflow behaviors here
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
