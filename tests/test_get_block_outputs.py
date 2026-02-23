"""
Unit tests for WorkflowRunner._get_block_outputs.

Verifies that Workflow blocks (returning Execution objects) emit only
declared outputs, not the full child execution tree via model_dump().
"""

from pydantic import BaseModel, Field

from workflows_mcp.engine.execution import Execution
from workflows_mcp.engine.metadata import Metadata
from workflows_mcp.engine.workflow_runner import WorkflowRunner


class FakeBlockOutput(BaseModel):
    """Mimics ShellOutput/LLMCallOutput for testing."""

    stdout: str = Field(default="")
    exit_code: int = Field(default=0)


def _make_runner() -> WorkflowRunner:
    """Create a bare WorkflowRunner (no callback needed for _get_block_outputs)."""
    return WorkflowRunner()


def test_get_block_outputs_none_returns_empty_dict() -> None:
    """None input returns empty dict."""
    runner = _make_runner()
    assert runner._get_block_outputs(None) == {}


def test_get_block_outputs_regular_block_uses_model_dump() -> None:
    """Regular BlockOutput (Shell/LLMCall) returns full model_dump()."""
    runner = _make_runner()
    output = FakeBlockOutput(stdout="hello world", exit_code=0)
    result = runner._get_block_outputs(output)

    assert result == {"stdout": "hello world", "exit_code": 0}


def test_get_block_outputs_execution_returns_only_declared_outputs() -> None:
    """Execution (Workflow block) returns only declared outputs, not full tree.

    This is the core fix: Workflow blocks return Execution objects.
    Before the fix, model_dump() serialized the ENTIRE child execution tree
    including all child blocks, metadata, and nested executions (~100KB+).
    After the fix, only the declared outputs (~few KB) are returned.
    """
    runner = _make_runner()

    # Simulate a child workflow execution with nested blocks and large content
    child_execution = Execution(
        inputs={"source_url": "https://example.com/large-doc.pdf"},
        outputs={
            "document_text": "extracted text (trimmed for test)",
            "metadata_json": '{"title": "Test Document"}',
        },
        blocks={
            "extract": Execution(
                inputs={"url": "https://example.com/large-doc.pdf"},
                outputs={"stdout": "A" * 84000},  # Simulates 84K document text
                metadata=Metadata.create_leaf_success(
                    type="Shell",
                    id="extract",
                    duration_ms=1500,
                    started_at="2026-01-01T00:00:00Z",
                    wave=0,
                    execution_order=0,
                    index=0,
                    depth=1,
                ),
            ),
            "metadata": Execution(
                inputs={"text": "some text"},
                outputs={"response": '{"title": "Test Document"}'},
                metadata=Metadata.create_leaf_success(
                    type="LLMCall",
                    id="metadata",
                    duration_ms=2000,
                    started_at="2026-01-01T00:00:01Z",
                    wave=1,
                    execution_order=1,
                    index=1,
                    depth=1,
                ),
            ),
        },
    )

    result = runner._get_block_outputs(child_execution)

    # Should return ONLY declared outputs
    assert result == {
        "document_text": "extracted text (trimmed for test)",
        "metadata_json": '{"title": "Test Document"}',
    }

    # Crucially: should NOT contain the full child block tree
    assert "blocks" not in result
    assert "inputs" not in result
    assert "metadata" not in result
    # And should NOT contain the 84K content from child blocks
    assert "A" * 84000 not in str(result)


def test_get_block_outputs_execution_empty_outputs() -> None:
    """Execution with empty outputs returns empty dict."""
    runner = _make_runner()

    child_execution = Execution(
        inputs={"key": "value"},
        outputs={},
        blocks={"some_block": Execution(inputs={}, outputs={"data": "big"})},
    )

    result = runner._get_block_outputs(child_execution)
    assert result == {}


def test_get_block_outputs_does_not_affect_regular_block_with_extra_fields() -> None:
    """Regular BlockOutput with extra fields (custom outputs) still works."""

    class CustomOutput(BaseModel):
        model_config = {"extra": "allow"}
        stdout: str = ""
        exit_code: int = 0

    runner = _make_runner()
    output = CustomOutput(stdout="ok", exit_code=0, custom_field="custom_value")
    result = runner._get_block_outputs(output)

    assert result["stdout"] == "ok"
    assert result["exit_code"] == 0
    assert result["custom_field"] == "custom_value"
