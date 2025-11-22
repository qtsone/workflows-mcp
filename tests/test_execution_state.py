"""
Unit tests for ExecutionState serialization/deserialization.

Tests verify that pause_metadata survives the serialize/deserialize cycle
(ADR-010 bug fix).
"""

from workflows_mcp.engine.execution import Execution
from workflows_mcp.engine.execution_result import ExecutionResult, ExecutionState
from workflows_mcp.engine.workflow_runner import WorkflowRunner


def test_execution_state_pause_metadata_serialization() -> None:
    """
    Test that pause_metadata is preserved during serialize/deserialize cycle.

    This test verifies the ADR-010 bug fix where pause_metadata was defined
    but not serialized/deserialized, causing checkpoint data loss for for_each
    pause/resume.

    Bug: ExecutionState.pause_metadata was NOT serialized in _execution_state_to_dict()
         or deserialized in _extract_execution_state().

    Fix: Added pause_metadata to both methods.
    """
    # Create a minimal Execution context
    execution = Execution(
        workflow_name="test-workflow",
        inputs={},
        blocks={},
        outputs={},
    )

    # Create ExecutionState with pause_metadata (for_each checkpoint format)
    test_pause_metadata = {
        "type": "for_each_iteration",
        "for_each_block_id": "test_block",
        "current_iteration_key": "item1",
        "current_iteration_index": 0,
        "completed_iterations": [],
        "remaining_iteration_keys": ["item2", "item3"],
        "all_iterations": {"item1": "value1", "item2": "value2", "item3": "value3"},
        "executor_type": "Prompt",
        "inputs_template": {"prompt": "{{each.value}}"},
        "mode": "sequential",
    }

    execution_state = ExecutionState(
        context=execution,
        completed_blocks=[],
        current_wave_index=0,
        execution_waves=[["test_block"]],
        block_definitions={"test_block": {"type": "Prompt"}},
        workflow_stack=[],
        paused_block_id="test_block",
        workflow_name="test-workflow",
        runtime_inputs={"key": "value"},
        pause_metadata=test_pause_metadata,
    )

    # Create ExecutionResult with the state
    result = ExecutionResult.paused(
        prompt="Test prompt",
        execution=execution,
        execution_state=execution_state,
    )

    # Serialize: Call _build_debug_data() which uses _execution_state_to_dict()
    serialized = result._build_debug_data()

    # Verify pause_metadata is in serialized data
    assert "execution_state" in serialized, "execution_state missing from serialized data"
    assert "pause_metadata" in serialized["execution_state"], (
        "pause_metadata missing from serialized execution_state (BUG NOT FIXED!)"
    )
    assert serialized["execution_state"]["pause_metadata"] == test_pause_metadata, (
        "pause_metadata not serialized correctly"
    )

    # Deserialize: Call _extract_execution_state()
    restored_state = WorkflowRunner._extract_execution_state(serialized)

    # Verify pause_metadata is restored
    assert restored_state.pause_metadata is not None, (
        "pause_metadata is None after deserialization (BUG NOT FIXED!)"
    )
    assert restored_state.pause_metadata == test_pause_metadata, (
        "pause_metadata not deserialized correctly"
    )

    # Verify other fields are also preserved (sanity check)
    assert restored_state.paused_block_id == "test_block"
    assert restored_state.workflow_name == "test-workflow"
    assert restored_state.runtime_inputs == {"key": "value"}


def test_execution_state_without_pause_metadata() -> None:
    """
    Test that ExecutionState works correctly when pause_metadata is None.

    This ensures backward compatibility - not all paused workflows have
    pause_metadata (only for_each blocks use it).
    """
    # Create minimal Execution context
    execution = Execution(
        workflow_name="test-workflow",
        inputs={},
        blocks={},
        outputs={},
    )

    # Create ExecutionState WITHOUT pause_metadata
    execution_state = ExecutionState(
        context=execution,
        completed_blocks=[],
        current_wave_index=0,
        execution_waves=[["test_block"]],
        block_definitions={"test_block": {"type": "Prompt"}},
        workflow_stack=[],
        paused_block_id="test_block",
        workflow_name="test-workflow",
        runtime_inputs={},
        pause_metadata=None,  # Explicitly None
    )

    # Create ExecutionResult
    result = ExecutionResult.paused(
        prompt="Test prompt",
        execution=execution,
        execution_state=execution_state,
    )

    # Serialize
    serialized = result._build_debug_data()

    # Verify pause_metadata is present but None
    assert "execution_state" in serialized
    assert "pause_metadata" in serialized["execution_state"]
    assert serialized["execution_state"]["pause_metadata"] is None

    # Deserialize
    restored_state = WorkflowRunner._extract_execution_state(serialized)

    # Verify pause_metadata is None (not missing)
    assert restored_state.pause_metadata is None

    # Verify other fields
    assert restored_state.paused_block_id == "test_block"
    assert restored_state.workflow_name == "test-workflow"


def test_execution_state_nested_pause_metadata() -> None:
    """
    Test serialization with nested pause_metadata structure.

    For nested for_each blocks, pause_metadata can contain nested checkpoint data.
    Verify that complex nested structures are preserved.
    """
    # Create minimal Execution context
    execution = Execution(
        workflow_name="test-workflow",
        inputs={},
        blocks={},
        outputs={},
    )

    # Create nested pause_metadata (outer for_each with inner for_each pause)
    nested_pause_metadata = {
        "type": "for_each_iteration",
        "for_each_block_id": "outer_loop",
        "current_iteration_key": "file1",
        "paused_iteration_checkpoint": {
            "type": "workflow",
            "workflow_name": "process_file",
            "nested_pause": {
                "type": "for_each_iteration",
                "for_each_block_id": "inner_loop",
                "current_iteration_key": "item1",
                "deep_nested_data": {
                    "level": 3,
                    "values": [1, 2, 3],
                },
            },
        },
    }

    execution_state = ExecutionState(
        context=execution,
        completed_blocks=[],
        current_wave_index=0,
        execution_waves=[["outer_loop"]],
        block_definitions={},
        workflow_stack=[],
        paused_block_id="outer_loop",
        workflow_name="test-workflow",
        runtime_inputs={},
        pause_metadata=nested_pause_metadata,
    )

    # Create ExecutionResult
    result = ExecutionResult.paused(
        prompt="Test prompt",
        execution=execution,
        execution_state=execution_state,
    )

    # Serialize
    serialized = result._build_debug_data()

    # Deserialize
    restored_state = WorkflowRunner._extract_execution_state(serialized)

    # Verify nested structure is preserved
    assert restored_state.pause_metadata == nested_pause_metadata
    assert (
        restored_state.pause_metadata["paused_iteration_checkpoint"]["nested_pause"][
            "deep_nested_data"
        ]["level"]
        == 3
    )
