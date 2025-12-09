"""
Unit tests for nested ExecutionState serialization.

Tests verify that ExecutionState objects nested in pause_metadata
survive the serialize/deserialize cycle (critical bug fix for nested
workflow pause/resume after multiple parent resume cycles).
"""

from workflows_mcp.engine.execution import Execution
from workflows_mcp.engine.execution_result import ExecutionResult, ExecutionState
from workflows_mcp.engine.workflow_runner import WorkflowRunner


def test_execution_state_with_nested_execution_state() -> None:
    """
    Test that nested ExecutionState in pause_metadata serializes correctly.

    This reproduces the bug where nested workflows complete with empty outputs
    instead of pausing correctly after parent workflow has paused multiple times.

    Bug: ExecutionState objects in pause_metadata were converted to strings
         by json.dump(default=str), causing deserialization failures.

    Fix: Added _serialize_pause_metadata() to recursively convert ExecutionState
         objects to dicts before JSON serialization.
    """
    # Create inner Execution context (child workflow's context)
    inner_execution = Execution(
        workflow_name="inner-workflow",
        inputs={"param": "value"},
        blocks={},
        outputs={},
    )

    # Create inner ExecutionState (what child workflow stores when it pauses)
    inner_execution_state = ExecutionState(
        context=inner_execution,
        completed_blocks=["block1"],
        current_wave_index=1,
        execution_waves=[["block1"], ["block2"]],
        block_definitions={"block1": {"type": "Shell"}, "block2": {"type": "Prompt"}},
        workflow_stack=["parent", "inner-workflow"],
        paused_block_id="block2",
        workflow_name="inner-workflow",
        runtime_inputs={"param": "value"},
        pause_metadata=None,
    )

    # Create outer Execution context (parent workflow's context)
    outer_execution = Execution(
        workflow_name="outer-workflow",
        inputs={"outer_param": "outer_value"},
        blocks={},
        outputs={},
    )

    # Create outer ExecutionState with nested ExecutionState in pause_metadata
    # This is exactly what happens when parent workflow block (Workflow type)
    # catches ExecutionPaused from child and stores child's execution_state
    outer_pause_metadata = {
        "child_execution_state": inner_execution_state,  # ExecutionState object!
        "child_workflow": "inner-workflow",
        "child_pause_metadata": {"prompt": "inner prompt"},
    }

    outer_execution_state = ExecutionState(
        context=outer_execution,
        completed_blocks=["setup"],
        current_wave_index=2,
        execution_waves=[["setup"], ["call_inner"]],
        block_definitions={"setup": {"type": "Shell"}, "call_inner": {"type": "Workflow"}},
        workflow_stack=["parent"],
        paused_block_id="call_inner",
        workflow_name="outer-workflow",
        runtime_inputs={"outer_param": "outer_value"},
        pause_metadata=outer_pause_metadata,
    )

    # Create ExecutionResult
    result = ExecutionResult.paused(
        prompt="Inner workflow prompt",
        execution=outer_execution,
        execution_state=outer_execution_state,
    )

    # Serialize (this is what _build_debug_data does before json.dump)
    serialized = result._build_debug_data()

    # Verify child_execution_state is serialized as dict, not string!
    pause_meta = serialized["execution_state"]["pause_metadata"]
    assert isinstance(pause_meta["child_execution_state"], dict), (
        f"child_execution_state should be dict, got {type(pause_meta['child_execution_state'])}. "
        "BUG NOT FIXED: ExecutionState was converted to string by json.dump(default=str)"
    )

    # Verify nested structure is correct
    child_state = pause_meta["child_execution_state"]
    assert child_state["workflow_name"] == "inner-workflow"
    assert child_state["paused_block_id"] == "block2"
    assert child_state["completed_blocks"] == ["block1"]
    assert "context" in child_state
    assert isinstance(child_state["context"], dict)

    # Verify the full round-trip works (serialize -> deserialize)
    restored_state = WorkflowRunner._extract_execution_state(serialized)
    assert restored_state.workflow_name == "outer-workflow"
    assert restored_state.pause_metadata is not None
    assert isinstance(restored_state.pause_metadata["child_execution_state"], dict)


def test_deeply_nested_execution_states() -> None:
    """
    Test serialization of deeply nested ExecutionState (3+ levels).

    This tests the recursive nature of _serialize_pause_metadata() for
    workflows that call workflows that call workflows (fractal pattern).
    """
    # Level 3 (deepest) - the innermost workflow that actually paused
    level3_exec = Execution(workflow_name="level3", inputs={}, blocks={}, outputs={})
    level3_state = ExecutionState(
        context=level3_exec,
        completed_blocks=[],
        current_wave_index=0,
        execution_waves=[["prompt"]],
        block_definitions={"prompt": {"type": "Prompt"}},
        workflow_stack=["level1", "level2", "level3"],
        paused_block_id="prompt",
        workflow_name="level3",
        runtime_inputs={},
        pause_metadata=None,
    )

    # Level 2 - middle workflow calling level 3
    level2_exec = Execution(workflow_name="level2", inputs={}, blocks={}, outputs={})
    level2_state = ExecutionState(
        context=level2_exec,
        completed_blocks=[],
        current_wave_index=0,
        execution_waves=[["call_level3"]],
        block_definitions={"call_level3": {"type": "Workflow"}},
        workflow_stack=["level1", "level2"],
        paused_block_id="call_level3",
        workflow_name="level2",
        runtime_inputs={},
        pause_metadata={
            "child_execution_state": level3_state,
            "child_workflow": "level3",
        },
    )

    # Level 1 (top) - outermost workflow calling level 2
    level1_exec = Execution(workflow_name="level1", inputs={}, blocks={}, outputs={})
    level1_state = ExecutionState(
        context=level1_exec,
        completed_blocks=[],
        current_wave_index=0,
        execution_waves=[["call_level2"]],
        block_definitions={"call_level2": {"type": "Workflow"}},
        workflow_stack=["level1"],
        paused_block_id="call_level2",
        workflow_name="level1",
        runtime_inputs={},
        pause_metadata={
            "child_execution_state": level2_state,
            "child_workflow": "level2",
        },
    )

    # Create result and serialize
    result = ExecutionResult.paused(
        prompt="Level 3 prompt",
        execution=level1_exec,
        execution_state=level1_state,
    )
    serialized = result._build_debug_data()

    # Navigate to deepest level and verify all are dicts (not strings!)
    level1_meta = serialized["execution_state"]["pause_metadata"]
    assert isinstance(level1_meta["child_execution_state"], dict), "Level 2 state should be dict"

    level2_meta = level1_meta["child_execution_state"]["pause_metadata"]
    assert isinstance(level2_meta["child_execution_state"], dict), "Level 3 state should be dict"

    level3_meta = level2_meta["child_execution_state"]
    assert level3_meta["workflow_name"] == "level3"
    assert level3_meta["paused_block_id"] == "prompt"
    assert level3_meta["pause_metadata"] is None  # Deepest level has no children


def test_execution_state_with_list_containing_dicts() -> None:
    """
    Test that _serialize_pause_metadata handles lists containing dicts.

    Some pause metadata structures may contain lists of dicts that need
    recursive processing.
    """
    execution = Execution(workflow_name="test", inputs={}, blocks={}, outputs={})

    inner_state = ExecutionState(
        context=execution,
        completed_blocks=[],
        current_wave_index=0,
        execution_waves=[],
        block_definitions={},
        workflow_stack=[],
        paused_block_id="test",
        workflow_name="inner",
        runtime_inputs={},
        pause_metadata=None,
    )

    # pause_metadata with list containing a dict with ExecutionState
    complex_metadata = {
        "items": [
            {"name": "item1", "nested_state": inner_state},
            {"name": "item2", "value": 42},
        ],
        "simple_list": [1, 2, 3],
    }

    outer_state = ExecutionState(
        context=execution,
        completed_blocks=[],
        current_wave_index=0,
        execution_waves=[],
        block_definitions={},
        workflow_stack=[],
        paused_block_id="outer",
        workflow_name="outer",
        runtime_inputs={},
        pause_metadata=complex_metadata,
    )

    result = ExecutionResult.paused(
        prompt="Test",
        execution=execution,
        execution_state=outer_state,
    )
    serialized = result._build_debug_data()

    # Verify list handling
    items = serialized["execution_state"]["pause_metadata"]["items"]
    assert isinstance(items, list)
    assert len(items) == 2

    # First item should have serialized ExecutionState
    assert isinstance(items[0]["nested_state"], dict)
    assert items[0]["nested_state"]["workflow_name"] == "inner"

    # Second item should be unchanged
    assert items[1]["value"] == 42

    # Simple list should be unchanged
    assert serialized["execution_state"]["pause_metadata"]["simple_list"] == [1, 2, 3]


def test_serialize_pause_metadata_preserves_none() -> None:
    """Test that _serialize_pause_metadata handles None correctly."""
    execution = Execution(workflow_name="test", inputs={}, blocks={}, outputs={})
    execution_state = ExecutionState(
        context=execution,
        completed_blocks=[],
        current_wave_index=0,
        execution_waves=[],
        block_definitions={},
        workflow_stack=[],
        paused_block_id="test",
        workflow_name="test",
        runtime_inputs={},
        pause_metadata=None,
    )

    result = ExecutionResult.paused(
        prompt="Test",
        execution=execution,
        execution_state=execution_state,
    )
    serialized = result._build_debug_data()

    assert serialized["execution_state"]["pause_metadata"] is None


def test_serialize_pause_metadata_preserves_primitives() -> None:
    """Test that _serialize_pause_metadata preserves primitive values."""
    execution = Execution(workflow_name="test", inputs={}, blocks={}, outputs={})
    execution_state = ExecutionState(
        context=execution,
        completed_blocks=[],
        current_wave_index=0,
        execution_waves=[],
        block_definitions={},
        workflow_stack=[],
        paused_block_id="test",
        workflow_name="test",
        runtime_inputs={},
        pause_metadata={
            "string_value": "hello",
            "int_value": 42,
            "float_value": 3.14,
            "bool_value": True,
            "none_value": None,
        },
    )

    result = ExecutionResult.paused(
        prompt="Test",
        execution=execution,
        execution_state=execution_state,
    )
    serialized = result._build_debug_data()

    pause_meta = serialized["execution_state"]["pause_metadata"]
    assert pause_meta["string_value"] == "hello"
    assert pause_meta["int_value"] == 42
    assert pause_meta["float_value"] == 3.14
    assert pause_meta["bool_value"] is True
    assert pause_meta["none_value"] is None
