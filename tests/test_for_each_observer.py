#!/usr/bin/env python3
"""Unit tests for on_block_transition observer events on for_each blocks.

Tests that for_each blocks emit the same lifecycle events as regular blocks:
- block_started / block_completed / block_failed for the parent for_each block
- block_started / block_completed / block_failed for each iteration
"""

from pathlib import Path
from typing import Any

import pytest

from workflows_mcp.engine.execution_context import ExecutionContext
from workflows_mcp.engine.executor_base import create_default_registry
from workflows_mcp.engine.llm_config import LLMConfigLoader
from workflows_mcp.engine.registry import WorkflowRegistry
from workflows_mcp.engine.workflow_runner import WorkflowRunner

WORKFLOWS_DIR = Path(__file__).parent / "workflows"


def _load_registry() -> WorkflowRegistry:
    """Load test workflows from the standard test workflows directory."""
    registry = WorkflowRegistry()
    registry.load_from_directory(WORKFLOWS_DIR)
    return registry


class TestForEachObserverEvents:
    """Test on_block_transition events for for_each blocks."""

    @pytest.mark.asyncio
    async def test_for_each_parent_emits_started_and_completed(self):
        """Parent for_each block emits block_started and block_completed.

        Uses for-each-comprehensive workflow which has two for_each blocks:
        - process_files (parallel, list input)
        - configure_services (sequential, dict input)
        """
        events: list[dict[str, Any]] = []

        async def capture_event(event: dict[str, Any]) -> None:
            events.append(event)

        registry = _load_registry()
        executor_registry = create_default_registry()
        llm_config_loader = LLMConfigLoader()
        context = ExecutionContext(
            workflow_registry=registry,
            executor_registry=executor_registry,
            llm_config_loader=llm_config_loader,
            io_queue=None,
        )

        runner = WorkflowRunner(
            on_block_transition=capture_event,
        )

        workflow = registry.get("for-each-comprehensive")
        assert workflow is not None

        await runner.execute(workflow=workflow, context=context)

        # Extract parent-level events for for_each blocks
        process_files_events = [e for e in events if e["block_id"] == "process_files"]
        configure_services_events = [e for e in events if e["block_id"] == "configure_services"]

        # Verify process_files (parallel for_each) has started + completed
        event_types = [e["event"] for e in process_files_events]
        assert "block_started" in event_types, (
            f"process_files missing block_started. Got: {event_types}"
        )
        assert "block_completed" in event_types, (
            f"process_files missing block_completed. Got: {event_types}"
        )

        # Verify configure_services (sequential for_each) has started + completed
        event_types = [e["event"] for e in configure_services_events]
        assert "block_started" in event_types, (
            f"configure_services missing block_started. Got: {event_types}"
        )
        assert "block_completed" in event_types, (
            f"configure_services missing block_completed. Got: {event_types}"
        )

    @pytest.mark.asyncio
    async def test_for_each_parent_event_contains_block_type(self):
        """Parent for_each events use the executor type, not 'for_each'."""
        events: list[dict[str, Any]] = []

        async def capture_event(event: dict[str, Any]) -> None:
            events.append(event)

        registry = _load_registry()
        context = ExecutionContext(
            workflow_registry=registry,
            executor_registry=create_default_registry(),
            llm_config_loader=LLMConfigLoader(),
            io_queue=None,
        )

        runner = WorkflowRunner(on_block_transition=capture_event)
        workflow = registry.get("for-each-comprehensive")
        assert workflow is not None

        await runner.execute(workflow=workflow, context=context)

        # process_files is a Shell block with for_each — type must be "Shell"
        started = next(
            e for e in events if e["block_id"] == "process_files" and e["event"] == "block_started"
        )
        assert started["block_type"] == "Shell"

    @pytest.mark.asyncio
    async def test_for_each_iterations_emit_events(self):
        """Individual iterations within for_each emit their own events."""
        events: list[dict[str, Any]] = []

        async def capture_event(event: dict[str, Any]) -> None:
            events.append(event)

        registry = _load_registry()
        context = ExecutionContext(
            workflow_registry=registry,
            executor_registry=create_default_registry(),
            llm_config_loader=LLMConfigLoader(),
            io_queue=None,
        )

        runner = WorkflowRunner(on_block_transition=capture_event)
        workflow = registry.get("for-each-comprehensive")
        assert workflow is not None

        await runner.execute(workflow=workflow, context=context)

        # process_files iterates over 3 items: keys "0", "1", "2"
        # Each should have at least block_started
        iteration_keys = {"0", "1", "2"}
        iteration_started = [
            e for e in events if e["block_id"] in iteration_keys and e["event"] == "block_started"
        ]
        assert len(iteration_started) == 3, (
            f"Expected 3 iteration block_started events, "
            f"got {len(iteration_started)}: {iteration_started}"
        )

        # Iterations should have higher depth than parent and reference parent
        parent_started = next(
            e for e in events if e["block_id"] == "process_files" and e["event"] == "block_started"
        )
        for iter_event in iteration_started:
            assert iter_event["depth"] > parent_started["depth"], (
                "Iteration depth must be > parent depth"
            )
            assert iter_event.get("parent_block_id") == "process_files", (
                "Iteration events must reference parent_block_id"
            )

    @pytest.mark.asyncio
    async def test_for_each_failed_iteration_emits_block_failed(self):
        """Failed iterations emit block_failed events."""
        events: list[dict[str, Any]] = []

        async def capture_event(event: dict[str, Any]) -> None:
            events.append(event)

        registry = _load_registry()
        context = ExecutionContext(
            workflow_registry=registry,
            executor_registry=create_default_registry(),
            llm_config_loader=LLMConfigLoader(),
            io_queue=None,
        )

        runner = WorkflowRunner(on_block_transition=capture_event)
        workflow = registry.get("for-each-comprehensive")
        assert workflow is not None

        await runner.execute(workflow=workflow, context=context)

        # process_files has continue_on_error=true and iteration "2" (missing.txt)
        # fails with exit code 1 — should emit block_failed for key "2"
        failed_events = [e for e in events if e["block_id"] == "2" and e["event"] == "block_failed"]
        assert len(failed_events) == 1, (
            f"Expected 1 block_failed for iteration '2', got {len(failed_events)}"
        )

    @pytest.mark.asyncio
    async def test_for_each_completed_metadata_in_parent(self):
        """Parent block_completed event contains aggregated metadata."""
        events: list[dict[str, Any]] = []

        async def capture_event(event: dict[str, Any]) -> None:
            events.append(event)

        registry = _load_registry()
        context = ExecutionContext(
            workflow_registry=registry,
            executor_registry=create_default_registry(),
            llm_config_loader=LLMConfigLoader(),
            io_queue=None,
        )

        runner = WorkflowRunner(on_block_transition=capture_event)
        workflow = registry.get("for-each-comprehensive")
        assert workflow is not None

        await runner.execute(workflow=workflow, context=context)

        completed = next(
            e
            for e in events
            if e["block_id"] == "process_files" and e["event"] == "block_completed"
        )
        assert "metadata" in completed
        meta = completed["metadata"]
        assert meta["count"] == 3
        assert meta["count_failed"] == 1  # missing.txt fails

    @pytest.mark.asyncio
    async def test_event_ordering_started_before_completed(self):
        """block_started always appears before block_completed for same block."""
        events: list[dict[str, Any]] = []

        async def capture_event(event: dict[str, Any]) -> None:
            events.append(event)

        registry = _load_registry()
        context = ExecutionContext(
            workflow_registry=registry,
            executor_registry=create_default_registry(),
            llm_config_loader=LLMConfigLoader(),
            io_queue=None,
        )

        runner = WorkflowRunner(on_block_transition=capture_event)
        workflow = registry.get("for-each-comprehensive")
        assert workflow is not None

        await runner.execute(workflow=workflow, context=context)

        # For process_files: started must come before completed
        pf_events = [e for e in events if e["block_id"] == "process_files"]
        started_idx = next(i for i, e in enumerate(pf_events) if e["event"] == "block_started")
        completed_idx = next(i for i, e in enumerate(pf_events) if e["event"] == "block_completed")
        assert started_idx < completed_idx

        # Iteration events should appear between parent started and completed
        parent_started_idx = next(
            i
            for i, e in enumerate(events)
            if e["block_id"] == "process_files" and e["event"] == "block_started"
        )
        parent_completed_idx = next(
            i
            for i, e in enumerate(events)
            if e["block_id"] == "process_files" and e["event"] == "block_completed"
        )
        iteration_events = [
            (i, e)
            for i, e in enumerate(events)
            if e["block_id"] in {"0", "1", "2"}
            and e["event"] in {"block_started", "block_completed", "block_failed"}
        ]
        for idx, _ in iteration_events:
            assert parent_started_idx < idx < parent_completed_idx, (
                "Iteration events must be between parent start and complete"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
