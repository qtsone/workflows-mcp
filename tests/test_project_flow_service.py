"""Contract tests for ProjectFlowService — the resumable checkpoint state machine.

These tests exercise the service directly with a pure-dict fake executor, with no
Postgres backend. This is the central reason the flow was extracted into a
stateless service over an injected operation port (ADR-014): the gnarly resume
logic is testable in isolation, fast, and DB-free.
"""

from __future__ import annotations

from typing import Any

import pytest

from workflows_mcp.engine.memory_errors import MemoryContractError
from workflows_mcp.engine.project_flow_service import (
    PROJECT_FLOW_VERSION,
    FlowState,
    ProjectFlowService,
    StepSections,
    build_project_checkpoint_payload,
    classify_deletion_policy,
    compute_sync_delta,
    normalize_project_checkpoint,
    restore_checkpoint,
)


class _RecordingExecutor:
    """Fake operation executor: records calls and returns canned per-op results."""

    def __init__(self, results: dict[str, dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any], StepSections]] = []
        self._results = results or {}

    async def __call__(
        self, operation: str, scope: dict[str, Any], sections: StepSections
    ) -> dict[str, Any]:
        self.calls.append((operation, scope, sections))
        return self._results.get(operation, {"operation": operation, "ok": True})


def _shape_error(exc: Exception, stage: str) -> dict[str, Any]:
    code = exc.code if isinstance(exc, MemoryContractError) else "MEM_INTERNAL_ERROR"
    return {"error": {"code": code, "message": str(exc), "retryable": False, "stage": stage}}


def _new_state(
    *,
    ingest: dict[str, Any] | None = None,
    supersede: dict[str, Any] | None = None,
    archive: dict[str, Any] | None = None,
    maintain: dict[str, Any] | None = None,
    scope: dict[str, Any] | None = None,
) -> FlowState:
    scope_resolved, plan, next_index, completed = normalize_project_checkpoint(
        checkpoint=None,
        scope=scope or {"palace": "p"},
        ingest=ingest,
        supersede=supersede,
        archive=archive,
        maintain=maintain,
        require_ingest=False,
    )
    return FlowState(
        scope=scope_resolved,
        plan=plan,
        next_index=next_index,
        completed=completed,
    )


async def _advance(
    state: FlowState,
    executor: _RecordingExecutor,
    *,
    flow_name: str = "onboard",
    max_operations: int = 1,
    debug: bool = False,
) -> Any:
    service = ProjectFlowService()
    return await service.advance(
        state,
        executor=executor,
        error_shaper=_shape_error,
        flow_name=flow_name,
        max_operations=max_operations,
        debug=debug,
    )


# ---------------------------------------------------------------------------
# Single-step progression and pause
# ---------------------------------------------------------------------------


class TestStepProgression:
    @pytest.mark.asyncio
    async def test_single_step_of_multi_step_plan_pauses(self) -> None:
        state = _new_state(ingest={"format": "structured"}, maintain={"op": "x"})
        executor = _RecordingExecutor()
        result = await _advance(state, executor, max_operations=1)

        assert result.completed is False
        assert result.response["status"] == "checkpoint"
        assert result.response["remaining_operations"] == ["maintain"]
        assert result.response["completed_operations"] == ["ingest"]
        assert result.checkpoint["next_index"] == 1
        assert len(executor.calls) == 1

    @pytest.mark.asyncio
    async def test_full_plan_completes_in_one_call_when_budget_allows(self) -> None:
        state = _new_state(ingest={"format": "structured"}, maintain={"op": "x"})
        executor = _RecordingExecutor()
        result = await _advance(state, executor, max_operations=20)

        assert result.completed is True
        assert result.response["status"] == "completed"
        assert result.response["completed_operations"] == ["ingest", "maintain"]
        assert result.checkpoint["next_index"] == 2
        assert len(executor.calls) == 2

    @pytest.mark.asyncio
    async def test_resume_from_returned_checkpoint_runs_next_step(self) -> None:
        state = _new_state(ingest={"format": "structured"}, maintain={"op": "x"})
        executor = _RecordingExecutor()
        first = await _advance(state, executor, max_operations=1)

        # Round-trip the returned checkpoint exactly as an MCP client would.
        scope, plan, next_index, completed = restore_checkpoint(
            first.checkpoint, require_ingest=False
        )
        resumed = FlowState(scope=scope, plan=plan, next_index=next_index, completed=completed)
        second = await _advance(resumed, _RecordingExecutor(), max_operations=1)

        assert second.completed is True
        assert second.response["completed_operations"] == ["ingest", "maintain"]


# ---------------------------------------------------------------------------
# Failure handling — failed checkpoint at the failing index
# ---------------------------------------------------------------------------


class TestStepFailure:
    @pytest.mark.asyncio
    async def test_raised_exception_emits_failed_checkpoint_at_index(self) -> None:
        state = _new_state(ingest={"format": "structured"}, maintain={"op": "x"})

        class _Boom(_RecordingExecutor):
            async def __call__(
                self, operation: str, scope: dict[str, Any], sections: StepSections
            ) -> dict[str, Any]:
                if operation == "maintain":
                    raise MemoryContractError(code="MEM_BOOM", message="boom", retryable=False)
                return {"operation": operation, "ok": True}

        result = await _advance(state, _Boom(), max_operations=20)

        assert result.completed is False
        assert result.response["status"] == "checkpoint"
        assert result.response["failed_operation"] == "maintain"
        assert result.response["error"]["code"] == "MEM_BOOM"
        # ingest committed; resume re-runs from the failing step.
        assert result.checkpoint["next_index"] == 1
        assert result.response["completed_operations"] == ["ingest"]

    @pytest.mark.asyncio
    async def test_error_envelope_in_result_emits_failed_checkpoint(self) -> None:
        state = _new_state(ingest={"format": "structured"})
        executor = _RecordingExecutor(
            results={"ingest": {"error": {"code": "MEM_X", "message": "nope", "retryable": True}}}
        )
        result = await _advance(state, executor, max_operations=1)

        assert result.completed is False
        assert result.response["failed_operation"] == "ingest"
        assert result.response["error"]["code"] == "MEM_X"
        assert result.checkpoint["next_index"] == 0


# ---------------------------------------------------------------------------
# Scan snapshot back-population via the on_ingest port
# ---------------------------------------------------------------------------


class TestOnIngestBackfill:
    @pytest.mark.asyncio
    async def test_on_ingest_updates_carried_snapshot(self) -> None:
        captured: dict[str, Any] = {}

        def _on_ingest(step_result: dict[str, Any]) -> dict[str, Any]:
            captured["result"] = step_result
            return {"scan_config": {}, "entries": [{"path": "a.py", "memory_id": "m1"}]}

        scope, plan, next_index, completed = normalize_project_checkpoint(
            checkpoint=None,
            scope={"palace": "p"},
            ingest={"format": "structured"},
            supersede=None,
            archive=None,
            maintain=None,
            require_ingest=False,
        )
        state = FlowState(
            scope=scope,
            plan=plan,
            next_index=next_index,
            completed=completed,
            scan_snapshot={"scan_config": {}, "entries": [{"path": "a.py"}]},
            on_ingest=_on_ingest,
        )
        executor = _RecordingExecutor(results={"ingest": {"ids": ["m1"]}})
        result = await _advance(state, executor, max_operations=1)

        assert captured["result"] == {"ids": ["m1"]}
        assert result.checkpoint["scan_snapshot"]["entries"][0]["memory_id"] == "m1"


# ---------------------------------------------------------------------------
# Checkpoint validation contract
# ---------------------------------------------------------------------------


class TestCheckpointValidation:
    def test_unsupported_version_rejected(self) -> None:
        with pytest.raises(MemoryContractError) as exc:
            restore_checkpoint(
                {"version": "bad", "plan": [{"operation": "ingest"}]}, require_ingest=False
            )
        assert exc.value.code == "MEM_CHECKPOINT_INVALID"

    def test_checkpoint_and_new_plan_conflict(self) -> None:
        checkpoint = build_project_checkpoint_payload(
            scope={"palace": "p"},
            plan=[{"operation": "ingest", "payload": {}}],
            next_index=0,
            completed=[],
        )
        with pytest.raises(MemoryContractError) as exc:
            normalize_project_checkpoint(
                checkpoint=checkpoint,
                scope=None,
                ingest={"format": "structured"},
                supersede=None,
                archive=None,
                maintain=None,
                require_ingest=False,
            )
        assert exc.value.code == "MEM_CHECKPOINT_CONFLICT"

    def test_completed_length_must_equal_next_index(self) -> None:
        checkpoint = {
            "version": PROJECT_FLOW_VERSION,
            "scope": {"palace": "p"},
            "plan": [{"operation": "ingest", "payload": {}}],
            "next_index": 1,
            "completed": [],
        }
        with pytest.raises(MemoryContractError):
            restore_checkpoint(checkpoint, require_ingest=False)

    def test_onboard_requires_ingest_first(self) -> None:
        with pytest.raises(MemoryContractError) as exc:
            normalize_project_checkpoint(
                checkpoint=None,
                scope={"palace": "p"},
                ingest=None,
                supersede={"ids": ["x"]},
                archive=None,
                maintain=None,
                require_ingest=True,
            )
        assert exc.value.code == "MEM_PROJECT_ONBOARD_REQUIRES_INGEST"


# ---------------------------------------------------------------------------
# Sync-delta and deletion-policy logic (now owned by the service)
# ---------------------------------------------------------------------------


class TestSyncDeltaLogic:
    def test_four_class_delta(self) -> None:
        prior = [
            {"path": "keep.py", "content_hash": "h1"},
            {"path": "change.py", "content_hash": "h2"},
            {"path": "gone.py", "content_hash": "h3"},
        ]
        new = [
            {"path": "keep.py", "content_hash": "h1"},
            {"path": "change.py", "content_hash": "CHANGED"},
            {"path": "added.py", "content_hash": "h4"},
        ]
        delta = compute_sync_delta(prior, new)
        assert delta.added == ["added.py"]
        assert delta.modified == ["change.py"]
        assert delta.deleted == ["gone.py"]
        assert delta.unchanged == ["keep.py"]
        assert delta.has_semantic_delta is True

    def test_identical_inputs_have_no_semantic_delta(self) -> None:
        entries = [{"path": "a.py", "content_hash": "h"}]
        delta = compute_sync_delta(entries, entries)
        assert delta.has_semantic_delta is False

    def test_deletion_policy_aliases_collapse_to_archive(self) -> None:
        assert classify_deletion_policy("mark_missing") == "archive"
        assert classify_deletion_policy("archive_missing") == "archive"
        assert classify_deletion_policy("supersede") == "supersede"

    def test_unknown_deletion_policy_raises(self) -> None:
        with pytest.raises(ValueError):
            classify_deletion_policy("delete_forever")
