"""Compact-by-default response shape and checkpoint plan payload compaction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _helpers import _make_backend_mock, onboard, sync

from workflows_mcp.memory.memory_schema import ManageMemoryResult, MemoryResult

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Compact-by-default / debug expansion tests
# ---------------------------------------------------------------------------


class TestProjectFlowCompactResponse:
    """Verify that project_onboard/project_sync responses are compact by default.

    Default (no response arg or debug=False):
      - checkpoint.completed[].result blobs are stripped
      - checkpoint.scan_snapshot.entries are stripped (summary count only)
      - checkpoint.plan[].payload values are preserved (needed for resume)
      - outer `results` list is absent (replaced by summary counts)
      - status, last_operation, completed_operations, remaining_operations present

    debug=True:
      - full checkpoint with completed[].result blobs present
      - full scan_snapshot entries present
      - outer `results` list present with full result payloads
    """

    _BASE_PLAN = [
        {"operation": "ingest", "payload": {"content": "hello"}},
        {"operation": "archive", "payload": {"ids": ["m-old"]}},
    ]
    _BASE_SCOPE: dict[str, Any] = {"palace": "compacttest"}

    @staticmethod
    def _make_ingest_result() -> MemoryResult:
        return MemoryResult(
            operation="ingest",
            manage=ManageMemoryResult(
                operation="store",
                memory_ids=["m-new"],
                stored_count=1,
            ),
        )

    @staticmethod
    def _make_archive_result() -> MemoryResult:
        return MemoryResult(
            operation="archive",
            manage=ManageMemoryResult(operation="forget", archived_count=2),
        )

    @pytest.mark.asyncio
    async def test_onboard_checkpoint_response_default_strips_completed_results(
        self, mock_ctx: MagicMock
    ) -> None:
        """Default response: checkpoint.completed[].result must be absent."""
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(
                side_effect=lambda req: (
                    self._make_ingest_result()
                    if req.operation == "ingest"
                    else self._make_archive_result()
                )
            )
            result = await onboard(
                scope=self._BASE_SCOPE,
                ingest={"content": "hello"},
                archive={"ids": ["m-old"]},
                max_operations=1,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "checkpoint"
        completed_items = payload["checkpoint"]["completed"]
        for item in completed_items:
            assert "result" not in item, (
                f"Default response must strip result from completed items; got: {item}"
            )

    @pytest.mark.asyncio
    async def test_onboard_checkpoint_response_debug_includes_completed_results(
        self, mock_ctx: MagicMock
    ) -> None:
        """debug=True response: checkpoint.completed[].result must be present."""
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(
                side_effect=lambda req: (
                    self._make_ingest_result()
                    if req.operation == "ingest"
                    else self._make_archive_result()
                )
            )
            result = await onboard(
                scope=self._BASE_SCOPE,
                ingest={"content": "hello"},
                archive={"ids": ["m-old"]},
                max_operations=1,
                response={"debug": True},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "checkpoint"
        completed_items = payload["checkpoint"]["completed"]
        assert len(completed_items) == 1
        assert "result" in completed_items[0], (
            f"debug=True must include result in completed items; got: {completed_items[0]}"
        )

    @pytest.mark.asyncio
    async def test_onboard_completed_default_omits_results_list(self, mock_ctx: MagicMock) -> None:
        """Default response: completed status must not include a `results` list."""
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(return_value=self._make_ingest_result())
            result = await onboard(
                scope=self._BASE_SCOPE,
                ingest={"content": "hello"},
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        assert "results" not in payload, (
            f"Default response must not include `results` list; got keys: {list(payload)}"
        )
        assert "completed_operations" in payload

    @pytest.mark.asyncio
    async def test_onboard_completed_debug_includes_results_list(self, mock_ctx: MagicMock) -> None:
        """debug=True: completed status must include full `results` list."""
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(return_value=self._make_ingest_result())
            result = await onboard(
                scope=self._BASE_SCOPE,
                ingest={"content": "hello"},
                max_operations=5,
                response={"debug": True},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        assert "results" in payload, (
            f"debug=True must include `results` list; got keys: {list(payload)}"
        )
        assert isinstance(payload["results"], list)

    @pytest.mark.asyncio
    async def test_sync_completed_default_omits_results_list(self, mock_ctx: MagicMock) -> None:
        """Default response: project_sync completed must not include a `results` list."""
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(return_value=self._make_ingest_result())
            result = await sync(
                scope=self._BASE_SCOPE,
                ingest={"content": "hello"},
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        assert "results" not in payload, (
            f"Default project_sync must not include `results` list; got keys: {list(payload)}"
        )

    @pytest.mark.asyncio
    async def test_sync_completed_debug_includes_results_list(self, mock_ctx: MagicMock) -> None:
        """debug=True: project_sync completed must include full `results` list."""
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(return_value=self._make_ingest_result())
            result = await sync(
                scope=self._BASE_SCOPE,
                ingest={"content": "hello"},
                max_operations=5,
                response={"debug": True},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        assert "results" in payload, (
            f"debug=True project_sync must include `results` list; got keys: {list(payload)}"
        )

    @pytest.mark.asyncio
    async def test_checkpoint_plan_payloads_preserved_for_resume(self, mock_ctx: MagicMock) -> None:
        """Compact checkpoint must keep plan[].payload intact so resume works."""
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(return_value=self._make_ingest_result())
            result = await onboard(
                scope=self._BASE_SCOPE,
                ingest={"content": "hello"},
                archive={"ids": ["m-old"]},
                max_operations=1,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "checkpoint"
        plan = payload["checkpoint"]["plan"]
        assert len(plan) == 2  # noqa: PLR2004
        for step in plan:
            assert "payload" in step, (
                f"Compact checkpoint must preserve plan[].payload; got: {step}"
            )

    @pytest.mark.asyncio
    async def test_compact_checkpoint_is_resumable(self, mock_ctx: MagicMock) -> None:
        """The compact checkpoint returned by default must be accepted for resume."""
        backend_mock = _make_backend_mock()
        execute_results = [self._make_ingest_result(), self._make_archive_result()]
        call_index = 0

        async def _side_effect(_req: Any) -> MemoryResult:
            nonlocal call_index
            r = execute_results[call_index % len(execute_results)]
            call_index += 1
            return r

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(side_effect=_side_effect)

            # Step 1: get compact checkpoint from onboard
            r1 = await onboard(
                scope=self._BASE_SCOPE,
                ingest={"content": "hello"},
                archive={"ids": ["m-old"]},
                max_operations=1,
                ctx=mock_ctx,
            )
            p1 = json.loads(r1.content[0].text)
            assert p1["status"] == "checkpoint"

            # Step 2: resume with the compact checkpoint
            r2 = await sync(
                checkpoint=p1["checkpoint"],
                max_operations=5,
                ctx=mock_ctx,
            )
            p2 = json.loads(r2.content[0].text)
            assert p2["status"] == "completed", f"Compact checkpoint must be resumable; got: {p2}"
            assert "archive" in p2["completed_operations"]

    @pytest.mark.asyncio
    async def test_scan_snapshot_always_includes_entries_for_resume(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """scan_snapshot.entries must be preserved in checkpoint even in compact mode.

        Entries are required by project_sync for delta computation; stripping
        them would break resume functionality for scan-based flows.
        """
        py_file = workspace_tmp / "sample.py"
        py_file.write_text("x = 1\n")

        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(return_value=self._make_ingest_result())
            result = await onboard(
                scope=self._BASE_SCOPE,
                scan={"patterns": ["*.py"], "root": str(workspace_tmp)},
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        snap = payload["checkpoint"].get("scan_snapshot")
        assert snap is not None, "scan_snapshot must be present in checkpoint"
        # entries must be present (needed for project_sync delta computation)
        assert "entries" in snap, (
            f"scan_snapshot.entries must be preserved for resume; got keys: {list(snap)}"
        )

    @pytest.mark.asyncio
    async def test_scan_snapshot_debug_includes_full_entries(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """debug=True: scan_snapshot in checkpoint must include full entries list."""
        py_file = workspace_tmp / "sample.py"
        py_file.write_text("x = 1\n")

        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(return_value=self._make_ingest_result())
            result = await onboard(
                scope=self._BASE_SCOPE,
                scan={"patterns": ["*.py"], "root": str(workspace_tmp)},
                max_operations=5,
                response={"debug": True},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        snap = payload["checkpoint"].get("scan_snapshot")
        assert snap is not None
        assert "entries" in snap, (
            f"debug=True must include full scan_snapshot.entries; got keys: {list(snap)}"
        )

    @pytest.mark.asyncio
    async def test_sync_fast_path_compact_default_omits_results_list(
        self, mock_ctx: MagicMock
    ) -> None:
        """Default: project_sync fast-path (completed checkpoint) must not include results."""
        completed_checkpoint: dict[str, Any] = {
            "version": "oss-r3",
            "scope": {"palace": "compacttest"},
            "plan": [{"operation": "ingest", "payload": {"content": "data"}}],
            "next_index": 1,
            "completed": [{"operation": "ingest", "result": {"stored": 1}}],
        }
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(
                side_effect=AssertionError("must not execute on fast-path")
            )
            result = await sync(
                checkpoint=completed_checkpoint,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        assert payload.get("from_checkpoint") is True
        assert "results" not in payload, (
            f"Default fast-path must not include results list; got keys: {list(payload)}"
        )

    @pytest.mark.asyncio
    async def test_sync_fast_path_debug_includes_results_list(self, mock_ctx: MagicMock) -> None:
        """debug=True: project_sync fast-path must include results list."""
        completed_checkpoint: dict[str, Any] = {
            "version": "oss-r3",
            "scope": {"palace": "compacttest"},
            "plan": [{"operation": "ingest", "payload": {"content": "data"}}],
            "next_index": 1,
            "completed": [{"operation": "ingest", "result": {"stored": 1}}],
        }
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(
                side_effect=AssertionError("must not execute on fast-path")
            )
            result = await sync(
                checkpoint=completed_checkpoint,
                response={"debug": True},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        assert payload.get("from_checkpoint") is True
        assert "results" in payload, (
            f"debug=True fast-path must include results list; got keys: {list(payload)}"
        )


class TestCheckpointPlanPayloadCompaction:
    """Verify compact checkpoint strips plan[].payload for already-completed steps.

    Requirement (A): default checkpoint should not include full memory content in
    completed plan steps' payloads — only pending (not-yet-executed) step payloads
    need to be preserved for resume.

    Tests:
    1. Default onboard/sync response does not contain full plan payload content
       for completed steps.
    2. debug=true includes full plan payload content for all steps.
    3. Default compact checkpoint from onboard can be passed to sync and resume works.
    4. Backward compatibility: sync accepts legacy full checkpoint shape.
    5. No replay regression: pending steps keep their payloads; resume executes them.
    """

    _SCOPE: dict[str, Any] = {"palace": "payloadtest"}
    _LARGE_CONTENT = "x" * 500  # Simulates a large memory body

    @staticmethod
    def _make_ingest_result() -> MemoryResult:
        return MemoryResult(
            operation="ingest",
            manage=ManageMemoryResult(
                operation="store",
                memory_ids=["m-1"],
                stored_count=1,
            ),
        )

    @staticmethod
    def _make_archive_result() -> MemoryResult:
        return MemoryResult(
            operation="archive",
            manage=ManageMemoryResult(operation="forget", archived_count=1),
        )

    @pytest.mark.asyncio
    async def test_default_compact_checkpoint_strips_completed_plan_payload_content(
        self, mock_ctx: MagicMock
    ) -> None:
        """Default (no debug): plan[i].payload.memories[].content must be absent for
        completed steps (those at index < next_index)."""
        backend_mock = _make_backend_mock()
        large_memories = [{"content": self._LARGE_CONTENT}]
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(return_value=self._make_ingest_result())
            # Two steps; max_operations=1 → only ingest runs, archive is pending
            result = await onboard(
                scope=self._SCOPE,
                ingest={"format": "structured", "memories": large_memories},
                archive={"ids": ["m-old"]},
                max_operations=1,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "checkpoint"
        next_index = payload["checkpoint"]["next_index"]
        plan = payload["checkpoint"]["plan"]
        assert next_index == 1  # ingest completed

        # The completed step (index 0) must NOT contain heavy memories content
        completed_step_payload = plan[0].get("payload", {})
        memories_in_plan = completed_step_payload.get("memories", [])
        for mem in memories_in_plan:
            assert "content" not in mem, (
                "Compact checkpoint must strip content from completed plan step payloads; "
                f"got memory with content: {mem!r}"
            )

        # The pending step (index 1) must keep its payload intact for resume
        pending_step = plan[1]
        assert pending_step.get("payload") is not None

    @pytest.mark.asyncio
    async def test_debug_true_includes_full_plan_payload_content(self, mock_ctx: MagicMock) -> None:
        """debug=True: plan[i].payload must be present and intact for all steps."""
        backend_mock = _make_backend_mock()
        large_memories = [{"content": self._LARGE_CONTENT}]
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(return_value=self._make_ingest_result())
            result = await onboard(
                scope=self._SCOPE,
                ingest={"format": "structured", "memories": large_memories},
                archive={"ids": ["m-old"]},
                max_operations=1,
                response={"debug": True},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "checkpoint"
        plan = payload["checkpoint"]["plan"]

        # In debug mode the completed step payload must still carry memories with content
        completed_step_payload = plan[0].get("payload", {})
        memories_in_plan = completed_step_payload.get("memories", [])
        assert len(memories_in_plan) == 1
        assert memories_in_plan[0].get("content") == self._LARGE_CONTENT, (
            "debug=True must include full content in completed plan step payloads"
        )

    @pytest.mark.asyncio
    async def test_compact_checkpoint_from_onboard_is_resumable_by_sync(
        self, mock_ctx: MagicMock
    ) -> None:
        """The compact checkpoint returned by default onboard must be accepted by
        project_sync and allow the remaining steps to execute."""
        backend_mock = _make_backend_mock()
        call_order: list[str] = []

        async def _side_effect(req: Any) -> MemoryResult:
            call_order.append(req.operation)
            if req.operation == "ingest":
                return self._make_ingest_result()
            return self._make_archive_result()

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(side_effect=_side_effect)
            large_memories = [{"content": self._LARGE_CONTENT}]

            # Phase 1: onboard runs ingest, returns compact checkpoint
            r1 = await onboard(
                scope=self._SCOPE,
                ingest={"format": "structured", "memories": large_memories},
                archive={"ids": ["m-old"]},
                max_operations=1,
                ctx=mock_ctx,
            )
            p1 = json.loads(r1.content[0].text)
            assert p1["status"] == "checkpoint"
            compact_checkpoint = p1["checkpoint"]

            # Phase 2: sync resumes from compact checkpoint, executes archive
            r2 = await sync(
                checkpoint=compact_checkpoint,
                max_operations=5,
                ctx=mock_ctx,
            )
            p2 = json.loads(r2.content[0].text)

        assert p2["status"] == "completed", (
            f"Compact checkpoint must allow sync to complete; got: {p2}"
        )
        assert "archive" in p2["completed_operations"], (
            f"Sync must execute the pending archive step; got: {p2['completed_operations']}"
        )
        # ingest ran once (in onboard), archive once (in sync)
        assert call_order == ["ingest", "archive"]

    @pytest.mark.asyncio
    async def test_backward_compat_sync_accepts_legacy_full_checkpoint(
        self, mock_ctx: MagicMock
    ) -> None:
        """project_sync must accept a legacy checkpoint where plan[].payload carries
        full memories content (old format without compaction)."""
        backend_mock = _make_backend_mock()
        large_memories = [{"content": self._LARGE_CONTENT, "metadata": {"path": "a.py"}}]
        full_checkpoint: dict[str, Any] = {
            "version": "oss-r3",
            "scope": self._SCOPE,
            "plan": [
                {
                    "operation": "ingest",
                    "payload": {"format": "structured", "memories": large_memories},
                },
                {"operation": "archive", "payload": {"ids": ["m-old"]}},
            ],
            "next_index": 1,
            "completed": [
                # Legacy full format with result included
                {"operation": "ingest", "result": {"stored": 1, "id": "m-1"}},
            ],
        }
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(return_value=self._make_archive_result())
            result = await sync(
                checkpoint=full_checkpoint,
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed", (
            f"Sync must accept legacy full checkpoint; got: {payload}"
        )
        assert "archive" in payload["completed_operations"]

    @pytest.mark.asyncio
    async def test_no_replay_regression_pending_steps_keep_payload(
        self, mock_ctx: MagicMock
    ) -> None:
        """Pending steps (index >= next_index) must preserve their full payload
        so they can be executed on resume — no payload stripping on pending steps."""
        backend_mock = _make_backend_mock()
        archive_ids = ["m-a", "m-b", "m-c"]
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            executed_payloads: list[Any] = []

            async def _capture(req: Any) -> MemoryResult:
                executed_payloads.append(req.record)
                return self._make_ingest_result()

            mock_svc.return_value.execute = AsyncMock(side_effect=_capture)

            # Two steps: ingest then archive; run only first (ingest)
            r1 = await onboard(
                scope=self._SCOPE,
                ingest={"format": "structured", "memories": [{"content": "data"}]},
                archive={"ids": archive_ids},
                max_operations=1,
                ctx=mock_ctx,
            )
            p1 = json.loads(r1.content[0].text)
            assert p1["status"] == "checkpoint"
            compact_cp = p1["checkpoint"]

            # Verify pending archive step still has its payload
            plan = compact_cp["plan"]
            archive_step = next(s for s in plan if s["operation"] == "archive")
            assert archive_step.get("payload", {}).get("ids") == archive_ids, (
                f"Pending archive step must keep its ids payload; got: {archive_step}"
            )
