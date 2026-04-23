"""Phase 6 idempotency tests for the sync tool.

Covers the UNCHANGED early-return path: when a checkpoint carries a scan_snapshot
and the re-scan produces identical content_hashes, sync must return
``{"status": "UNCHANGED"}`` without executing any memory operations.

Test strategy: build a minimal checkpoint payload with an embedded scan_snapshot
whose entries match the files in a temp directory, then call sync() and assert
the UNCHANGED response shape.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from workflows_mcp.engine.memory_onboard_sync_orchestrator import (
    compute_sync_delta,
)

# Register memory tools against the real MCP server instance so _get_sync_tool works.
from workflows_mcp.server import mcp as _mcp_server
from workflows_mcp.tools_memory import register_memory_tools

register_memory_tools(_mcp_server)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _entry(path: str, content: str, memory_id: str | None = None) -> dict[str, Any]:
    return {
        "path": path,
        "content_hash": _sha256(content),
        "memory_id": memory_id,
        "size_bytes": len(content.encode()),
        "mtime_ns": 0,
    }


def _minimal_checkpoint(
    scope: dict[str, Any],
    entries: list[dict[str, Any]],
    base_path: str,
    patterns: list[str] | None = None,
) -> dict[str, Any]:
    """Build the minimal checkpoint payload accepted by the sync tool."""
    # plan must be non-empty; a dummy ingest step satisfies validation.
    # The scan-delta re-scan path rebuilds the plan from scratch, so the plan
    # content here is irrelevant when all files are unchanged.
    dummy_plan = [{"operation": "ingest", "payload": {"records": []}}]
    return {
        "version": "oss-r3",
        "scope": scope,
        "plan": dummy_plan,
        "next_index": 0,
        "completed": [],
        "scan": {
            "root": base_path,
            "patterns": patterns or ["*.py"],
            "deletion_policy": "archive",
        },
        "scan_snapshot": {
            "entries": entries,
            "scan_config": {
                "root": base_path,
                "patterns": patterns or ["*.py"],
                "deletion_policy": "archive",
            },
        },
    }


def _get_sync_tool() -> Any:
    tool = _mcp_server._tool_manager._tools.get("sync")
    if tool is None:
        raise ValueError("sync tool not registered")
    return tool.fn


def _make_ctx() -> MagicMock:
    ctx = MagicMock()
    app_ctx = MagicMock()
    app_ctx.memory_backend = None
    app_ctx.memory_backend_lock = None
    ctx.request_context.lifespan_context = app_ctx
    exec_context = MagicMock()
    exec_context.user_string_id = None
    app_ctx.create_execution_context.return_value = exec_context
    app_ctx.get_user_context.return_value = (uuid.UUID(int=0), "test-user", "OS_USER")
    return ctx


# ---------------------------------------------------------------------------
# Unit tests: compute_sync_delta idempotency property
# ---------------------------------------------------------------------------


class TestSyncDeltaIdempotency:
    """compute_sync_delta must report zero semantic delta when inputs are identical."""

    def test_empty_snapshots_no_delta(self) -> None:
        delta = compute_sync_delta([], [])
        assert not delta.has_semantic_delta
        assert delta.total_files == 0

    def test_single_file_unchanged(self) -> None:
        e = {"path": "a.py", "content_hash": "abc123"}
        delta = compute_sync_delta([e], [e])
        assert not delta.has_semantic_delta
        assert delta.unchanged == ["a.py"]

    def test_multiple_files_all_unchanged(self) -> None:
        entries = [
            {"path": "a.py", "content_hash": "h1"},
            {"path": "b.py", "content_hash": "h2"},
            {"path": "c.py", "content_hash": "h3"},
        ]
        delta = compute_sync_delta(entries, entries)
        assert not delta.has_semantic_delta
        assert len(delta.unchanged) == 3
        assert delta.added == []
        assert delta.modified == []
        assert delta.deleted == []

    def test_one_modified_breaks_idempotency(self) -> None:
        prior = [{"path": "a.py", "content_hash": "h1"}]
        new = [{"path": "a.py", "content_hash": "h2"}]
        delta = compute_sync_delta(prior, new)
        assert delta.has_semantic_delta

    def test_one_added_breaks_idempotency(self) -> None:
        prior: list[dict[str, Any]] = []
        new = [{"path": "new.py", "content_hash": "h1"}]
        delta = compute_sync_delta(prior, new)
        assert delta.has_semantic_delta

    def test_one_deleted_breaks_idempotency(self) -> None:
        prior = [{"path": "gone.py", "content_hash": "h1"}]
        delta = compute_sync_delta(prior, [])
        assert delta.has_semantic_delta

    def test_to_debug_dict_has_semantic_delta_false(self) -> None:
        entries = [{"path": "x.py", "content_hash": "hx"}]
        delta = compute_sync_delta(entries, entries)
        d = delta.to_debug_dict()
        assert d["has_semantic_delta"] is False
        assert d["counts"]["unchanged"] == 1
        assert d["counts"]["added"] == 0
        assert d["counts"]["modified"] == 0
        assert d["counts"]["deleted"] == 0


# ---------------------------------------------------------------------------
# Integration tests: sync tool UNCHANGED early-return
# ---------------------------------------------------------------------------


class TestSyncToolUnchangedEarlyReturn:
    """When the re-scan produces identical hashes, sync must return UNCHANGED."""

    @pytest.mark.asyncio
    async def test_unchanged_status_returned(self, tmp_path: Path) -> None:
        content = "x = 1\n"
        (tmp_path / "a.py").write_text(content)
        checkpoint = _minimal_checkpoint(
            scope={"palace": "proj"},
            entries=[_entry("a.py", content)],
            base_path=str(tmp_path),
        )
        sync = _get_sync_tool()
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await sync(checkpoint=checkpoint, ctx=_make_ctx())
        payload = json.loads(result.content[0].text)
        assert payload.get("status") == "UNCHANGED", f"Expected UNCHANGED, got: {payload}"

    @pytest.mark.asyncio
    async def test_unchanged_has_no_error_key(self, tmp_path: Path) -> None:
        content = "y = 2\n"
        (tmp_path / "b.py").write_text(content)
        checkpoint = _minimal_checkpoint(
            scope={"palace": "p"},
            entries=[_entry("b.py", content)],
            base_path=str(tmp_path),
        )
        sync = _get_sync_tool()
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await sync(checkpoint=checkpoint, ctx=_make_ctx())
        payload = json.loads(result.content[0].text)
        assert "error" not in payload

    @pytest.mark.asyncio
    async def test_unchanged_includes_message(self, tmp_path: Path) -> None:
        content = "z = 3\n"
        (tmp_path / "c.py").write_text(content)
        checkpoint = _minimal_checkpoint(
            scope={"palace": "q"},
            entries=[_entry("c.py", content)],
            base_path=str(tmp_path),
        )
        sync = _get_sync_tool()
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await sync(checkpoint=checkpoint, ctx=_make_ctx())
        payload = json.loads(result.content[0].text)
        assert "message" in payload

    @pytest.mark.asyncio
    async def test_unchanged_debug_includes_delta(self, tmp_path: Path) -> None:
        content = "w = 4\n"
        (tmp_path / "d.py").write_text(content)
        checkpoint = _minimal_checkpoint(
            scope={"palace": "r"},
            entries=[_entry("d.py", content)],
            base_path=str(tmp_path),
        )
        sync = _get_sync_tool()
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await sync(
                checkpoint=checkpoint,
                response={"debug": True},
                ctx=_make_ctx(),
            )
        payload = json.loads(result.content[0].text)
        assert payload.get("status") == "UNCHANGED"
        assert "delta" in payload
        delta = payload["delta"]
        assert delta["has_semantic_delta"] is False

    @pytest.mark.asyncio
    async def test_modified_file_does_not_return_unchanged(self, tmp_path: Path) -> None:
        old_content = "original\n"
        new_content = "modified\n"
        (tmp_path / "e.py").write_text(new_content)
        checkpoint = _minimal_checkpoint(
            scope={"palace": "s"},
            entries=[_entry("e.py", old_content)],  # prior hash is for old content
            base_path=str(tmp_path),
        )
        sync = _get_sync_tool()
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await sync(checkpoint=checkpoint, ctx=_make_ctx())
        payload = json.loads(result.content[0].text)
        assert payload.get("status") != "UNCHANGED", f"Should not be UNCHANGED: {payload}"

    @pytest.mark.asyncio
    async def test_multiple_unchanged_files(self, tmp_path: Path) -> None:
        files = {"a.py": "a=1\n", "b.py": "b=2\n", "c.py": "c=3\n"}
        for name, content in files.items():
            (tmp_path / name).write_text(content)
        checkpoint = _minimal_checkpoint(
            scope={"palace": "multi"},
            entries=[_entry(name, content) for name, content in files.items()],
            base_path=str(tmp_path),
        )
        sync = _get_sync_tool()
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await sync(checkpoint=checkpoint, ctx=_make_ctx())
        payload = json.loads(result.content[0].text)
        assert payload.get("status") == "UNCHANGED"
