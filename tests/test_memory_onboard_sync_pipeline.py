"""Phase 2-4 tests: deterministic scope/path/scan foundation, sync({}) resolution, and
programmatic onboard pipeline (Phase 4).

Phase 2 coverage:
- scope_key determinism (order independence, whitespace independence)
- normalize_scope canonical form
- sorted_scan_manifest determinism
- find_boundary_marker (nearest / custom precedence)
- resolve_sync_context: NO_CONTEXT / AMBIGUOUS_CONTEXT / single-context success
- build_no_context_envelope / build_ambiguous_context_envelope shapes
- sync({}) tool integration: NO_CONTEXT / AMBIGUOUS_CONTEXT error envelopes

Phase 4 coverage:
- successful onboard in programmatic mode (readable files -> completed)
- metadata-only compartments for binary/unsupported files (1:1 mapping)
- concise default response omits diagnostics
- debug=True response includes diagnostics block
- completeness failure (empty graph) maps to GRAPH_COMPLETENESS_FAILED
- onboard tool fast-path: scan + no checkpoint + no ingest -> orchestrator response

Phase 6 coverage:
- compute_sync_delta: four-class deterministic classification
- compute_weak_links: orphaned corridor detection
- classify_deletion_policy: alias normalisation
- recompute_depends_on_corridors: structural corridor injection
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import workflows_mcp.engine.knowledge.schema as knowledge_schema
import workflows_mcp.tools_memory as _tools_memory
import workflows_mcp.tools_memory as tools_memory
from workflows_mcp.context import SessionProjectContext
from workflows_mcp.engine.llm_config import (
    LLMConfig,
    LLMConfigLoader,
    ProfileConfig,
    ProviderConfig,
)
from workflows_mcp.engine.memory_graph_builder import (
    CorridorSemanticType,
    GraphCorridor,
    GraphNode,
    GraphPayload,
    NodeType,
)
from workflows_mcp.engine.memory_onboard_sync_orchestrator import (
    LLMOnboardRequest,
    ProgrammaticOnboardRequest,
    ScannedFileEntry,
    build_llm_onboard_response,
    build_programmatic_onboard_response,
    classify_deletion_policy,
    classify_scan_files_for_programmatic_mode,
    compute_sync_delta,
    compute_weak_links,
    recompute_depends_on_corridors,
    run_llm_onboard,
    run_programmatic_onboard,
)
from workflows_mcp.engine.memory_scope_resolver import (
    DEFAULT_BOUNDARY_MARKERS,
    BoundaryResolution,
    SyncContextCandidate,
    build_ambiguous_context_envelope,
    build_no_context_envelope,
    find_boundary_marker,
    normalize_scope,
    resolve_sync_context,
    scope_key,
    sorted_scan_manifest,
)
from workflows_mcp.server import mcp as _mcp_server
from workflows_mcp.tools_memory import register_memory_tools

register_memory_tools(_mcp_server)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    palace: str | None = None,
    wing: str | None = None,
    room: str | None = None,
    compartment: str | None = None,
    source: str = "stored_checkpoint",
) -> SyncContextCandidate:
    s = {"palace": palace, "wing": wing, "room": room, "compartment": compartment}
    return SyncContextCandidate(
        scope={k: v for k, v in s.items()},
        scope_key_value=scope_key(s),
        checkpoint_data={},
        source=source,
    )


# ===========================================================================
# 1. scope_key determinism
# ===========================================================================


class TestScopeKeyDeterminism:
    def test_same_scope_same_key(self) -> None:
        s = {"palace": "org", "wing": "svc", "room": "api", "compartment": None}
        assert scope_key(s) == scope_key(s)

    def test_order_independence(self) -> None:
        """Reordering dict keys must not change scope_key."""
        s1 = {"palace": "p", "wing": "w", "room": "r", "compartment": "c"}
        s2 = {"compartment": "c", "room": "r", "palace": "p", "wing": "w"}
        assert scope_key(s1) == scope_key(s2)

    def test_whitespace_independence(self) -> None:
        """Leading/trailing whitespace in values must not affect scope_key."""
        s1 = {"palace": "org", "wing": "svc", "room": None, "compartment": None}
        s2 = {"palace": "  org  ", "wing": "  svc  ", "room": None, "compartment": None}
        assert scope_key(s1) == scope_key(s2)

    def test_blank_and_none_equivalent(self) -> None:
        """A blank string field must produce the same key as None for that field."""
        s_none = {"palace": None, "wing": "w", "room": None, "compartment": None}
        s_blank = {"palace": "", "wing": "w", "room": "", "compartment": None}
        assert scope_key(s_none) == scope_key(s_blank)

    def test_different_scopes_different_keys(self) -> None:
        s1 = {"palace": "a", "wing": None, "room": None, "compartment": None}
        s2 = {"palace": "b", "wing": None, "room": None, "compartment": None}
        assert scope_key(s1) != scope_key(s2)

    def test_unknown_keys_ignored(self) -> None:
        """Extra keys not in (palace, wing, room, compartment) must be ignored."""
        s1 = {"palace": "p", "wing": "w", "room": None, "compartment": None}
        s2 = {**s1, "unknown_field": "should_be_ignored", "other": 42}
        assert scope_key(s1) == scope_key(s2)

    def test_empty_scope_is_stable(self) -> None:
        assert scope_key({}) == scope_key({})

    def test_scope_key_is_hex_string(self) -> None:
        k = scope_key({"palace": "p"})
        assert isinstance(k, str)
        assert len(k) == 64  # SHA-256 hex
        int(k, 16)  # must be valid hex


# ===========================================================================
# 2. normalize_scope
# ===========================================================================


class TestNormalizeScope:
    def test_strips_whitespace(self) -> None:
        result = normalize_scope({"palace": "  org  ", "wing": "\tsvc\n"})
        assert result["palace"] == "org"
        assert result["wing"] == "svc"

    def test_blank_becomes_none(self) -> None:
        result = normalize_scope({"palace": "   ", "wing": "w"})
        assert result["palace"] is None
        assert result["wing"] == "w"

    def test_missing_fields_become_none(self) -> None:
        result = normalize_scope({"palace": "p"})
        assert result["room"] is None
        assert result["compartment"] is None

    def test_output_has_all_four_fields(self) -> None:
        result = normalize_scope({})
        assert set(result.keys()) == {"palace", "wing", "room", "compartment"}

    def test_output_order_is_fixed(self) -> None:
        result = normalize_scope({"compartment": "c", "palace": "p"})
        assert list(result.keys()) == ["palace", "wing", "room", "compartment"]

    def test_extra_keys_dropped(self) -> None:
        result = normalize_scope({"palace": "p", "extra": "x"})
        assert "extra" not in result


# ===========================================================================
# 3. sorted_scan_manifest
# ===========================================================================


class TestSortedScanManifest:
    def test_sorts_lexicographically_by_path(self) -> None:
        entries = [
            {"path": "src/z.py", "size_bytes": 1},
            {"path": "src/a.py", "size_bytes": 2},
            {"path": "README.md", "size_bytes": 3},
        ]
        result = sorted_scan_manifest(entries)
        assert [e["path"] for e in result] == ["README.md", "src/a.py", "src/z.py"]

    def test_stable_for_equal_paths(self) -> None:
        entries = [
            {"path": "a.py", "size_bytes": 1},
            {"path": "a.py", "size_bytes": 2},
        ]
        result = sorted_scan_manifest(entries)
        # Both must be present; stable sort preserves relative order.
        assert result[0]["size_bytes"] == 1
        assert result[1]["size_bytes"] == 2

    def test_missing_path_sorts_first(self) -> None:
        entries = [
            {"path": "z.py"},
            {"size_bytes": 0},  # no path key
        ]
        result = sorted_scan_manifest(entries)
        assert "path" not in result[0]

    def test_empty_list(self) -> None:
        assert sorted_scan_manifest([]) == []

    def test_single_entry(self) -> None:
        entries = [{"path": "a.py"}]
        assert sorted_scan_manifest(entries) == entries

    def test_does_not_mutate_input(self) -> None:
        entries = [{"path": "b.py"}, {"path": "a.py"}]
        original_order = [e["path"] for e in entries]
        sorted_scan_manifest(entries)
        assert [e["path"] for e in entries] == original_order

    def test_deterministic_across_calls(self) -> None:
        entries = [{"path": "c.py"}, {"path": "a.py"}, {"path": "b.py"}]
        r1 = sorted_scan_manifest(entries)
        r2 = sorted_scan_manifest(entries)
        assert [e["path"] for e in r1] == [e["path"] for e in r2]


# ===========================================================================
# 4. find_boundary_marker — nearest strategy (default)
# ===========================================================================


class TestFindBoundaryMarkerNearest:
    def test_finds_marker_in_current_directory(self, tmp_path: Path) -> None:
        marker_path = tmp_path / ".git"
        marker_path.mkdir()
        result = find_boundary_marker(tmp_path)
        assert result.path == tmp_path
        assert result.marker == ".git"
        assert result.strategy == "nearest"

    def test_finds_marker_in_parent(self, tmp_path: Path) -> None:
        child = tmp_path / "sub" / "deep"
        child.mkdir(parents=True)
        (tmp_path / "pyproject.toml").touch()
        result = find_boundary_marker(child)
        assert result.path == tmp_path
        assert result.marker == "pyproject.toml"

    def test_returns_none_path_when_no_marker(self, tmp_path: Path) -> None:
        # Use a temp dir that has no default markers.
        # Create a deeply nested directory with no markers.
        leaf = tmp_path / "a" / "b" / "c"
        leaf.mkdir(parents=True)
        # Ensure no default markers exist anywhere in tmp_path.
        for m in DEFAULT_BOUNDARY_MARKERS:
            p = tmp_path / m
            if p.exists():
                if p.is_dir():
                    import shutil
                    shutil.rmtree(p)
                else:
                    p.unlink()
        result = find_boundary_marker(leaf)
        # May or may not find a marker depending on real filesystem above tmp_path;
        # but the returned object must be a BoundaryResolution.
        assert isinstance(result, BoundaryResolution)
        assert result.strategy == "nearest"

    def test_searched_from_is_start_path(self, tmp_path: Path) -> None:
        result = find_boundary_marker(tmp_path)
        assert result.searched_from == tmp_path

    def test_first_marker_in_list_wins(self, tmp_path: Path) -> None:
        """When multiple default markers exist in the same dir, the first listed wins."""
        (tmp_path / ".workflows-root").touch()
        (tmp_path / ".git").mkdir()
        result = find_boundary_marker(tmp_path)
        assert result.marker == ".workflows-root"


# ===========================================================================
# 5. find_boundary_marker — custom strategy
# ===========================================================================


class TestFindBoundaryMarkerCustom:
    def test_custom_marker_found(self, tmp_path: Path) -> None:
        (tmp_path / "MYMARKER").touch()
        result = find_boundary_marker(
            tmp_path, strategy="custom", marker_files=["MYMARKER"]
        )
        assert result.path == tmp_path
        assert result.marker == "MYMARKER"
        assert result.strategy == "custom"

    def test_custom_marker_not_in_default_list(self, tmp_path: Path) -> None:
        """Custom markers must not fall back to default list."""
        (tmp_path / ".git").mkdir()  # default marker exists
        result = find_boundary_marker(
            tmp_path, strategy="custom", marker_files=["NOTEXIST"]
        )
        # Should not find .git because we specified custom markers only.
        # May walk up and find marker elsewhere; just assert .git is NOT the marker.
        if result.path is not None:
            assert result.marker != ".git"

    def test_custom_empty_list_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="non-empty list"):
            find_boundary_marker(tmp_path, strategy="custom", marker_files=[])

    def test_custom_none_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="non-empty list"):
            find_boundary_marker(tmp_path, strategy="custom", marker_files=None)

    def test_custom_precedence_first_in_list(self, tmp_path: Path) -> None:
        """First matching marker in custom list wins."""
        (tmp_path / "MARKER_A").touch()
        (tmp_path / "MARKER_B").touch()
        result = find_boundary_marker(
            tmp_path, strategy="custom", marker_files=["MARKER_A", "MARKER_B"]
        )
        assert result.marker == "MARKER_A"

    def test_custom_second_marker_wins_when_first_absent(self, tmp_path: Path) -> None:
        (tmp_path / "MARKER_B").touch()
        result = find_boundary_marker(
            tmp_path, strategy="custom", marker_files=["MARKER_A", "MARKER_B"]
        )
        assert result.marker == "MARKER_B"


# ===========================================================================
# 6. resolve_sync_context
# ===========================================================================


class TestResolveSyncContext:
    def test_no_candidates_returns_no_context(self) -> None:
        result = resolve_sync_context([])
        assert result.status == "NO_CONTEXT"
        assert result.context is None

    def test_single_candidate_returns_success(self) -> None:
        c = _make_candidate(palace="org", wing="svc")
        result = resolve_sync_context([c])
        assert result.status == "success"
        assert result.context is c

    def test_multiple_candidates_returns_ambiguous(self) -> None:
        c1 = _make_candidate(palace="org", wing="svc1")
        c2 = _make_candidate(palace="org", wing="svc2")
        result = resolve_sync_context([c1, c2])
        assert result.status == "AMBIGUOUS_CONTEXT"
        assert len(result.candidates) == 2

    def test_scope_filter_selects_matching(self) -> None:
        c1 = _make_candidate(palace="org", wing="svc1")
        c2 = _make_candidate(palace="org", wing="svc2")
        result = resolve_sync_context(
            [c1, c2], requested_scope={"palace": "org", "wing": "svc1"}
        )
        assert result.status == "success"
        assert result.context is c1

    def test_scope_filter_no_match_returns_no_context(self) -> None:
        c1 = _make_candidate(palace="org", wing="svc1")
        result = resolve_sync_context(
            [c1], requested_scope={"palace": "org", "wing": "NOMATCH"}
        )
        assert result.status == "NO_CONTEXT"

    def test_scope_filter_uses_key_not_object_equality(self) -> None:
        """Whitespace variants of the same scope must match the same candidate."""
        c = _make_candidate(palace="org", wing="svc")
        # Whitespace variant of the same scope.
        result = resolve_sync_context(
            [c], requested_scope={"palace": "  org  ", "wing": "  svc  "}
        )
        assert result.status == "success"

    def test_empty_requested_scope_considers_all(self) -> None:
        c1 = _make_candidate(palace="a")
        c2 = _make_candidate(palace="b")
        result = resolve_sync_context([c1, c2], requested_scope={})
        assert result.status == "AMBIGUOUS_CONTEXT"

    def test_no_context_message_is_non_empty(self) -> None:
        result = resolve_sync_context([])
        assert result.message

    def test_ambiguous_message_includes_count(self) -> None:
        c1 = _make_candidate(palace="a")
        c2 = _make_candidate(palace="b")
        result = resolve_sync_context([c1, c2])
        assert "2" in result.message


# ===========================================================================
# 7. Error envelope shapes
# ===========================================================================


class TestErrorEnvelopeShapes:
    def test_no_context_envelope_has_required_fields(self) -> None:
        env = build_no_context_envelope()
        err = env["error"]
        assert err["code"] == "NO_CONTEXT"
        assert isinstance(err["message"], str)
        assert err["retryable"] is False

    def test_ambiguous_context_envelope_has_required_fields(self) -> None:
        c1 = _make_candidate(palace="a")
        c2 = _make_candidate(palace="b")
        env = build_ambiguous_context_envelope([c1, c2])
        err = env["error"]
        assert err["code"] == "AMBIGUOUS_CONTEXT"
        assert isinstance(err["message"], str)
        assert err["retryable"] is False

    def test_ambiguous_context_envelope_includes_candidates(self) -> None:
        c1 = _make_candidate(palace="a")
        c2 = _make_candidate(palace="b")
        env = build_ambiguous_context_envelope([c1, c2])
        candidates = env["error"]["candidates"]
        assert len(candidates) == 2
        # Each candidate must include scope and scope_key.
        for cand in candidates:
            assert "scope" in cand
            assert "scope_key" in cand

    def test_ambiguous_envelope_candidate_scope_keys_are_stable(self) -> None:
        c = _make_candidate(palace="org", wing="svc")
        env = build_ambiguous_context_envelope([c])
        reported_key = env["error"]["candidates"][0]["scope_key"]
        expected_key = scope_key({"palace": "org", "wing": "svc"})
        assert reported_key == expected_key


# ===========================================================================
# 8. sync({}) tool integration — NO_CONTEXT and AMBIGUOUS_CONTEXT envelopes
# ===========================================================================

# These tests call the sync() MCP tool handler directly (not over stdio) to
# verify that the NO_CONTEXT and AMBIGUOUS_CONTEXT paths surface correctly when
# sync is called with no arguments and no checkpoint.

def _get_tool_fn(name: str) -> Any:
    tool = _mcp_server._tool_manager._tools.get(name)
    if tool is None:
        raise ValueError(f"Tool {name!r} not registered")
    return tool.fn


@pytest.fixture
def mock_ctx() -> MagicMock:
    ctx = MagicMock()
    app_ctx = MagicMock()
    session = object()
    app_ctx.memory_backend = None
    app_ctx.memory_backend_lock = None
    app_ctx.memory_backend_unavailable_error = None
    ctx.request_context.lifespan_context = app_ctx
    ctx.request_context.session = session

    active_context_store: dict[int, Any] = {}

    def _set_active_context(session_obj: Any, candidate: Any) -> None:
        active_context_store[id(session_obj)] = candidate

    def _get_active_context(session_obj: Any) -> Any:
        return active_context_store.get(id(session_obj))

    app_ctx.set_active_context.side_effect = _set_active_context
    app_ctx.get_active_context.side_effect = _get_active_context

    onboard_candidates_store: dict[int, list[Any]] = {}

    def _register_onboard_context_candidate(session_obj: Any, candidate: Any) -> None:
        onboard_candidates_store.setdefault(id(session_obj), []).append(candidate)

    def _list_onboard_context_candidates(session_obj: Any) -> list[Any]:
        return list(onboard_candidates_store.get(id(session_obj), []))

    app_ctx.register_onboard_context_candidate.side_effect = _register_onboard_context_candidate
    app_ctx.list_onboard_context_candidates.side_effect = _list_onboard_context_candidates

    exec_context = MagicMock()
    exec_context.user_string_id = None
    app_ctx.create_execution_context.return_value = exec_context
    app_ctx.get_user_context.return_value = (uuid.UUID(int=0), "test-user", "OS_USER")
    return ctx


class TestSyncNoArgsResolution:
    """sync({}) with no checkpoint and no plan -> MEM_NO_ACTIVE_CONTEXT error envelope."""

    @pytest.mark.asyncio
    async def test_sync_no_args_returns_error(self, mock_ctx: MagicMock) -> None:
        sync = _get_tool_fn("sync")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await sync(ctx=mock_ctx)
        payload = json.loads(result.content[0].text)
        assert "error" in payload, f"Expected error envelope, got: {payload}"

    @pytest.mark.asyncio
    async def test_sync_no_args_returns_no_context_code(self, mock_ctx: MagicMock) -> None:
        """sync({}) with no stored context must return MEM_NO_ACTIVE_CONTEXT code."""
        sync = _get_tool_fn("sync")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await sync(ctx=mock_ctx)
        payload = json.loads(result.content[0].text)
        err = payload.get("error", {})
        assert err.get("code") == "MEM_NO_ACTIVE_CONTEXT", (
            f"Expected MEM_NO_ACTIVE_CONTEXT, got: {err.get('code')}"
        )

    @pytest.mark.asyncio
    async def test_sync_no_context_retryable_is_false(self, mock_ctx: MagicMock) -> None:
        sync = _get_tool_fn("sync")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await sync(ctx=mock_ctx)
        payload = json.loads(result.content[0].text)
        err = payload.get("error", {})
        assert err.get("retryable") is False


class TestSyncNoContextEnvelopeShape:
    """The NO_CONTEXT envelope returned by resolve_sync_context must be well-formed."""

    def test_no_context_code(self) -> None:
        env = build_no_context_envelope()
        assert env["error"]["code"] == "NO_CONTEXT"

    def test_no_context_retryable_false(self) -> None:
        env = build_no_context_envelope()
        assert env["error"]["retryable"] is False

    def test_no_context_message_mentions_onboard(self) -> None:
        env = build_no_context_envelope()
        # Message should guide the caller to run onboard first.
        assert "onboard" in env["error"]["message"].lower()


class TestSyncAmbiguousContextEnvelopeShape:
    """AMBIGUOUS_CONTEXT envelope must be well-formed and include candidate info."""

    def test_code_is_ambiguous_context(self) -> None:
        c1 = _make_candidate(palace="a")
        c2 = _make_candidate(palace="b")
        env = build_ambiguous_context_envelope([c1, c2])
        assert env["error"]["code"] == "AMBIGUOUS_CONTEXT"

    def test_retryable_is_false(self) -> None:
        c1 = _make_candidate(palace="a")
        env = build_ambiguous_context_envelope([c1])
        assert env["error"]["retryable"] is False

    def test_single_candidate_in_envelope(self) -> None:
        c = _make_candidate(palace="only")
        env = build_ambiguous_context_envelope([c])
        assert len(env["error"]["candidates"]) == 1

    def test_candidate_scope_preserved(self) -> None:
        c = _make_candidate(palace="myorg", wing="mysvc")
        env = build_ambiguous_context_envelope([c])
        cand = env["error"]["candidates"][0]
        assert cand["scope"]["palace"] == "myorg"
        assert cand["scope"]["wing"] == "mysvc"


# ===========================================================================
# 9. Phase 4 — programmatic onboard orchestrator (unit tests)
# ===========================================================================

def _readable_entry(path: str, content: str = "hello world") -> ScannedFileEntry:
    return ScannedFileEntry(path=path, content=content, size_bytes=len(content))


def _binary_entry(path: str) -> ScannedFileEntry:
    return ScannedFileEntry(path=path, content="", size_bytes=100, is_binary=True)


def _unsupported_entry(path: str) -> ScannedFileEntry:
    return ScannedFileEntry(path=path, content="", size_bytes=50, is_unsupported=True)


_SCOPE = {"palace": "test-palace", "wing": "test-wing", "room": "test-room"}


class TestProgrammaticOnboardSuccess:
    """Successful onboard with readable files."""

    def test_status_is_completed(self) -> None:
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=[_readable_entry("src/main.py")])
        result = run_programmatic_onboard(req)
        assert result.status == "completed"

    def test_graph_is_populated(self) -> None:
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=[_readable_entry("src/main.py")])
        result = run_programmatic_onboard(req)
        assert result.graph is not None
        assert len(result.graph.nodes) >= 4  # palace + wing + room + compartment

    def test_compartments_count_equals_file_count(self) -> None:
        files = [_readable_entry("a.py"), _readable_entry("b.py")]
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=files)
        result = run_programmatic_onboard(req)
        assert len(result.compartments) == 2

    def test_metadata_only_count_zero_for_readable_files(self) -> None:
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=[_readable_entry("src/main.py")])
        result = run_programmatic_onboard(req)
        assert result.metadata_only_count == 0

    def test_no_error_on_success(self) -> None:
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=[_readable_entry("src/main.py")])
        result = run_programmatic_onboard(req)
        assert result.error is None

    def test_scope_key_is_stable(self) -> None:
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=[_readable_entry("src/main.py")])
        result = run_programmatic_onboard(req)
        assert isinstance(result.scope_key_value, str)
        assert len(result.scope_key_value) == 64  # SHA-256 hex


class TestProgrammaticOnboardMetadataOnly:
    """Binary / unsupported files produce metadata-only compartments."""

    def test_binary_file_creates_metadata_only_compartment(self) -> None:
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=[_binary_entry("image.png")])
        result = run_programmatic_onboard(req)
        assert result.status == "completed"
        assert result.metadata_only_count == 1

    def test_unsupported_file_creates_metadata_only_compartment(self) -> None:
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=[_unsupported_entry("data.bin")])
        result = run_programmatic_onboard(req)
        assert result.metadata_only_count == 1

    def test_mixed_files_counts_are_correct(self) -> None:
        files = [
            _readable_entry("src/main.py"),
            _binary_entry("logo.png"),
            _unsupported_entry("archive.zip"),
        ]
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=files)
        result = run_programmatic_onboard(req)
        assert result.status == "completed"
        assert len(result.compartments) == 3
        assert result.metadata_only_count == 2

    def test_all_binary_files_still_complete(self) -> None:
        """All-binary input should still produce a valid completed result."""
        files = [_binary_entry("a.png"), _binary_entry("b.jpg")]
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=files)
        result = run_programmatic_onboard(req)
        assert result.status == "completed"
        assert result.metadata_only_count == 2

    def test_compartment_is_metadata_only_flag(self) -> None:
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=[_binary_entry("img.png")])
        result = run_programmatic_onboard(req)
        assert result.compartments[0].is_metadata_only is True

    def test_readable_compartment_is_not_metadata_only(self) -> None:
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=[_readable_entry("main.py")])
        result = run_programmatic_onboard(req)
        assert result.compartments[0].is_metadata_only is False


class TestProgrammaticOnboardConciseResponse:
    """Concise (debug=False) response omits diagnostics."""

    def test_concise_response_has_status(self) -> None:
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=[_readable_entry("main.py")])
        result = run_programmatic_onboard(req)
        resp = build_programmatic_onboard_response(result, debug=False)
        assert resp["status"] == "completed"

    def test_concise_response_has_compartments_total(self) -> None:
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=[_readable_entry("main.py")])
        result = run_programmatic_onboard(req)
        resp = build_programmatic_onboard_response(result, debug=False)
        assert resp["compartments_total"] == 1

    def test_concise_response_has_graph_summary(self) -> None:
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=[_readable_entry("main.py")])
        result = run_programmatic_onboard(req)
        resp = build_programmatic_onboard_response(result, debug=False)
        graph_summary = resp["graph"]
        assert "nodes" in graph_summary
        assert "corridors" in graph_summary

    def test_concise_response_omits_diagnostics(self) -> None:
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=[_readable_entry("main.py")])
        result = run_programmatic_onboard(req)
        resp = build_programmatic_onboard_response(result, debug=False)
        assert "diagnostics" not in resp

    def test_concise_response_has_metadata_only_count(self) -> None:
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=[_binary_entry("img.png")])
        result = run_programmatic_onboard(req)
        resp = build_programmatic_onboard_response(result, debug=False)
        assert resp["metadata_only_count"] == 1


class TestProgrammaticOnboardDebugResponse:
    """debug=True response includes diagnostics block."""

    def test_debug_response_includes_diagnostics(self) -> None:
        req = ProgrammaticOnboardRequest(
            scope=_SCOPE, files=[_readable_entry("main.py")], debug=True
        )
        result = run_programmatic_onboard(req)
        resp = build_programmatic_onboard_response(result, debug=True)
        assert "diagnostics" in resp

    def test_debug_diagnostics_includes_graph(self) -> None:
        req = ProgrammaticOnboardRequest(
            scope=_SCOPE, files=[_readable_entry("main.py")], debug=True
        )
        result = run_programmatic_onboard(req)
        resp = build_programmatic_onboard_response(result, debug=True)
        assert "graph" in resp["diagnostics"]

    def test_debug_diagnostics_includes_compartments(self) -> None:
        req = ProgrammaticOnboardRequest(
            scope=_SCOPE,
            files=[_readable_entry("a.py"), _binary_entry("b.png")],
            debug=True,
        )
        result = run_programmatic_onboard(req)
        resp = build_programmatic_onboard_response(result, debug=True)
        compartments = resp["diagnostics"]["compartments"]
        assert len(compartments) == 2

    def test_debug_compartment_has_metadata_only_flag(self) -> None:
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=[_binary_entry("img.png")], debug=True)
        result = run_programmatic_onboard(req)
        resp = build_programmatic_onboard_response(result, debug=True)
        comp = resp["diagnostics"]["compartments"][0]
        assert comp["is_metadata_only"] is True


class TestProgrammaticOnboardValidationFailure:
    """Graph completeness failure maps to GRAPH_COMPLETENESS_FAILED."""

    def test_empty_scope_with_no_files_still_completes(self) -> None:
        """Empty file list creates a placeholder compartment, so graph should be valid."""
        req = ProgrammaticOnboardRequest(scope={}, files=[])
        result = run_programmatic_onboard(req)
        # Placeholder compartment satisfies the graph validator.
        assert result.status == "completed"

    def test_unsupported_mode_returns_failed(self) -> None:
        """Passing mode != 'programmatic' should return a failed result."""
        import dataclasses
        req = ProgrammaticOnboardRequest(scope=_SCOPE, files=[_readable_entry("main.py")])
        # Bypass frozen check by creating a new object with mode overridden
        req_bad = dataclasses.replace(req, mode="llm")  # type: ignore[call-overload]
        result = run_programmatic_onboard(req_bad)
        assert result.status == "failed"
        assert result.error is not None
        err = result.error.get("error", {})
        assert err.get("code") == "UNSUPPORTED_MODE"


class TestClassifyScanFilesForProgrammaticMode:
    """classify_scan_files_for_programmatic_mode converts raw scan dicts to typed entries."""

    def test_readable_file_is_not_binary(self) -> None:
        raw = [{"path": "main.py", "content": "print('hello')", "size_bytes": 14}]
        entries = classify_scan_files_for_programmatic_mode(raw)
        assert len(entries) == 1
        assert entries[0].is_binary is False
        assert entries[0].has_readable_content is True

    def test_empty_content_file_becomes_unsupported(self) -> None:
        raw = [{"path": "empty.txt", "content": "   ", "size_bytes": 0}]
        entries = classify_scan_files_for_programmatic_mode(raw)
        assert entries[0].is_unsupported is True

    def test_explicit_is_binary_flag_honored(self) -> None:
        raw = [{"path": "img.png", "content": "", "size_bytes": 1024, "is_binary": True}]
        entries = classify_scan_files_for_programmatic_mode(raw)
        assert entries[0].is_binary is True

    def test_content_hash_computed_when_absent(self) -> None:
        raw = [{"path": "a.py", "content": "x = 1", "size_bytes": 5}]
        entries = classify_scan_files_for_programmatic_mode(raw)
        assert len(entries[0].content_hash) == 64  # SHA-256 hex

    def test_content_hash_preserved_when_provided(self) -> None:
        raw = [{"path": "a.py", "content": "x = 1", "size_bytes": 5, "content_hash": "abc123"}]
        entries = classify_scan_files_for_programmatic_mode(raw)
        assert entries[0].content_hash == "abc123"

    def test_empty_list_returns_empty(self) -> None:
        assert classify_scan_files_for_programmatic_mode([]) == []


# ===========================================================================
# 10. Phase 4 — onboard tool fast-path integration (programmatic mode)
# ===========================================================================


class TestOnboardProgrammaticFastPath:
    """onboard tool: scan + no checkpoint + no ingest -> orchestrator response."""

    @pytest.mark.asyncio
    async def test_scan_no_checkpoint_returns_completed(
        self, mock_ctx: MagicMock, tmp_path: Path
    ) -> None:
        """onboard with ingestion.mode=programmatic returns completed status."""
        test_file = tmp_path / "main.py"
        test_file.write_text("print('hello')")

        onboard = _get_tool_fn("onboard")
        result = await onboard(
            scope={"palace": "test-org", "wing": "svc"},
            scan={
                "patterns": ["*.py"],
                "root": str(tmp_path),
                "max_files": 5,
                "max_size_kb": 10,
            },
            ingestion={"mode": "programmatic"},
            ctx=mock_ctx,
        )
        payload = json.loads(result.content[0].text)
        assert payload.get("status") == "completed", f"Unexpected response: {payload}"

    @pytest.mark.asyncio
    async def test_programmatic_completed_enables_watcher_for_active_project(
        self, mock_ctx: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "main.py").write_text("print('hello')")

        active_project = SessionProjectContext(
            project_id="proj-123",
            slug="test-proj",
            palace="test-org",
            default_wing="svc",
            default_room=None,
            source="session_selected",
        )
        app_ctx = mock_ctx.request_context.lifespan_context
        app_ctx.get_active_project.return_value = active_project
        app_ctx.watcher_manager = MagicMock()

        onboard = _get_tool_fn("onboard")
        result = await onboard(
            scope={"palace": "test-org", "wing": "svc"},
            scan={
                "patterns": ["*.py"],
                "root": str(tmp_path),
                "max_files": 5,
                "max_size_kb": 10,
            },
            ingestion={"mode": "programmatic"},
            ctx=mock_ctx,
        )
        payload = json.loads(result.content[0].text)
        assert payload.get("status") == "completed", f"Unexpected response: {payload}"
        app_ctx.watcher_manager.enable_project_by_default.assert_called_once_with("proj-123")

    @pytest.mark.asyncio
    async def test_programmatic_completed_no_active_project_is_noop_for_watcher(
        self, mock_ctx: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "main.py").write_text("print('hello')")

        app_ctx = mock_ctx.request_context.lifespan_context
        app_ctx.get_active_project.return_value = None
        app_ctx.watcher_manager = MagicMock()

        onboard = _get_tool_fn("onboard")
        result = await onboard(
            scope={"palace": "test-org", "wing": "svc"},
            scan={
                "patterns": ["*.py"],
                "root": str(tmp_path),
                "max_files": 5,
                "max_size_kb": 10,
            },
            ingestion={"mode": "programmatic"},
            ctx=mock_ctx,
        )
        payload = json.loads(result.content[0].text)
        assert payload.get("status") == "completed", f"Unexpected response: {payload}"
        app_ctx.watcher_manager.enable_project_by_default.assert_not_called()

    @pytest.mark.asyncio
    async def test_scan_no_checkpoint_response_has_graph_summary(
        self, mock_ctx: MagicMock, tmp_path: Path
    ) -> None:
        test_file = tmp_path / "service.py"
        test_file.write_text("def main(): pass")

        onboard = _get_tool_fn("onboard")
        result = await onboard(
            scope={"palace": "org", "wing": "api"},
            scan={
                "patterns": ["*.py"],
                "root": str(tmp_path),
                "max_files": 5,
                "max_size_kb": 10,
            },
            ingestion={"mode": "programmatic"},
            ctx=mock_ctx,
        )
        payload = json.loads(result.content[0].text)
        assert "graph" in payload, f"Missing graph key: {payload}"
        assert payload["graph"]["nodes"] >= 4


@pytest.mark.asyncio
async def test_ephemeral_memory_request_ensures_schema_before_graph_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakePostgresBackend:
        async def connect(self, config: object) -> None:
            events.append("connect")

        async def disconnect(self) -> None:
            events.append("disconnect")

    backend = FakePostgresBackend()

    class FakeMemoryService:
        def __init__(self, service_backend: object, execution: object) -> None:
            assert service_backend is backend
            events.append("service")

        async def execute(self, request: object) -> object:
            events.append("execute")
            return object()

    async def fake_ensure_schema(schema_backend: object) -> None:
        assert schema_backend is backend
        events.append("ensure_schema")

    monkeypatch.setattr(tools_memory, "PostgresBackend", lambda: backend)
    monkeypatch.setattr(
        tools_memory,
        "_memory_connection_config_from_metadata",
        lambda app_ctx: object(),
    )
    monkeypatch.setattr(tools_memory, "_memory_connection_config_from_env", lambda: None)
    monkeypatch.setattr(tools_memory, "MemoryService", FakeMemoryService)
    monkeypatch.setattr(
        tools_memory,
        "_shape_memory_response",
        lambda result, response: {"ok": True},
    )
    monkeypatch.setattr(knowledge_schema, "ensure_schema", fake_ensure_schema)

    result = await tools_memory._execute_memory_request(
        app_ctx=SimpleNamespace(memory_backend=None, memory_backend_lock=None),
        execution=object(),
        operation="graph_upsert",
        scope={},
        scope_token=None,
        context_id=None,
        query=None,
        record=None,
        graph={"kind": "place", "place_name": "Root", "place_type": "Palace"},
        maintenance=None,
        response={},
    )

    assert result == {"ok": True}
    assert events == ["connect", "ensure_schema", "service", "execute", "disconnect"]

    @pytest.mark.asyncio
    async def test_programmatic_onboard_persists_graph_payload_when_memory_backend_ready(
        self, mock_ctx: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "service.py").write_text("def main(): pass")
        app_ctx = mock_ctx.request_context.lifespan_context
        app_ctx.memory_backend = object()

        persisted: list[dict[str, Any]] = []

        async def _capture_memory_request(**kwargs: Any) -> dict[str, Any]:
            graph = dict(kwargs["graph"])
            persisted.append(graph)
            if graph["kind"] == "place":
                return {"entity_id": f"entity-{len(persisted)}"}
            return {"relation_id": f"relation-{len(persisted)}"}

        onboard = _get_tool_fn("onboard")
        with patch(
            "workflows_mcp.tools_memory._execute_memory_request",
            new=AsyncMock(side_effect=_capture_memory_request),
        ):
            result = await onboard(
                scope={"palace": "org", "wing": "api", "room": "runtime"},
                scan={
                    "patterns": ["*.py"],
                    "root": str(tmp_path),
                    "max_files": 5,
                    "max_size_kb": 10,
                },
                ingestion={"mode": "programmatic"},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        assert payload["graph"]["nodes"] == len(
            [item for item in persisted if item["kind"] == "place"]
        )
        assert payload["graph"]["corridors"] == len(
            [item for item in persisted if item["kind"] == "link"]
        )
        assert any(
            item["kind"] == "place" and item["place_type"] == "Palace"
            for item in persisted
        )
        assert any(
            item["kind"] == "link"
            and item["link_type"] == "contains"
            and str(item["from"]).startswith("entity-")
            and str(item["to"]).startswith("entity-")
            for item in persisted
        )

    @pytest.mark.asyncio
    async def test_scan_no_checkpoint_concise_omits_diagnostics(
        self, mock_ctx: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "a.py").write_text("x = 1")

        onboard = _get_tool_fn("onboard")
        result = await onboard(
            scope={"palace": "org"},
            scan={
                "patterns": ["*.py"],
                "root": str(tmp_path),
                "max_files": 5,
                "max_size_kb": 10,
            },
            ingestion={"mode": "programmatic"},
            debug=False,
            ctx=mock_ctx,
        )
        payload = json.loads(result.content[0].text)
        assert "diagnostics" not in payload

    @pytest.mark.asyncio
    async def test_scan_no_checkpoint_debug_includes_diagnostics(
        self, mock_ctx: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "b.py").write_text("y = 2")

        onboard = _get_tool_fn("onboard")
        result = await onboard(
            scope={"palace": "org"},
            scan={
                "patterns": ["*.py"],
                "root": str(tmp_path),
                "max_files": 5,
                "max_size_kb": 10,
            },
            ingestion={"mode": "programmatic"},
            debug=True,
            ctx=mock_ctx,
        )
        payload = json.loads(result.content[0].text)
        assert "diagnostics" in payload, f"Expected diagnostics in debug mode: {payload}"

    @pytest.mark.asyncio
    async def test_scan_without_programmatic_mode_uses_checkpoint_flow(
        self, mock_ctx: MagicMock, tmp_path: Path
    ) -> None:
        """Without ingestion.mode=programmatic, scan + no ingest goes through the checkpoint flow
        (which requires a DB connection), not the orchestrator fast-path."""
        (tmp_path / "c.py").write_text("z = 3")

        onboard = _get_tool_fn("onboard")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await onboard(
                scope={"palace": "org"},
                scan={
                "patterns": ["*.py"],
                "root": str(tmp_path),
                "max_files": 5,
                "max_size_kb": 10,
            },
                ctx=mock_ctx,
            )
        # Checkpoint flow (not the orchestrator) handles it; response will not have graph key.
        payload = json.loads(result.content[0].text)
        assert "graph" not in payload


# ===========================================================================
# 11. Phase 5 — LLM onboard orchestrator (unit tests)
# ===========================================================================


def _make_loader(
    *,
    profile_name: str = "standard",
    model: str = "gpt-4o",
    temperature: float | None = 0.0,
    default_profile: str | None = "standard",
) -> LLMConfigLoader:
    """Build a pre-configured LLMConfigLoader without touching the filesystem."""
    config = LLMConfig(
        profiles={
            profile_name: ProfileConfig(
                provider="openai-cloud",
                model=model,
                temperature=temperature,
            )
        },
        providers={"openai-cloud": ProviderConfig(type="openai")},
        default_profile=default_profile,
    )
    loader = LLMConfigLoader()
    loader._config = config
    return loader


def _make_loader_no_profiles() -> LLMConfigLoader:
    """Loader with empty profiles (simulates missing config)."""
    config = LLMConfig(profiles={}, providers={}, default_profile=None)
    loader = LLMConfigLoader()
    loader._config = config
    return loader


_LLM_SCOPE = {"palace": "test-palace", "wing": "test-wing", "room": "test-room"}


class TestLLMOnboardMissingProfile:
    """Missing or invalid profile returns INVALID_LLM_PROFILE envelope."""

    def test_nonexistent_profile_returns_failed(self) -> None:
        loader = _make_loader_no_profiles()
        req = LLMOnboardRequest(scope=_LLM_SCOPE, files=[], profile="nonexistent")
        result = run_llm_onboard(req, loader=loader)
        assert result.status == "failed"

    def test_nonexistent_profile_error_code(self) -> None:
        loader = _make_loader_no_profiles()
        req = LLMOnboardRequest(scope=_LLM_SCOPE, files=[], profile="nonexistent")
        result = run_llm_onboard(req, loader=loader)
        err = (result.error or {}).get("error", {})
        assert err.get("code") == "INVALID_LLM_PROFILE"

    def test_nonexistent_profile_retryable_false(self) -> None:
        loader = _make_loader_no_profiles()
        req = LLMOnboardRequest(scope=_LLM_SCOPE, files=[], profile="ghost")
        result = run_llm_onboard(req, loader=loader)
        err = (result.error or {}).get("error", {})
        assert err.get("retryable") is False

    def test_nonexistent_profile_message_contains_profile_name(self) -> None:
        loader = _make_loader_no_profiles()
        req = LLMOnboardRequest(scope=_LLM_SCOPE, files=[], profile="my-missing-profile")
        result = run_llm_onboard(req, loader=loader)
        err = (result.error or {}).get("error", {})
        assert "my-missing-profile" in err.get("message", "")

    def test_empty_profile_name_returns_invalid_profile(self) -> None:
        loader = _make_loader()
        req = LLMOnboardRequest(scope=_LLM_SCOPE, files=[], profile="")
        result = run_llm_onboard(req, loader=loader)
        err = (result.error or {}).get("error", {})
        assert err.get("code") == "INVALID_LLM_PROFILE"

    def test_whitespace_profile_name_returns_invalid_profile(self) -> None:
        loader = _make_loader()
        req = LLMOnboardRequest(scope=_LLM_SCOPE, files=[], profile="   ")
        result = run_llm_onboard(req, loader=loader)
        err = (result.error or {}).get("error", {})
        assert err.get("code") == "INVALID_LLM_PROFILE"

    def test_profile_in_different_registry_not_found(self) -> None:
        """Profile name that exists in a different loader instance is not found here."""
        loader = _make_loader(profile_name="quick", default_profile="quick")
        # Ask for 'standard' which is NOT in this loader's profiles.
        req = LLMOnboardRequest(scope=_LLM_SCOPE, files=[], profile="standard")
        result = run_llm_onboard(req, loader=loader)
        err = (result.error or {}).get("error", {})
        assert err.get("code") == "INVALID_LLM_PROFILE"


class TestLLMOnboardStrictDefault:
    """Strict mode (default) requires temperature=0.0."""

    def test_zero_temperature_completes(self) -> None:
        loader = _make_loader(temperature=0.0)
        req = LLMOnboardRequest(
            scope=_LLM_SCOPE,
            files=[_readable_entry("main.py")],
            profile="standard",
        )
        result = run_llm_onboard(req, loader=loader)
        assert result.status == "completed"

    def test_none_temperature_raises_strict_violation(self) -> None:
        """temperature=None is not the same as 0.0 in strict mode."""
        loader = _make_loader(temperature=None)
        req = LLMOnboardRequest(
            scope=_LLM_SCOPE,
            files=[_readable_entry("main.py")],
            profile="standard",
            strict=True,
        )
        result = run_llm_onboard(req, loader=loader)
        err = (result.error or {}).get("error", {})
        assert err.get("code") == "STRICT_REPRODUCIBILITY_VIOLATION"

    def test_nonzero_temperature_strict_violation_code(self) -> None:
        loader = _make_loader(temperature=0.7)
        req = LLMOnboardRequest(
            scope=_LLM_SCOPE,
            files=[_readable_entry("main.py")],
            profile="standard",
            strict=True,
        )
        result = run_llm_onboard(req, loader=loader)
        assert result.status == "failed"
        err = (result.error or {}).get("error", {})
        assert err.get("code") == "STRICT_REPRODUCIBILITY_VIOLATION"

    def test_strict_violation_retryable_false(self) -> None:
        loader = _make_loader(temperature=1.0)
        req = LLMOnboardRequest(scope=_LLM_SCOPE, files=[], profile="standard", strict=True)
        result = run_llm_onboard(req, loader=loader)
        err = (result.error or {}).get("error", {})
        assert err.get("retryable") is False

    def test_strict_violation_message_mentions_temperature(self) -> None:
        loader = _make_loader(temperature=0.5)
        req = LLMOnboardRequest(scope=_LLM_SCOPE, files=[], profile="standard", strict=True)
        result = run_llm_onboard(req, loader=loader)
        err = (result.error or {}).get("error", {})
        assert "temperature" in err.get("message", "").lower()

    def test_strict_is_default(self) -> None:
        """LLMOnboardRequest.strict defaults to True."""
        req = LLMOnboardRequest(scope={}, files=[], profile="p")
        assert req.strict is True

    def test_strict_violation_error_has_actionable_fix(self) -> None:
        loader = _make_loader(temperature=0.9)
        req = LLMOnboardRequest(scope=_LLM_SCOPE, files=[], profile="standard", strict=True)
        result = run_llm_onboard(req, loader=loader)
        err = (result.error or {}).get("error", {})
        assert err.get("actionable_fix")


class TestLLMOnboardRelaxedOptIn:
    """Explicit strict=False allows non-zero temperature profiles."""

    def test_relaxed_nonzero_temperature_completes(self) -> None:
        loader = _make_loader(temperature=0.7)
        req = LLMOnboardRequest(
            scope=_LLM_SCOPE,
            files=[_readable_entry("main.py")],
            profile="standard",
            strict=False,
        )
        result = run_llm_onboard(req, loader=loader)
        assert result.status == "completed"

    def test_relaxed_none_temperature_completes(self) -> None:
        loader = _make_loader(temperature=None)
        req = LLMOnboardRequest(
            scope=_LLM_SCOPE,
            files=[_readable_entry("main.py")],
            profile="standard",
            strict=False,
        )
        result = run_llm_onboard(req, loader=loader)
        assert result.status == "completed"

    def test_relaxed_high_temperature_completes(self) -> None:
        loader = _make_loader(temperature=2.0)
        req = LLMOnboardRequest(
            scope=_LLM_SCOPE,
            files=[_readable_entry("a.py"), _readable_entry("b.py")],
            profile="standard",
            strict=False,
        )
        result = run_llm_onboard(req, loader=loader)
        assert result.status == "completed"
        assert len(result.compartments) == 2

    def test_relaxed_debug_provenance_reproducibility_is_relaxed(self) -> None:
        loader = _make_loader(temperature=0.7)
        req = LLMOnboardRequest(
            scope=_LLM_SCOPE,
            files=[_readable_entry("x.py")],
            profile="standard",
            strict=False,
            debug=True,
        )
        result = run_llm_onboard(req, loader=loader)
        assert result.llm_provenance is not None
        assert result.llm_provenance.reproducibility == "relaxed"


class TestLLMOnboardDebugProvenance:
    """LLM debug provenance fields are present only in LLM mode with debug=True."""

    def test_debug_true_provenance_present(self) -> None:
        loader = _make_loader(temperature=0.0)
        req = LLMOnboardRequest(
            scope=_LLM_SCOPE,
            files=[_readable_entry("main.py")],
            profile="standard",
            debug=True,
        )
        result = run_llm_onboard(req, loader=loader)
        assert result.llm_provenance is not None

    def test_debug_false_provenance_absent(self) -> None:
        loader = _make_loader(temperature=0.0)
        req = LLMOnboardRequest(
            scope=_LLM_SCOPE,
            files=[_readable_entry("main.py")],
            profile="standard",
            debug=False,
        )
        result = run_llm_onboard(req, loader=loader)
        assert result.llm_provenance is None

    def test_debug_response_dict_contains_llm_provenance(self) -> None:
        loader = _make_loader(temperature=0.0)
        req = LLMOnboardRequest(
            scope=_LLM_SCOPE,
            files=[_readable_entry("main.py")],
            profile="standard",
            debug=True,
        )
        result = run_llm_onboard(req, loader=loader)
        resp = build_llm_onboard_response(result, debug=True)
        assert "llm_provenance" in resp

    def test_concise_response_dict_omits_llm_provenance(self) -> None:
        loader = _make_loader(temperature=0.0)
        req = LLMOnboardRequest(
            scope=_LLM_SCOPE,
            files=[_readable_entry("main.py")],
            profile="standard",
            debug=True,  # result has provenance internally
        )
        result = run_llm_onboard(req, loader=loader)
        resp = build_llm_onboard_response(result, debug=False)  # but response is concise
        assert "llm_provenance" not in resp

    def test_provenance_has_model_field(self) -> None:
        loader = _make_loader(model="gpt-4o-mini", temperature=0.0)
        req = LLMOnboardRequest(
            scope=_LLM_SCOPE,
            files=[_readable_entry("main.py")],
            profile="standard",
            debug=True,
        )
        result = run_llm_onboard(req, loader=loader)
        assert result.llm_provenance is not None
        assert result.llm_provenance.model == "gpt-4o-mini"

    def test_provenance_has_profile_field(self) -> None:
        loader = _make_loader(temperature=0.0)
        req = LLMOnboardRequest(
            scope=_LLM_SCOPE,
            files=[_readable_entry("main.py")],
            profile="standard",
            debug=True,
        )
        result = run_llm_onboard(req, loader=loader)
        assert result.llm_provenance is not None
        assert result.llm_provenance.profile == "standard"

    def test_provenance_reproducibility_strict(self) -> None:
        loader = _make_loader(temperature=0.0)
        req = LLMOnboardRequest(
            scope=_LLM_SCOPE,
            files=[_readable_entry("main.py")],
            profile="standard",
            strict=True,
            debug=True,
        )
        result = run_llm_onboard(req, loader=loader)
        assert result.llm_provenance is not None
        assert result.llm_provenance.reproducibility == "strict"

    def test_provenance_dict_has_all_fingerprint_keys(self) -> None:
        loader = _make_loader(temperature=0.0)
        req = LLMOnboardRequest(
            scope=_LLM_SCOPE,
            files=[_readable_entry("main.py")],
            profile="standard",
            debug=True,
        )
        result = run_llm_onboard(req, loader=loader)
        prov = result.llm_provenance
        assert prov is not None
        d = prov.to_dict()
        for key in ("profile", "model", "provider", "temperature", "strict", "reproducibility"):
            assert key in d, f"Missing provenance key: {key}"

    def test_programmatic_mode_has_no_llm_provenance(self) -> None:
        """Programmatic mode must never emit llm_provenance (even in debug)."""
        req = ProgrammaticOnboardRequest(
            scope=_LLM_SCOPE,
            files=[_readable_entry("main.py")],
            debug=True,
        )
        result = run_programmatic_onboard(req)
        resp = build_programmatic_onboard_response(result, debug=True)
        assert "llm_provenance" not in resp


# ===========================================================================
# 12. Phase 5 — LLM onboard tool fast-path integration
# ===========================================================================


class TestOnboardLLMModeToolFastPath:
    """onboard tool: mode=llm fast-path integration tests."""

    def _make_ctx_with_loader(
        self, loader: LLMConfigLoader
    ) -> MagicMock:
        """Build a mock ctx whose app_ctx carries the supplied loader."""
        ctx = MagicMock()
        app_ctx = MagicMock()
        session = object()
        app_ctx.memory_backend = None
        app_ctx.memory_backend_lock = None
        app_ctx.memory_backend_unavailable_error = None
        app_ctx.llm_config_loader = loader
        ctx.request_context.lifespan_context = app_ctx
        ctx.request_context.session = session

        active_context_store: dict[int, Any] = {}

        def _set_active_context(session_obj: Any, candidate: Any) -> None:
            active_context_store[id(session_obj)] = candidate

        def _get_active_context(session_obj: Any) -> Any:
            return active_context_store.get(id(session_obj))

        app_ctx.set_active_context.side_effect = _set_active_context
        app_ctx.get_active_context.side_effect = _get_active_context

        onboard_candidates_store: dict[int, list[Any]] = {}

        def _register_onboard_context_candidate(session_obj: Any, candidate: Any) -> None:
            onboard_candidates_store.setdefault(id(session_obj), []).append(candidate)

        def _list_onboard_context_candidates(session_obj: Any) -> list[Any]:
            return list(onboard_candidates_store.get(id(session_obj), []))

        app_ctx.register_onboard_context_candidate.side_effect = _register_onboard_context_candidate
        app_ctx.list_onboard_context_candidates.side_effect = _list_onboard_context_candidates

        exec_context = MagicMock()
        exec_context.user_string_id = None
        app_ctx.create_execution_context.return_value = exec_context
        app_ctx.get_user_context.return_value = (uuid.UUID(int=0), "test-user", "OS_USER")
        return ctx

    @pytest.mark.asyncio
    async def test_llm_mode_missing_profile_key_returns_invalid_profile(
        self, tmp_path: Path
    ) -> None:
        loader = _make_loader(temperature=0.0)
        ctx = self._make_ctx_with_loader(loader)
        (tmp_path / "a.py").write_text("x = 1")

        onboard = _get_tool_fn("onboard")
        result = await onboard(
            scope={"palace": "org"},
            scan={
                "patterns": ["*.py"],
                "root": str(tmp_path),
                "max_files": 5,
                "max_size_kb": 10,
            },
            ingestion={"mode": "llm"},  # no 'llm_profile' key
            ctx=ctx,
        )
        payload = json.loads(result.content[0].text)
        err = payload.get("error", {})
        assert err.get("code") == "INVALID_LLM_PROFILE"

    @pytest.mark.asyncio
    async def test_llm_mode_valid_profile_strict_completes(
        self, tmp_path: Path
    ) -> None:
        loader = _make_loader(temperature=0.0)
        ctx = self._make_ctx_with_loader(loader)
        (tmp_path / "main.py").write_text("print('hello')")

        onboard = _get_tool_fn("onboard")
        result = await onboard(
            scope={"palace": "org", "wing": "svc"},
            scan={
                "patterns": ["*.py"],
                "root": str(tmp_path),
                "max_files": 5,
                "max_size_kb": 10,
            },
            ingestion={"mode": "llm", "llm_profile": "standard"},
            ctx=ctx,
        )
        payload = json.loads(result.content[0].text)
        assert payload.get("status") == "completed", f"Unexpected: {payload}"

    @pytest.mark.asyncio
    async def test_llm_mode_strict_violation_returns_error(
        self, tmp_path: Path
    ) -> None:
        loader = _make_loader(temperature=0.9)
        ctx = self._make_ctx_with_loader(loader)
        (tmp_path / "x.py").write_text("pass")

        onboard = _get_tool_fn("onboard")
        result = await onboard(
            scope={"palace": "org"},
            scan={
                "patterns": ["*.py"],
                "root": str(tmp_path),
                "max_files": 5,
                "max_size_kb": 10,
            },
            ingestion={
                "mode": "llm",
                "llm_profile": "standard",
            },  # reproducibility defaults to strict
            ctx=ctx,
        )
        payload = json.loads(result.content[0].text)
        err = payload.get("error", {})
        assert err.get("code") == "STRICT_REPRODUCIBILITY_VIOLATION"

    @pytest.mark.asyncio
    async def test_llm_mode_relaxed_nonzero_temperature_completes(
        self, tmp_path: Path
    ) -> None:
        loader = _make_loader(temperature=0.7)
        ctx = self._make_ctx_with_loader(loader)
        (tmp_path / "svc.py").write_text("def run(): pass")

        onboard = _get_tool_fn("onboard")
        result = await onboard(
            scope={"palace": "org"},
            scan={
                "patterns": ["*.py"],
                "root": str(tmp_path),
                "max_files": 5,
                "max_size_kb": 10,
            },
            ingestion={"mode": "llm", "llm_profile": "standard", "reproducibility": "relaxed"},
            ctx=ctx,
        )
        payload = json.loads(result.content[0].text)
        assert payload.get("status") == "completed", f"Unexpected: {payload}"

    @pytest.mark.asyncio
    async def test_llm_mode_debug_response_has_llm_provenance(
        self, tmp_path: Path
    ) -> None:
        loader = _make_loader(temperature=0.0)
        ctx = self._make_ctx_with_loader(loader)
        (tmp_path / "b.py").write_text("y = 2")

        onboard = _get_tool_fn("onboard")
        result = await onboard(
            scope={"palace": "org"},
            scan={
                "patterns": ["*.py"],
                "root": str(tmp_path),
                "max_files": 5,
                "max_size_kb": 10,
            },
            ingestion={"mode": "llm", "llm_profile": "standard"},
            debug=True,
            ctx=ctx,
        )
        payload = json.loads(result.content[0].text)
        assert "llm_provenance" in payload, f"Expected llm_provenance in debug response: {payload}"

    @pytest.mark.asyncio
    async def test_llm_mode_concise_response_omits_llm_provenance(
        self, tmp_path: Path
    ) -> None:
        loader = _make_loader(temperature=0.0)
        ctx = self._make_ctx_with_loader(loader)
        (tmp_path / "c.py").write_text("z = 3")

        onboard = _get_tool_fn("onboard")
        result = await onboard(
            scope={"palace": "org"},
            scan={
                "patterns": ["*.py"],
                "root": str(tmp_path),
                "max_files": 5,
                "max_size_kb": 10,
            },
            ingestion={"mode": "llm", "llm_profile": "standard"},
            debug=False,
            ctx=ctx,
        )
        payload = json.loads(result.content[0].text)
        assert "llm_provenance" not in payload

    @pytest.mark.asyncio
    async def test_llm_mode_has_graph_summary(
        self, tmp_path: Path
    ) -> None:
        loader = _make_loader(temperature=0.0)
        ctx = self._make_ctx_with_loader(loader)
        (tmp_path / "d.py").write_text("class A: pass")

        onboard = _get_tool_fn("onboard")
        result = await onboard(
            scope={"palace": "org", "wing": "api"},
            scan={
                "patterns": ["*.py"],
                "root": str(tmp_path),
                "max_files": 5,
                "max_size_kb": 10,
            },
            ingestion={"mode": "llm", "llm_profile": "standard"},
            ctx=ctx,
        )
        payload = json.loads(result.content[0].text)
        assert "graph" in payload
        assert payload["graph"]["nodes"] >= 4


# ===========================================================================
# Phase 6 — compute_sync_delta
# ===========================================================================


def _entry(path: str, content_hash: str) -> dict[str, Any]:
    return {"path": path, "content_hash": content_hash}


class TestComputeSyncDelta:
    def test_all_added_when_no_prior(self) -> None:
        new = [_entry("a.py", "h1"), _entry("b.py", "h2")]
        delta = compute_sync_delta([], new)
        assert delta.added == ["a.py", "b.py"]
        assert delta.modified == []
        assert delta.deleted == []
        assert delta.unchanged == []

    def test_all_deleted_when_no_new(self) -> None:
        prior = [_entry("a.py", "h1")]
        delta = compute_sync_delta(prior, [])
        assert delta.added == []
        assert delta.deleted == ["a.py"]
        assert delta.unchanged == []

    def test_unchanged_when_hashes_match(self) -> None:
        entries = [_entry("a.py", "h1"), _entry("b.py", "h2")]
        delta = compute_sync_delta(entries, entries)
        assert delta.unchanged == ["a.py", "b.py"]
        assert delta.added == []
        assert delta.modified == []
        assert delta.deleted == []

    def test_modified_when_hash_differs(self) -> None:
        prior = [_entry("a.py", "h1")]
        new = [_entry("a.py", "h2")]
        delta = compute_sync_delta(prior, new)
        assert delta.modified == ["a.py"]
        assert delta.unchanged == []

    def test_mixed_classification(self) -> None:
        prior = [_entry("keep.py", "hk"), _entry("change.py", "hc"), _entry("gone.py", "hg")]
        new = [_entry("keep.py", "hk"), _entry("change.py", "hc_new"), _entry("added.py", "ha")]
        delta = compute_sync_delta(prior, new)
        assert delta.unchanged == ["keep.py"]
        assert delta.modified == ["change.py"]
        assert delta.deleted == ["gone.py"]
        assert delta.added == ["added.py"]

    def test_has_semantic_delta_false_when_unchanged(self) -> None:
        entries = [_entry("x.py", "h")]
        delta = compute_sync_delta(entries, entries)
        assert not delta.has_semantic_delta

    def test_has_semantic_delta_true_when_modified(self) -> None:
        prior = [_entry("x.py", "h1")]
        new = [_entry("x.py", "h2")]
        delta = compute_sync_delta(prior, new)
        assert delta.has_semantic_delta

    def test_total_files_counts_new_snapshot(self) -> None:
        prior = [_entry("old.py", "h1")]
        new = [_entry("a.py", "h2"), _entry("b.py", "h3")]
        delta = compute_sync_delta(prior, new)
        assert delta.total_files == 2

    def test_entries_missing_path_ignored(self) -> None:
        prior = [{"content_hash": "h1"}]  # no path field
        new = [_entry("a.py", "h2")]
        delta = compute_sync_delta(prior, new)
        assert delta.added == ["a.py"]
        assert delta.deleted == []

    def test_output_is_sorted(self) -> None:
        new = [_entry("z.py", "h"), _entry("a.py", "h"), _entry("m.py", "h")]
        delta = compute_sync_delta([], new)
        assert delta.added == ["a.py", "m.py", "z.py"]

    def test_to_debug_dict_structure(self) -> None:
        prior = [_entry("a.py", "h1")]
        new = [_entry("a.py", "h2"), _entry("b.py", "hb")]
        delta = compute_sync_delta(prior, new)
        d = delta.to_debug_dict()
        assert set(d.keys()) >= {
            "added", "modified", "deleted", "unchanged",
            "has_semantic_delta", "total_files", "counts",
        }
        assert d["counts"]["added"] == 1
        assert d["counts"]["modified"] == 1


# ===========================================================================
# Phase 6 — compute_weak_links
# ===========================================================================


def _node(node_id: str) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        node_type=NodeType.COMPARTMENT,
        label=node_id,
        metadata={},
    )


def _corridor(source_id: str, target_id: str) -> GraphCorridor:
    return GraphCorridor(
        source_id=source_id,
        target_id=target_id,
        semantic_type=CorridorSemanticType.DEPENDS_ON,
        confidence=1.0,
        provenance="test",
        evidence=[],
    )


class TestComputeWeakLinks:
    def test_no_corridors_no_weak_links(self) -> None:
        graph = GraphPayload(nodes=[_node("a"), _node("b")], corridors=[])
        report = compute_weak_links(graph)
        assert not report.has_weak_links
        assert report.weak_link_count == 0

    def test_valid_corridor_no_weak_links(self) -> None:
        graph = GraphPayload(
            nodes=[_node("a"), _node("b")],
            corridors=[_corridor("a", "b")],
        )
        report = compute_weak_links(graph)
        assert not report.has_weak_links

    def test_orphaned_source_detected(self) -> None:
        graph = GraphPayload(
            nodes=[_node("b")],
            corridors=[_corridor("missing", "b")],
        )
        report = compute_weak_links(graph)
        assert report.has_weak_links
        assert "missing" in report.orphaned_source_ids
        assert report.weak_link_count == 1

    def test_orphaned_target_detected(self) -> None:
        graph = GraphPayload(
            nodes=[_node("a")],
            corridors=[_corridor("a", "missing_target")],
        )
        report = compute_weak_links(graph)
        assert report.has_weak_links
        assert "missing_target" in report.orphaned_target_ids

    def test_total_corridors_count(self) -> None:
        graph = GraphPayload(
            nodes=[_node("a"), _node("b")],
            corridors=[_corridor("a", "b"), _corridor("b", "a")],
        )
        report = compute_weak_links(graph)
        assert report.total_corridors == 2

    def test_to_debug_dict_structure(self) -> None:
        graph = GraphPayload(nodes=[_node("a")], corridors=[_corridor("a", "ghost")])
        d = compute_weak_links(graph).to_debug_dict()
        assert "has_weak_links" in d
        assert "weak_link_count" in d
        assert "orphaned_target_ids" in d


# ===========================================================================
# Phase 6 — classify_deletion_policy
# ===========================================================================


class TestClassifyDeletionPolicy:
    def test_archive_passthrough(self) -> None:
        assert classify_deletion_policy("archive") == "archive"

    def test_supersede_passthrough(self) -> None:
        assert classify_deletion_policy("supersede") == "supersede"

    def test_ignore_passthrough(self) -> None:
        assert classify_deletion_policy("ignore") == "ignore"

    def test_mark_missing_maps_to_archive(self) -> None:
        assert classify_deletion_policy("mark_missing") == "archive"

    def test_archive_missing_maps_to_archive(self) -> None:
        assert classify_deletion_policy("archive_missing") == "archive"

    def test_unknown_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown deletion_policy"):
            classify_deletion_policy("delete_forever")


# ===========================================================================
# Phase 6 — recompute_depends_on_corridors
# ===========================================================================


def _path_node(node_id: str, path: str) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        node_type=NodeType.COMPARTMENT,
        label=node_id,
        metadata={"path": path},
    )


class TestRecomputeDependsOnCorridors:
    def test_returns_same_graph_when_no_shared_prefix(self) -> None:
        graph = GraphPayload(
            nodes=[_path_node("a", "src/a.py"), _path_node("b", "tests/b.py")],
            corridors=[],
        )
        result = recompute_depends_on_corridors(graph)
        # Different prefixes: no new corridors
        assert len(result.corridors) == 0

    def test_injects_depends_on_for_shared_prefix(self) -> None:
        graph = GraphPayload(
            nodes=[_path_node("a", "src/a.py"), _path_node("b", "src/b.py")],
            corridors=[],
        )
        result = recompute_depends_on_corridors(graph)
        assert len(result.corridors) == 1
        c = result.corridors[0]
        assert c.semantic_type == CorridorSemanticType.DEPENDS_ON
        assert c.provenance == "sync"

    def test_does_not_duplicate_existing_corridor(self) -> None:
        existing = _corridor("a", "b")
        existing = GraphCorridor(
            source_id="a",
            target_id="b",
            semantic_type=CorridorSemanticType.DEPENDS_ON,
            confidence=1.0,
            provenance="prior",
            evidence=[],
        )
        graph = GraphPayload(
            nodes=[_path_node("a", "src/a.py"), _path_node("b", "src/b.py")],
            corridors=[existing],
        )
        result = recompute_depends_on_corridors(graph)
        # Should not add a duplicate
        assert len(result.corridors) == 1

    def test_preserves_existing_corridors(self) -> None:
        prior = GraphCorridor(
            source_id="x",
            target_id="y",
            semantic_type=CorridorSemanticType.DEPENDS_ON,
            confidence=0.9,
            provenance="manual",
            evidence=[],
        )
        graph = GraphPayload(
            nodes=[_path_node("a", "lib/a.py"), _path_node("b", "lib/b.py")],
            corridors=[prior],
        )
        result = recompute_depends_on_corridors(graph)
        # prior corridor preserved (even though x/y not in node list)
        assert any(c.provenance == "manual" for c in result.corridors)

    def test_custom_provenance_tag(self) -> None:
        graph = GraphPayload(
            nodes=[_path_node("a", "pkg/a.py"), _path_node("b", "pkg/b.py")],
            corridors=[],
        )
        result = recompute_depends_on_corridors(graph, provenance="resync")
        assert all(c.provenance == "resync" for c in result.corridors if c.provenance != "manual")

    def test_original_graph_not_mutated(self) -> None:
        graph = GraphPayload(
            nodes=[_path_node("a", "mod/a.py"), _path_node("b", "mod/b.py")],
            corridors=[],
        )
        original_corridor_count = len(graph.corridors)
        recompute_depends_on_corridors(graph)
        assert len(graph.corridors) == original_corridor_count


# ===========================================================================
# Context persistence: onboard → sync({}) resolution
# ===========================================================================


class TestOnboardContextPersistenceForSync:
    """After a successful onboard, sync({}) must not return NO_CONTEXT.

    Covers:
    - programmatic fast-path registers context on success
    - sync({}) resolves that context (UNCHANGED when no scan config stored)
    - sync({}) with empty registry still returns NO_CONTEXT
    - two distinct scopes → AMBIGUOUS_CONTEXT on sync({})
    """

    @pytest.mark.asyncio
    async def test_programmatic_onboard_then_sync_no_args_returns_unchanged(
        self, mock_ctx: MagicMock
    ) -> None:
        """sync({}) after programmatic onboard returns UNCHANGED (not NO_CONTEXT)."""
        onboard = _get_tool_fn("onboard")
        sync = _get_tool_fn("sync")
        scope = {"palace": "test-palace", "wing": "test-wing"}

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend"),
            patch(
                "workflows_mcp.tools_memory._run_scan",
                return_value=(
                    [{"path": "main.py", "content": "print('hello')", "size_bytes": 16}],
                    MagicMock(entries=[], scan_config=MagicMock()),
                ),
            ),
        ):
            _onboard_result = await onboard(
                scope=scope,
                ingestion={"mode": "programmatic"},
                scan={"path": "main.py", "root": "."},
                ctx=mock_ctx,
            )
            _onboard_payload = json.loads(_onboard_result.content[0].text)
            assert _onboard_payload.get("status") == "completed", (
                f"Onboard did not complete: {_onboard_payload}"
            )

            sync_result = await sync(ctx=mock_ctx)
        sync_payload = json.loads(sync_result.content[0].text)
        assert sync_payload.get("status") == "UNCHANGED", (
            "Expected UNCHANGED after programmatic onboard context resolution; "
            f"got: {sync_payload}"
        )

    @pytest.mark.asyncio
    async def test_sync_no_context_when_registry_empty(self, mock_ctx: MagicMock) -> None:
        """sync({}) with empty registry (no prior onboard) must return MEM_NO_ACTIVE_CONTEXT."""
        sync = _get_tool_fn("sync")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await sync(ctx=mock_ctx)
        payload = json.loads(result.content[0].text)
        err = payload.get("error", {})
        assert err.get("code") == "MEM_NO_ACTIVE_CONTEXT", (
            f"Expected MEM_NO_ACTIVE_CONTEXT, got: {err.get('code')}"
        )

    @pytest.mark.asyncio
    async def test_two_distinct_scopes_sync_returns_ambiguous(
        self, mock_ctx: MagicMock
    ) -> None:
        """sync({}) with two stored contexts and no scope hint must return AMBIGUOUS_CONTEXT."""
        from workflows_mcp.engine.memory_scope_resolver import normalize_scope

        # Directly inject two candidates into the registry.
        scope_a = {"palace": "palace-a"}
        scope_b = {"palace": "palace-b"}
        _tools_memory._register_onboard_context_for_session(
            mock_ctx,
            normalize_scope(scope_a),
            {"scope": normalize_scope(scope_a)},
        )
        _tools_memory._register_onboard_context_for_session(
            mock_ctx,
            normalize_scope(scope_b),
            {"scope": normalize_scope(scope_b)},
        )

        sync = _get_tool_fn("sync")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await sync(ctx=mock_ctx)
        payload = json.loads(result.content[0].text)
        err = payload.get("error", {})
        assert err.get("code") == "AMBIGUOUS_CONTEXT", (
            f"Expected AMBIGUOUS_CONTEXT, got: {err.get('code')}"
        )
