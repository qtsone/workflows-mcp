"""Unified memory MCP tool — conditionally registered at startup."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ToolAnnotations
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .context import (
    AppContext,
    AppContextType,
    MemoryBackendUnavailableError,
    SessionProjectContext,
)
from .engine.sql.postgres_backend import PostgresBackend
from .http_models import OnboardRequest, SyncRequest
from .memory.memory_graph_builder import GraphPayload
from .memory.memory_graph_validator import (
    GraphValidationResult,
    build_graph_error_envelope,
    validate_graph_payload,
)
from .memory.memory_onboard_sync_orchestrator import (
    LLMOnboardRequest,
    ProgrammaticOnboardRequest,
    build_llm_onboard_response,
    build_programmatic_onboard_response,
    classify_scan_files_for_programmatic_mode,
    run_llm_onboard,
    run_programmatic_onboard,
    run_programmatic_onboard_with_cycle_recording,
)
from .memory.memory_schema import (
    MemoryRequest,
    MemoryResponseInput,
    MemoryResult,
)
from .memory.memory_scope_resolver import (
    SyncContextCandidate,
    build_ambiguous_context_envelope,
    resolve_sync_context,
    scope_key,
    sorted_scan_manifest,
)
from .memory.memory_service import (
    MemoryContractError,
    MemoryService,
)
from .memory.project_flow_service import (
    PROJECT_FLOW_OPERATIONS,
    PROJECT_FLOW_VERSION,
    FlowState,
    OperationExecutor,
    ProjectFlowService,
    StepSections,
    SyncDelta,
    build_project_checkpoint_payload,
    checkpoint_error,
    classify_deletion_policy,
    compact_checkpoint,
    compute_sync_delta,
    normalize_project_checkpoint,
    project_flow_plan,
    restore_checkpoint,
)
from .memory_runtime import (
    memory_connection_config_from_env,
    memory_connection_config_from_metadata,
)
from .security.filesystem_boundaries import (
    normalize_project_roots,
    validate_path_within_effective_boundary,
)
from .tool_helpers import AUTH_SCOPE_KEY, get_request_scope, json_response

logger = logging.getLogger(__name__)

# Stateless and shared across all sessions — the checkpoint dict is the state.
_PROJECT_FLOW_SERVICE = ProjectFlowService()

# ---------------------------------------------------------------------------
# Session-scoped active context helpers
# ---------------------------------------------------------------------------


def _get_session(ctx: AppContextType) -> Any:
    """Return the underlying session object from an MCP tool context."""
    return ctx.request_context.session


def _get_header_from_scope(scope: dict[str, Any], header_name: bytes) -> str | None:
    headers = dict(scope.get("headers", []))
    raw = headers.get(header_name)
    if raw is None:
        return None
    value = raw.decode("latin-1", errors="replace").strip()
    return value or None


def _prime_session_project_context_from_auth(ctx: AppContextType) -> None:
    """Hydrate per-session allowed/active projects from auth middleware context."""
    scope = get_request_scope(ctx)
    if scope is None:
        return

    auth_ctx = scope.get(AUTH_SCOPE_KEY)
    projects = getattr(auth_ctx, "projects", None)
    if not isinstance(projects, tuple) or not all(
        isinstance(project, SessionProjectContext) for project in projects
    ):
        return

    app_ctx = ctx.request_context.lifespan_context
    session = _get_session(ctx)

    transport_session_id = _get_header_from_scope(scope, b"mcp-session-id")
    binder = getattr(app_ctx, "bind_transport_session", None)
    if transport_session_id and callable(binder):
        auth_token_id = getattr(auth_ctx, "token_id", None)
        binder(transport_session_id, session, token_id=auth_token_id)

    allowed = list(projects)
    app_ctx.register_allowed_projects(session, allowed)

    if len(allowed) == 1 and allowed[0].source == "token_bound":
        app_ctx.set_active_project(session, allowed[0])
        return

    active = app_ctx.get_active_project(session)
    if active is None:
        return
    if any(active.project_id == candidate.project_id for candidate in allowed):
        return

    clearer = getattr(app_ctx, "clear_active_project", None)
    if callable(clearer):
        clearer(session)


def _get_active_context(ctx: AppContextType) -> Any:
    """Return the active SyncContextCandidate for this session, or None."""
    app_ctx = ctx.request_context.lifespan_context
    session = _get_session(ctx)
    return app_ctx.get_active_context(session)


def _set_active_context(ctx: AppContextType, candidate: Any) -> None:
    """Persist the active SyncContextCandidate for this session."""
    app_ctx = ctx.request_context.lifespan_context
    session = _get_session(ctx)
    app_ctx.set_active_context(session, candidate)


def _list_session_onboard_candidates(ctx: AppContextType) -> list[SyncContextCandidate]:
    """Return onboard/sync candidates visible to the current session."""
    app_ctx = ctx.request_context.lifespan_context
    session = _get_session(ctx)
    lister = getattr(app_ctx, "list_onboard_context_candidates", None)
    if callable(lister):
        listed = lister(session)
        if isinstance(listed, list) and all(
            isinstance(item, SyncContextCandidate) for item in listed
        ):
            return listed
    # Fail closed: never fall back to process-global candidate visibility.
    return []


def _register_onboard_context_for_session(
    ctx: AppContextType,
    resolved_scope: dict[str, str | None],
    checkpoint_payload: dict[str, Any],
) -> SyncContextCandidate:
    """Register candidate in current session context only."""
    key = scope_key(resolved_scope)
    candidate = SyncContextCandidate(
        scope=resolved_scope,
        scope_key_value=key,
        checkpoint_data=checkpoint_payload,
        source="stored_checkpoint",
    )

    app_ctx = ctx.request_context.lifespan_context
    register = getattr(app_ctx, "register_onboard_context_candidate", None)
    if callable(register):
        register(_get_session(ctx), candidate)
    return candidate


def _get_active_project(ctx: AppContextType) -> SessionProjectContext | None:
    """Return the active project for this session, or None."""
    _prime_session_project_context_from_auth(ctx)
    app_ctx = ctx.request_context.lifespan_context
    getter = getattr(app_ctx, "get_active_project", None)
    if getter is None:
        return None
    project = getter(_get_session(ctx))
    if isinstance(project, SessionProjectContext):
        return project
    return None


def _enable_watcher_for_active_project_after_onboard(ctx: AppContextType) -> None:
    """Best-effort default watcher enablement for the active project.

    Onboarding success must not fail if watcher manager is unavailable or errors.
    """
    active_project = _get_active_project(ctx)
    if active_project is None:
        return

    app_ctx = ctx.request_context.lifespan_context
    watcher_manager = getattr(app_ctx, "watcher_manager", None)
    if watcher_manager is None:
        return

    enable_by_default = getattr(watcher_manager, "enable_project_by_default", None)
    if not callable(enable_by_default):
        return

    try:
        enable_by_default(active_project.project_id)
    except Exception:
        logger.warning(
            "Failed to enable default watcher after onboard for project_id=%s",
            active_project.project_id,
            exc_info=True,
        )


def _merge_scope_with_active_project(
    scope: dict[str, Any] | None,
    active_project: SessionProjectContext,
) -> dict[str, Any]:
    """Resolve scope with project defaults per Slice D precedence.

    Rules:
    - No explicit scope: use active project's palace/wing/room defaults.
    - Partial scope missing palace: inject palace, wing, room from active defaults.
    - Explicit palace is never overridden.
    - Missing wing/room are only defaulted when explicit palace matches active palace.
    """
    if scope is None:
        return {
            "palace": active_project.palace,
            "wing": active_project.default_wing,
            "room": active_project.default_room,
        }

    merged: dict[str, Any] = dict(scope)
    explicit_palace = merged.get("palace")
    if not explicit_palace:
        merged["palace"] = active_project.palace
        if not merged.get("wing") and active_project.default_wing is not None:
            merged["wing"] = active_project.default_wing
        if not merged.get("room") and active_project.default_room is not None:
            merged["room"] = active_project.default_room
        return merged

    if explicit_palace == active_project.palace:
        if not merged.get("wing") and active_project.default_wing is not None:
            merged["wing"] = active_project.default_wing
        if not merged.get("room") and active_project.default_room is not None:
            merged["room"] = active_project.default_room

    return merged


def _build_no_active_context_envelope() -> dict[str, Any]:
    """Actionable error when memory/sync is called with no scope and no active context."""
    return {
        "error": {
            "code": "MEM_NO_ACTIVE_CONTEXT",
            "message": (
                "No active memory context for this session. "
                "Call onboard() first to initialise a project context, "
                "or pass an explicit 'scope' argument."
            ),
            "retryable": False,
            "actionable_fix": (
                "Run onboard(scope={...}, ingest={...}) to create a context, "
                "or use select(scope={...}) to switch to an existing one, "
                "or pass scope={...} directly to this call."
            ),
        }
    }


def _memory_backend_unavailable_envelope(app_ctx: AppContext) -> dict[str, Any] | None:
    """Return actionable fail-closed envelope when memory backend is unavailable."""
    unavailable = app_ctx.memory_backend_unavailable_error
    if unavailable is None or not isinstance(unavailable, MemoryBackendUnavailableError):
        return None

    return {
        "error": {
            "code": unavailable.code,
            "message": unavailable.message,
            "retryable": unavailable.retryable,
            "actionable_fix": unavailable.actionable_fix,
        }
    }


# ---------------------------------------------------------------------------
# Legacy in-process onboarding registry retained for tests/compatibility only.
# Runtime session resolution must not read/write this process-global store.
# ---------------------------------------------------------------------------
_onboard_context_registry: dict[str, SyncContextCandidate] = {}


# ============================================================================
# Scan config and snapshot Pydantic models
# ============================================================================


class ScanConfig(BaseModel, extra="forbid"):
    """Configuration for file-system scan used by onboard / sync."""

    path: str | None = Field(
        default=None,
        description="Optional single file path. Mutually exclusive with patterns.",
    )
    patterns: list[str] = Field(
        default_factory=list,
        description="Glob patterns for files to scan (e.g. ['src/**/*.py', 'docs/**/*.md'])",
    )
    root: str = Field(
        default=".",
        description="Base directory for glob expansion (absolute or relative).",
    )
    exclude_patterns: list[str] = Field(
        default_factory=list,
        description="Additional patterns to exclude (beyond built-in defaults).",
    )
    max_files: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of files to scan.",
    )
    max_size_kb: int = Field(
        default=100,
        ge=1,
        le=10240,
        description="Maximum individual file size in KB.",
    )
    respect_gitignore: bool = Field(
        default=True,
        description="Whether to respect .gitignore patterns.",
    )
    mode: Literal["full", "outline", "summary"] = Field(
        default="full",
        description="Read mode for scanned files.",
    )
    deletion_policy: Literal[
        "archive", "supersede", "ignore", "mark_missing", "archive_missing"
    ] = Field(
        default="archive",
        description=(
            "How to handle files present in snapshot but absent on re-scan. "
            "archive/mark_missing/archive_missing: mark inactive. "
            "supersede: replace. ignore: do nothing."
        ),
    )

    @field_validator("patterns", mode="before")
    @classmethod
    def _normalize_patterns(cls, value: Any) -> Any:
        if value is None:
            return []
        return value

    @model_validator(mode="after")
    def _validate_path_or_patterns(self) -> ScanConfig:
        has_path = self.path is not None
        has_patterns = len(self.patterns) > 0
        if has_path and has_patterns:
            raise ValueError("scan.path and scan.patterns are mutually exclusive")
        if not has_path and not has_patterns:
            raise ValueError("scan requires either path or patterns")
        return self


class FileSnapshotEntry(BaseModel, extra="forbid"):
    """Metadata snapshot for a single scanned file."""

    path: str = Field(description="Relative file path (from root).")
    size_bytes: int = Field(description="File size in bytes at scan time.")
    mtime_ns: int = Field(description="File mtime in nanoseconds at scan time.")
    content_hash: str = Field(description="SHA-256 hex digest of file content.")
    memory_id: str | None = Field(
        default=None,
        description="Memory ID stored for this file (populated after successful ingest).",
    )


class ScanSnapshot(BaseModel, extra="forbid"):
    """Snapshot of all files captured during a scan pass."""

    scan_config: ScanConfig = Field(description="Scan config used to produce this snapshot.")
    entries: list[FileSnapshotEntry] = Field(
        default_factory=list,
        description="One entry per scanned file.",
    )


def _merge_dict_prefer_explicit(
    generated: dict[str, Any],
    explicit: dict[str, Any] | None,
) -> dict[str, Any]:
    if explicit is None:
        return generated
    return {**generated, **explicit}


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _hash_file_bytes(abs_path: Path) -> str:
    """Compute SHA-256 of raw file bytes (encoding-independent)."""
    h = hashlib.sha256()
    with abs_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_scan_root() -> Path:
    """Return the single allowed scan root directory.

    Defaults to ``Path('/')`` (filesystem root) when ``WORKFLOWS_SCAN_ROOT``
    is unset or blank.  Override by setting ``WORKFLOWS_SCAN_ROOT`` to a single
    absolute path — any subfolder under that path will be accepted.
    """
    raw = os.environ.get("WORKFLOWS_SCAN_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path("/")


def _validate_scan_path_within_workspace(resolved: Path, label: str) -> None:
    """Raise MemoryContractError when *resolved* is outside the allowed scan root.

    The allowed root is ``Path('/')`` by default, or the path set in the
    ``WORKFLOWS_SCAN_ROOT`` environment variable (single path, no list parsing).
    """
    root = _get_scan_root()
    if validate_path_within_effective_boundary(
        resolved,
        global_root=root,
        project_roots=None,
    ):
        return
    raise MemoryContractError(
        code="MEM_SCAN_PATH_OUT_OF_ROOT",
        message=(
            f"MEM_SCAN_PATH_OUT_OF_ROOT: {label} must be inside the allowed scan root "
            f"({root}); got {resolved}. "
            f"Set WORKFLOWS_SCAN_ROOT to a single directory path to override."
        ),
        retryable=False,
    )


def _validate_scan_path_with_project_boundary(
    resolved: Path,
    label: str,
    active_project: SessionProjectContext | None,
) -> None:
    """Validate scan path against active project boundaries when configured.

    Enforced only when an active project with fs_root is present.
    """
    global_root = _get_scan_root()
    project_roots = None
    if active_project is not None and active_project.fs_root:
        project_roots = normalize_project_roots(
            fs_root=active_project.fs_root,
            fs_allowlist=active_project.fs_allowlist,
        )

    if validate_path_within_effective_boundary(
        resolved,
        global_root=global_root,
        project_roots=project_roots,
    ):
        return

    if project_roots:
        roots_rendered = ", ".join(str(root) for root in project_roots)
        message = (
            f"MEM_SCAN_PATH_OUT_OF_ROOT: {label} must be inside project boundary "
            f"([{roots_rendered}]) and global scan root ({global_root}); got {resolved}."
        )
    else:
        message = (
            f"MEM_SCAN_PATH_OUT_OF_ROOT: {label} must be inside the allowed scan root "
            f"({global_root}); got {resolved}. "
            "Set WORKFLOWS_SCAN_ROOT to a single directory path to override."
        )

    raise MemoryContractError(
        code="MEM_SCAN_PATH_OUT_OF_ROOT",
        message=message,
        retryable=False,
    )


async def _run_scan(
    scan_config: ScanConfig,
    *,
    active_project: SessionProjectContext | None = None,
) -> tuple[list[dict[str, Any]], ScanSnapshot]:
    """Run a file scan using the ReadFiles executor helper.

    Returns:
        (scanned_files, snapshot) where scanned_files is a list of dicts with
        keys 'path', 'content', 'size_bytes' and snapshot contains metadata.

    Raises:
        MemoryContractError: When scan.root or path resolves outside workspace root.
    """
    from .engine.executors_file import run_readfiles_scan

    # --- path safety: validate root is inside workspace root ---
    base = Path(scan_config.root).expanduser().resolve()
    _validate_scan_path_with_project_boundary(base, "scan.root", active_project)

    if scan_config.path is not None:
        single_path = (base / scan_config.path).resolve()
        _validate_scan_path_with_project_boundary(single_path, "scan.path", active_project)

    try:
        scanned_files = await run_readfiles_scan(
            path=scan_config.path,
            patterns=scan_config.patterns,
            base_path=scan_config.root,
            exclude_patterns=scan_config.exclude_patterns,
            max_files=scan_config.max_files,
            max_size_kb=scan_config.max_size_kb,
            respect_gitignore=scan_config.respect_gitignore,
            mode=scan_config.mode,
        )
    except FileNotFoundError as exc:
        raise MemoryContractError(
            code="MEM_SCAN_NO_FILES_MATCHED",
            message=f"MEM_SCAN_NO_FILES_MATCHED: {exc}",
            retryable=False,
        ) from exc
    except ValueError as exc:
        raise MemoryContractError(
            code="MEM_SCAN_INVALID_CONFIG",
            message=f"MEM_SCAN_INVALID_CONFIG: {exc}",
            retryable=False,
        ) from exc

    entries: list[FileSnapshotEntry] = []
    for f in scanned_files:
        rel_path = f["path"]
        abs_path = (base / rel_path).resolve()
        _validate_scan_path_with_project_boundary(
            abs_path,
            f"scan.result:{rel_path}",
            active_project,
        )
        try:
            stat = abs_path.stat()
            mtime_ns = stat.st_mtime_ns
            size_bytes = stat.st_size
            # Fix 4: hash from raw bytes (encoding-independent, avoids round-trip drift)
            content_hash = _hash_file_bytes(abs_path)
        except OSError:
            mtime_ns = 0
            size_bytes = f.get("size_bytes", 0)
            content_hash = _hash_content(f["content"])
        entries.append(
            FileSnapshotEntry(
                path=rel_path,
                size_bytes=size_bytes,
                mtime_ns=mtime_ns,
                content_hash=content_hash,
            )
        )

    snapshot = ScanSnapshot(scan_config=scan_config, entries=entries)
    # Phase 2: always return scan manifest in deterministic (sorted) order.
    scanned_files = sorted_scan_manifest(scanned_files)
    return scanned_files, snapshot


def _build_ingest_from_files(
    scanned_files: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Build a minimal ingest payload from scanned file list.

    Returns:
        (payload, ingested_paths) where ingested_paths is the ordered list of file paths
        corresponding to each entry in payload["memories"], enabling memory_id back-propagation.
    """
    memories = []
    ingested_paths: list[str] = []
    for f in scanned_files:
        if f.get("content", "").strip():
            memories.append({"content": f["content"], "metadata": {"path": f["path"]}})
            ingested_paths.append(f["path"])
    return {"format": "structured", "memories": memories}, ingested_paths


def _update_snapshot_with_memory_ids(
    snapshot: ScanSnapshot,
    ingested_paths: list[str],
    memory_ids: list[str],
) -> ScanSnapshot:
    """Return a new ScanSnapshot with memory_ids back-populated from ingest results.

    Maps memory_ids[i] to the snapshot entry whose path matches ingested_paths[i].
    Entries with no matching ingested path are left unchanged.
    """
    if not ingested_paths or not memory_ids:
        return snapshot

    path_to_id: dict[str, str] = {}
    for i, path in enumerate(ingested_paths):
        if i < len(memory_ids):
            path_to_id[path] = memory_ids[i]

    updated_entries = [
        FileSnapshotEntry(
            path=entry.path,
            size_bytes=entry.size_bytes,
            mtime_ns=entry.mtime_ns,
            content_hash=entry.content_hash,
            memory_id=path_to_id.get(entry.path, entry.memory_id),
        )
        for entry in snapshot.entries
    ]
    return ScanSnapshot(scan_config=snapshot.scan_config, entries=updated_entries)


def _make_scan_snapshot_backfill(
    snapshot: ScanSnapshot | None,
    ingested_paths: list[str],
) -> Callable[[dict[str, Any]], dict[str, Any] | None] | None:
    """Build the ``FlowState.on_ingest`` callback for scan-driven flows.

    After the ingest step returns its memory ids, fold them back into the scan
    snapshot so a later sync can target deletions by id. Returns ``None`` when
    there is no scan snapshot to back-populate (the snapshot stays untouched).
    """
    if snapshot is None or not ingested_paths:
        return None

    def _backfill(step_result: dict[str, Any]) -> dict[str, Any]:
        raw_ids = step_result.get("ids") or step_result.get("id")
        returned_ids: list[str] = []
        if isinstance(raw_ids, list):
            returned_ids = [str(x) for x in raw_ids]
        elif isinstance(raw_ids, str):
            returned_ids = [raw_ids]
        updated = _update_snapshot_with_memory_ids(snapshot, ingested_paths, returned_ids)
        return updated.model_dump()

    return _backfill


def _compute_scan_delta(
    snapshot: ScanSnapshot,
    new_snapshot: ScanSnapshot,
) -> tuple[list[str], list[str], list[str]]:
    """Compute added, modified, deleted relative paths between two snapshots.

    Delegates to the canonical ``compute_sync_delta`` in ``project_flow_service``
    so that the four-class delta (added/modified/deleted/unchanged) is always
    computed consistently.  This wrapper preserves the existing (added, modified,
    deleted) three-tuple return type for backwards-compatible call sites.

    Returns:
        (added, modified, deleted) lists of relative paths.
    """
    prior_entries = [{"path": e.path, "content_hash": e.content_hash} for e in snapshot.entries]
    new_entries = [{"path": e.path, "content_hash": e.content_hash} for e in new_snapshot.entries]
    delta = compute_sync_delta(prior_entries, new_entries)
    return delta.added, delta.modified, delta.deleted


def _compute_full_scan_delta(
    snapshot: ScanSnapshot,
    new_snapshot: ScanSnapshot,
) -> SyncDelta:
    """Compute a full four-class SyncDelta between two ScanSnapshots.

    Returns the complete SyncDelta (added/modified/deleted/unchanged) for
    use in Phase 6 idempotency checks and debug diagnostics.
    """
    prior_entries = [{"path": e.path, "content_hash": e.content_hash} for e in snapshot.entries]
    new_entries = [{"path": e.path, "content_hash": e.content_hash} for e in new_snapshot.entries]
    return compute_sync_delta(prior_entries, new_entries)


def _tool_error_payload(
    tool: str,
    err: Exception,
    *,
    stage: str | None = None,
) -> dict[str, Any]:
    correlation_id = str(uuid.uuid4())
    actionable_fix: str | None = None

    if isinstance(err, MemoryContractError):
        code = err.code
        message = err.message
        retryable = err.retryable
        actionable_fix = err.actionable_fix
        logger.warning(
            "memory tool contract error tool=%s code=%s retryable=%s correlation_id=%s",
            tool,
            code,
            retryable,
            correlation_id,
        )
    elif isinstance(err, ValidationError):
        contract_error: MemoryContractError | None = None
        for item in err.errors(include_url=False):
            ctx = item.get("ctx")
            if not isinstance(ctx, dict):
                continue
            nested = ctx.get("error")
            if isinstance(nested, MemoryContractError):
                contract_error = nested
                break
        if contract_error is not None:
            code = contract_error.code
            message = contract_error.message
            retryable = contract_error.retryable
            actionable_fix = contract_error.actionable_fix
            logger.warning(
                (
                    "memory tool contract validation error "
                    "tool=%s code=%s retryable=%s correlation_id=%s"
                ),
                tool,
                code,
                retryable,
                correlation_id,
            )
        else:
            code = "MEM_SCHEMA_VALIDATION_FAILED"
            # Surface field-level detail so the caller can identify which field
            # and constraint failed without consulting external schema docs.
            field_errors = []
            for item in err.errors(include_url=False):
                loc = ".".join(str(p) for p in item.get("loc", ())) or "(root)"
                msg = item.get("msg", "")
                field_errors.append(f"{loc}: {msg}")
            if field_errors:
                message = "Request schema validation failed — " + "; ".join(field_errors)
                actionable_fix = "Fix the following field(s) and retry: " + "; ".join(field_errors)
            else:
                message = "Request schema validation failed"
                actionable_fix = "Check required fields and types against the schema."
            retryable = False
            logger.warning(
                "memory tool schema validation failed tool=%s code=%s correlation_id=%s",
                tool,
                code,
                correlation_id,
            )
    else:
        code = "MEM_INTERNAL_ERROR"
        message = f"{tool} failed"
        retryable = False
        logger.exception(
            "memory tool internal error tool=%s code=%s correlation_id=%s",
            tool,
            code,
            correlation_id,
        )

    error_body: dict[str, Any] = {
        "code": code,
        "message": message,
        "retryable": retryable,
        "correlation_id": correlation_id,
        "stage": stage,
        "actionable_fix": actionable_fix,
    }
    return {"error": error_body}


def _error_envelope(
    *,
    code: str,
    message: str,
    retryable: bool = False,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        }
    }
    if correlation_id is not None:
        envelope["error"]["correlation_id"] = correlation_id
    return envelope


def _shape_scope_fields(result: MemoryResult) -> dict[str, Any]:
    if result.resolved_scope is None:
        return {}
    return {
        "resolved_scope": result.resolved_scope.model_dump(exclude_none=False),
        "scope_source": result.scope_source,
    }


def _relabel_scope_source_active_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Replace all scope_source field values with 'active_context'.

    Called when the effective scope came from the session's active context fallback
    (not an explicit caller-provided scope/scope_token/context_id).  The service
    layer labels every resolved scope field as 'request' because the scope dict
    was injected as if it were a caller request — which is misleading.

    This function rewrites each field's source label to 'active_context' so the
    caller can distinguish implicit-context resolution from explicit scope passing.
    The outer response shape and all other fields are preserved unchanged.
    """
    scope_source = payload.get("scope_source")
    if not isinstance(scope_source, dict) or not scope_source:
        return payload
    return {
        **payload,
        "scope_source": {k: "active_context" for k in scope_source},
    }


def _lean_memory_item(m: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {"content": m.get("content", "")}
    if m.get("path"):
        item["path"] = m["path"]
    if m.get("source"):
        item["source"] = m["source"]
    return item


def _shape_memory_response(result: MemoryResult, response: MemoryResponseInput) -> dict[str, Any]:
    if response.debug:
        return result.model_dump(by_alias=True)

    scope_fields = _shape_scope_fields(result)

    if result.query is not None:
        q = result.query
        diagnostics = q.diagnostics if isinstance(q.diagnostics, dict) else {}
        strategy = diagnostics.get("strategy")

        if strategy == "graph":
            out: dict[str, Any] = {"paths": q.paths, "diagnostics": q.diagnostics}
            if q.evidence:
                first = q.evidence[0]
                if "nodes" in first:
                    out["nodes"] = first["nodes"]
                if "edges" in first:
                    out["edges"] = first["edges"]
            return {**out, **scope_fields}

        payload: dict[str, Any] = {}
        if q.facts:
            payload["facts"] = [_lean_memory_item(item) for item in q.facts]
        if q.memories:
            payload["memories"] = [_lean_memory_item(item) for item in q.memories]
        if q.communities:
            payload["communities"] = [{"content": c.get("content", "")} for c in q.communities]
        if strategy in {"topology", "evidence"}:
            payload["diagnostics"] = q.diagnostics
            payload["evidence"] = q.evidence
        if result.merge is not None:
            payload["merge"] = result.merge.model_dump(by_alias=True)
        if not payload:
            return {"found": False, **scope_fields}
        return {**payload, **scope_fields}

    if result.manage is not None:
        m = result.manage
        if not m.success:
            return {
                **_error_envelope(
                    code="MEM_OPERATION_FAILED",
                    message=m.error or "operation failed",
                    retryable=False,
                ),
                **scope_fields,
            }

        op = result.operation
        if op == "ingest":
            ids = m.memory_ids
            base: dict[str, Any] = {"stored": m.stored_count}
            if len(ids) == 1:
                base["id"] = ids[0]
            else:
                base["ids"] = ids
            if m.entity_ids:
                base["entity_ids"] = m.entity_ids
            if m.relation_ids:
                base["relation_ids"] = m.relation_ids
            if m.entities_stored_count:
                base["entities_stored"] = m.entities_stored_count
            if m.relations_stored_count:
                base["relations_stored"] = m.relations_stored_count
            return {**base, **scope_fields}
        if op == "validate":
            return {"validated": m.validated_count, **scope_fields}
        if op == "supersede":
            return {"superseded": len(m.superseded_ids), **scope_fields}
        if op == "archive":
            archive_out = {"archived": m.archived_count}
            if m.skipped_count:
                archive_out["skipped"] = m.skipped_count
            return {**archive_out, **scope_fields}
        if op == "maintain":
            maintain_out: dict[str, Any] = {}
            if m.communities_updated:
                maintain_out["communities_updated"] = m.communities_updated
            if m.assessed_count:
                maintain_out["assessed_count"] = m.assessed_count
            if m.expired_count:
                maintain_out["expired"] = m.expired_count
            if m.resolved_count:
                maintain_out["resolved"] = m.resolved_count
            if m.needs_review:
                maintain_out["needs_review_count"] = len(m.needs_review)
            if m.auto_archive_ids:
                maintain_out["auto_archived"] = len(m.auto_archive_ids)
            if response.include_candidates and m.prune_candidates:
                maintain_out["candidates"] = m.prune_candidates
            return {**(maintain_out or {"status": "ok"}), **scope_fields}
        if op == "graph_upsert":
            if m.entity_id:
                return {"entity_id": m.entity_id, **scope_fields}
            if m.relation_id:
                return {"relation_id": m.relation_id, **scope_fields}
            return {"status": "ok", **scope_fields}
        if op == "graph_delete":
            if m.deleted_entity_count:
                return {"deleted_places": m.deleted_entity_count, **scope_fields}
            return {"deleted_links": m.deleted_relation_count, **scope_fields}
        return {**m.model_dump(), **scope_fields}

    return {
        **_error_envelope(
            code="MEM_EMPTY_RESULT",
            message="empty result",
            retryable=False,
        ),
        **scope_fields,
    }


def _get_standalone_user_context() -> tuple[uuid.UUID | None, str | None, str]:
    import getpass

    from .memory.memory_service import SYSTEM_USER_UUID

    for env_var in [
        "MEMORY_USER_ID",
        "WORKFLOWS_USER_ID",
        "WORKFLOWS_USER",
        "MCP_USER_ID",
        "USER",
        "USERNAME",
        "LOGNAME",
    ]:
        if os_user := os.environ.get(env_var):
            os_user = os_user.strip()
            try:
                return (uuid.UUID(os_user), os_user, "ENV_UUID")
            except ValueError:
                det_uuid = uuid.uuid5(uuid.NAMESPACE_OID, f"workflows-user:{os_user}")
                return (det_uuid, os_user, "OS_USER")

    try:
        os_user = getpass.getuser()
        det_uuid = uuid.uuid5(uuid.NAMESPACE_OID, f"workflows-user:{os_user}")
        return (det_uuid, os_user, "OS_USER")
    except Exception:
        return (SYSTEM_USER_UUID, "system", "SYSTEM")


def _has_configured_memory_connection(app_ctx: Any) -> bool:
    if getattr(app_ctx, "memory_backend", None) is not None:
        return True
    if memory_connection_config_from_env() is not None:
        return True
    return memory_connection_config_from_metadata(app_ctx) is not None


def _create_memory_execution(ctx: AppContextType) -> Any:
    from .engine.execution import Execution

    app_ctx = ctx.request_context.lifespan_context
    if app_ctx.get_user_context:
        user_uuid, user_string, auth_method = app_ctx.get_user_context()
    else:
        user_uuid, user_string, auth_method = _get_standalone_user_context()

    exec_context = app_ctx.create_execution_context(user_id=user_uuid, auth_method=auth_method)
    if user_string:
        exec_context.user_string_id = user_string

    execution = Execution()
    execution.set_execution_context(exec_context)
    return execution


async def _execute_memory_request(
    *,
    app_ctx: Any,
    execution: Any,
    operation: str,
    scope: dict[str, Any] | None,
    scope_token: str | None,
    context_id: str | None,
    query: dict[str, Any] | None,
    record: dict[str, Any] | None,
    graph: dict[str, Any] | None,
    maintenance: dict[str, Any] | None,
    response: dict[str, Any] | None,
) -> dict[str, Any]:
    shared_backend = getattr(app_ctx, "memory_backend", None)
    shared_backend_lock = getattr(app_ctx, "memory_backend_lock", None)
    uses_ephemeral_backend = shared_backend is None
    backend = shared_backend if shared_backend is not None else PostgresBackend()

    try:
        if uses_ephemeral_backend:
            config = (
                memory_connection_config_from_metadata(app_ctx)
                or memory_connection_config_from_env()
            )
            if config is None:
                raise MemoryContractError(
                    code="MEMORY_BACKEND_UNAVAILABLE",
                    message=(
                        "MEMORY_BACKEND_UNAVAILABLE: no memory PostgreSQL backend is "
                        "configured. Save database settings or set MEMORY_DB_HOST."
                    ),
                    retryable=False,
                )
            await backend.connect(config)
            from .memory.knowledge.schema import ensure_schema

            await ensure_schema(backend)

        service = MemoryService(backend, execution)
        request = MemoryRequest.model_validate(
            {
                "operation": operation,
                "scope": scope or {},
                "scope_token": scope_token,
                "context_id": context_id,
                "query": query,
                "record": record,
                "graph": graph,
                "maintenance": maintenance,
                "response": response or {},
            }
        )

        if shared_backend is not None and shared_backend_lock is not None:
            async with shared_backend_lock:
                result = await service.execute(request)
        else:
            result = await service.execute(request)

        response_cfg = MemoryResponseInput.model_validate(response or {})
        return _shape_memory_response(result, response_cfg)
    finally:
        if uses_ephemeral_backend:
            await backend.disconnect()


def _build_step_executor(
    *,
    app_ctx: Any,
    execution: Any,
    response: dict[str, Any] | None,
) -> OperationExecutor:
    """Adapt the per-step memory backend call to the ProjectFlowService port.

    Captures the transport-bound parameters (backend, execution, response
    shaping) so the service sees only ``(operation, scope, sections)``.
    """

    async def _executor(
        operation: str, scope: dict[str, Any], sections: StepSections
    ) -> dict[str, Any]:
        query_payload, record_payload, maintenance_payload = sections
        return await _execute_memory_request(
            app_ctx=app_ctx,
            execution=execution,
            operation=operation,
            scope=scope,
            scope_token=None,
            context_id=None,
            query=query_payload,
            record=record_payload,
            graph=None,
            maintenance=maintenance_payload,
            response=response,
        )

    return _executor


async def persist_graph_payload(
    *,
    app_ctx: Any,
    execution: Any,
    scope: dict[str, Any] | None,
    graph: GraphPayload,
    response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a validated onboard graph payload through graph_upsert operations."""
    entity_ids_by_node_id: dict[str, str] = {}
    relation_ids: list[str] = []

    for node in graph.nodes:
        result = await _execute_memory_request(
            app_ctx=app_ctx,
            execution=execution,
            operation="graph_upsert",
            scope=scope,
            scope_token=None,
            context_id=None,
            query=None,
            record=None,
            graph={
                "kind": "place",
                "place_name": node.label,
                "place_type": node.node_type.value,
            },
            maintenance=None,
            response=response,
        )
        entity_id = result.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            raise MemoryContractError(
                code="MEM_GRAPH_PERSIST_FAILED",
                message=(
                    "MEM_GRAPH_PERSIST_FAILED: graph node persistence did not return "
                    f"an entity id for {node.node_id!r}"
                ),
                retryable=False,
            )
        entity_ids_by_node_id[node.node_id] = entity_id

    for corridor in graph.corridors:
        source_entity_id = entity_ids_by_node_id.get(corridor.source_id)
        target_entity_id = entity_ids_by_node_id.get(corridor.target_id)
        if source_entity_id is None or target_entity_id is None:
            raise MemoryContractError(
                code="MEM_GRAPH_PERSIST_FAILED",
                message=(
                    "MEM_GRAPH_PERSIST_FAILED: graph corridor references a node "
                    "that was not persisted"
                ),
                retryable=False,
            )
        graph_payload: dict[str, Any] = {
            "kind": "link",
            "from": source_entity_id,
            "to": target_entity_id,
            "link_type": corridor.semantic_type.value,
            "curated": not bool(corridor.evidence),
        }
        if corridor.evidence:
            graph_payload["evidence_memory_ids"] = list(corridor.evidence)

        result = await _execute_memory_request(
            app_ctx=app_ctx,
            execution=execution,
            operation="graph_upsert",
            scope=scope,
            scope_token=None,
            context_id=None,
            query=None,
            record=None,
            graph=graph_payload,
            maintenance=None,
            response=response,
        )
        relation_id = result.get("relation_id")
        if not isinstance(relation_id, str) or not relation_id:
            raise MemoryContractError(
                code="MEM_GRAPH_PERSIST_FAILED",
                message=(
                    "MEM_GRAPH_PERSIST_FAILED: graph corridor persistence did not "
                    "return a relation id"
                ),
                retryable=False,
            )
        relation_ids.append(relation_id)

    return {
        "nodes": len(entity_ids_by_node_id),
        "corridors": len(relation_ids),
        "entity_ids": list(entity_ids_by_node_id.values()),
        "relation_ids": relation_ids,
    }


async def _persist_graph_payload_if_configured(
    *,
    app_ctx: Any,
    execution: Any,
    scope: dict[str, Any] | None,
    graph: GraphPayload | None,
    response: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if graph is None or not _has_configured_memory_connection(app_ctx):
        return None
    return await persist_graph_payload(
        app_ctx=app_ctx,
        execution=execution,
        scope=scope,
        graph=graph,
        response=response,
    )


def _validate_graph_step_payload(
    step_payload: dict[str, Any],
    *,
    stage: str = "graph_validation",
) -> dict[str, Any] | None:
    """Validate a graph step payload from an onboard/sync checkpoint plan.

    Returns a deterministic error envelope dict when the payload fails
    validation (code=GRAPH_COMPLETENESS_FAILED), or None when valid.

    Phase 3: this is the atomic persistence boundary gate — graph payloads
    that fail validation are rejected before any DB write is attempted.
    """
    # Graph step payloads may embed a "graph" sub-key or be a raw graph dict.
    graph_data = step_payload.get("graph") if "graph" in step_payload else step_payload
    if not isinstance(graph_data, dict):
        return None  # Not a graph payload — no validation needed here.
    # Only validate if the dict contains "nodes" or "corridors" keys (batch payload).
    if "nodes" not in graph_data and "corridors" not in graph_data:
        return None

    result: GraphValidationResult = validate_graph_payload(graph_data)
    if result.valid:
        return None
    envelope = result.error_envelope
    if isinstance(envelope, dict) and "error" in envelope:
        err = dict(envelope["error"])
        err.setdefault("stage", stage)
        err.setdefault(
            "actionable_fix",
            (
                "Fix graph payload: ensure required node types and corridor fields are present, "
                "remove illegal same-level links, and eliminate orphan nodes."
            ),
        )
        return {"error": err}
    return build_graph_error_envelope(
        code="GRAPH_COMPLETENESS_FAILED",
        message="Graph payload failed completeness validation.",
        stage=stage,
        actionable_fix=(
            "Fix graph payload: ensure required node types and corridor fields are present, "
            "remove illegal same-level links, and eliminate orphan nodes."
        ),
        violations=result.violations,
    )


def _is_debug_response(response: dict[str, Any] | None) -> bool:
    """Return True when the caller requested debug-level project flow output."""
    if response is None:
        return False
    return bool(response.get("debug"))


def memory_schema_payload() -> dict[str, Any]:
    """Return a static contract/schema snapshot for the memory tool.

    Does not require DB connectivity.  Useful for agents to discover available
    operations, checkpoint format, and scan configuration options at runtime.
    Callable from the HTTP transport layer as well as the MCP tool layer.
    """
    return {
        "version": PROJECT_FLOW_VERSION,
        "operations": [
            "query",
            "ingest",
            "validate",
            "supersede",
            "archive",
            "maintain",
            "graph_upsert",
            "graph_delete",
            "schema",
        ],
        "query": {
            "description": (
                "Search and retrieve memories. Pass a 'query' object as the 'query' parameter "
                "to memory(operation='query', query={...})."
            ),
            "supported_keys": [
                "text",
                "mode",
                "limit",
                "min_score",
                "filter",
                "include_communities",
                "include_facts",
            ],
            "examples": [
                {"query": {"text": "incident response runbook"}},
                {"query": {"text": "API rate limit", "mode": "search", "limit": 10}},
                {"query": {"text": "auth flow", "mode": "evidence"}},
                {"query": {"text": "topology", "mode": "graph"}},
            ],
            "notes": (
                "The 'text' key is required for all search modes. "
                "'mode' defaults to 'search' when omitted. "
                "Supported modes: search, evidence, graph."
            ),
        },
        "scan": {
            "fields": {
                "path": "Optional single file path (mutually exclusive with patterns).",
                "patterns": "Glob patterns for files to scan (e.g. ['src/**/*.py']).",
                "root": "Base directory for glob expansion (absolute or relative).",
                "exclude_patterns": "Additional exclude patterns beyond built-in defaults.",
                "max_files": "Maximum number of files to scan (1-100, default 20).",
                "max_size_kb": "Maximum individual file size in KB (1-10240, default 100).",
                "respect_gitignore": "Whether to respect .gitignore patterns (default true).",
                "mode": "Read mode: full | outline | summary (default full).",
                "deletion_policy": (
                    "How to handle files absent on re-scan: archive | supersede | ignore."
                ),
            },
            "deletion_policies": ["archive", "supersede", "ignore"],
            "scan_root": (
                "scan.root must be inside the allowed scan root. "
                "Default root is '/' (filesystem root). "
                "Override via WORKFLOWS_SCAN_ROOT (single path, no list parsing)."
            ),
        },
        "checkpoint": {
            "version": PROJECT_FLOW_VERSION,
            "fields": {
                "version": "Checkpoint format version string (must match server version).",
                "scope": "Project scope dict passed through all steps.",
                "plan": "List of {operation, payload} steps to execute.",
                "next_index": "Index of next step to execute (0-based).",
                "completed": (
                    "List of completed {operation} entries (result stripped in compact mode)."
                ),
                "scan": "ScanConfig used for file-system scans (optional).",
                "scan_snapshot": (
                    "ScanSnapshot from last scan pass (optional, entries preserved for delta)."
                ),
            },
            "flow_operations": list(PROJECT_FLOW_OPERATIONS),
            "response_debug": {
                "description": (
                    "Pass response={'debug': True} to onboard or sync to "
                    "receive full checkpoint internals. Default (debug=False or omitted): "
                    "compact mode strips completed[].result blobs and "
                    "plan[i].payload.memories[].content for completed steps (index < next_index). "
                    "Pending steps keep full payload for resume. scan_snapshot.entries are always "
                    "preserved (required for sync delta computation). "
                    "The outer 'results' list is only included when debug=True."
                ),
                "compact_default": [
                    "completed[].result blobs stripped",
                    "plan[i<next_index].payload.memories[].content stripped",
                    "outer results[] omitted",
                ],
                "debug_full": [
                    "completed[].result included",
                    "plan[].payload fully intact",
                    "outer results[] included",
                ],
            },
        },
    }


def register_memory_tools(
    mcp_server: FastMCP,
    *,
    enable_project_tools: bool = True,
) -> None:
    """Register unified memory MCP tool (and optional project flow tools)."""

    @mcp_server.tool(
        description=(
            "Query and update memory in one tool (search, ingest, maintain, validate, "
            "and graph updates). Use this when you need direct memory operations."
        ),
        annotations=ToolAnnotations(
            title="Memory",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        ),
    )
    async def memory(
        operation: Annotated[
            str,
            Field(
                description=(
                    "Operation: query, ingest, validate, supersede, archive, maintain, "
                    "graph_upsert, graph_delete, schema"
                )
            ),
        ],
        scope: Annotated[dict[str, Any] | None, Field(default=None)] = None,
        scope_token: Annotated[str | None, Field(default=None)] = None,
        context_id: Annotated[str | None, Field(default=None)] = None,
        query: Annotated[dict[str, Any] | None, Field(default=None)] = None,
        record: Annotated[dict[str, Any] | None, Field(default=None)] = None,
        graph: Annotated[dict[str, Any] | None, Field(default=None)] = None,
        maintenance: Annotated[dict[str, Any] | None, Field(default=None)] = None,
        response: Annotated[dict[str, Any] | None, Field(default=None)] = None,
        *,
        ctx: AppContextType,
    ) -> CallToolResult:
        """Run one memory operation and return compact JSON results."""
        # Schema operation short-circuits before any DB connectivity.
        if operation == "schema":
            return json_response(memory_schema_payload())

        app_ctx = ctx.request_context.lifespan_context
        unavailable = _memory_backend_unavailable_envelope(app_ctx)
        if unavailable is not None:
            return json_response(unavailable)

        # Scope precedence for Slice D:
        # 1) scope_token/context_id (handled in service) — never overridden here.
        # 2) explicit scope (merged with active project defaults per rules below).
        # 3) active project defaults.
        # 4) active context fallback (onboard/select scope context).
        effective_scope = scope
        _scope_from_active_context = False
        # Direct memory() locality source policy:
        # - scope_token/context_id are handled in MemoryService and are never overridden here.
        # - query may use active project defaults or active context for broad reads.
        # - non-query operations reach MemoryService without session fallback so the
        #   operation locality contract decides whether scope-less execution is valid.
        if scope_token is None and context_id is None and operation == "query":
            active_project = _get_active_project(ctx)
            if active_project is not None:
                effective_scope = _merge_scope_with_active_project(effective_scope, active_project)

            if effective_scope is None:
                active = _get_active_context(ctx)
                if active is not None:
                    effective_scope = active.scope
                    _scope_from_active_context = True
                else:
                    # No active context/project and no explicit scope — actionable error.
                    return json_response(_build_no_active_context_envelope())

        execution = _create_memory_execution(ctx)
        try:
            payload = await _execute_memory_request(
                app_ctx=app_ctx,
                execution=execution,
                operation=operation,
                scope=effective_scope,
                scope_token=scope_token,
                context_id=context_id,
                query=query,
                record=record,
                graph=graph,
                maintenance=maintenance,
                response=response,
            )
            # Correct misleading scope_source when scope came from active context fallback.
            # The service layer labels those fields as 'request' because the scope dict
            # was injected as a caller-provided scope — but the true source is the session's
            # active context, not an explicit caller argument.
            if _scope_from_active_context:
                payload = _relabel_scope_source_active_context(payload)
            return json_response(payload)
        except Exception as e:
            return json_response(_tool_error_payload("memory", e))

    if not enable_project_tools:
        return

    @mcp_server.tool(
        description=(
            "Start or continue project memory onboarding using resumable checkpoints. "
            "Call this first to run ingest/supersede/archive/maintain in order."
        ),
        annotations=ToolAnnotations(
            title="Onboard",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        ),
    )
    async def onboard(
        scope: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Project scope for onboarding (for example palace/wing/room/compartment)."
                ),
            ),
        ] = None,
        ingestion: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Ingestion configuration: mode ('programmatic'|'llm'), "
                    "llm_profile (required when mode='llm'), "
                    "reproducibility ('strict'|'relaxed', default 'strict')."
                ),
            ),
        ] = None,
        ingest: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Memory ingest payload (record section for memory(operation='ingest')). "
                    "Required when starting a new onboarding flow."
                ),
            ),
        ] = None,
        supersede: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Optional supersede payload (record section for memory(operation='supersede'))."
                ),
            ),
        ] = None,
        archive: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Optional archive payload (record section for memory(operation='archive'))."
                ),
            ),
        ] = None,
        maintain: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Optional maintain payload "
                    "(maintenance section for memory(operation='maintain'))."
                ),
            ),
        ] = None,
        checkpoint: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=("Checkpoint returned by onboard or sync to resume progress."),
            ),
        ] = None,
        scan: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Optional file-system scan config (ScanConfig). "
                    "When provided and ingest is omitted, an ingest payload is auto-generated "
                    "from scanned files. Explicit ingest keys win on conflict. "
                    "The scan snapshot is persisted in the returned checkpoint."
                ),
            ),
        ] = None,
        response: Annotated[
            dict[str, Any] | None,
            Field(default=None, description="Optional memory response shaping options."),
        ] = None,
        max_operations: Annotated[
            int,
            Field(
                default=1,
                ge=1,
                le=20,
                description=(
                    "Maximum onboarding steps to execute before returning. "
                    "Use 1 for strict checkpoint progression."
                ),
            ),
        ] = 1,
        debug: Annotated[
            bool,
            Field(
                default=False,
                description=(
                    "When True, include full checkpoint internals and per-step results "
                    "in the response. Default (False): compact mode strips completed blobs."
                ),
            ),
        ] = False,
        *,
        ctx: AppContextType,
    ) -> CallToolResult:
        """Run project onboarding steps and return a checkpoint for the next call."""
        app_ctx = ctx.request_context.lifespan_context
        unavailable = _memory_backend_unavailable_envelope(app_ctx)
        if unavailable is not None:
            return json_response(unavailable)

        # Merge root-level debug flag into response shaping dict.
        if debug and response is None:
            response = {"debug": True}
        elif debug and isinstance(response, dict) and not response.get("debug"):
            response = {**response, "debug": True}
        execution = _create_memory_execution(ctx)
        try:
            MemoryResponseInput.model_validate(response or {})
            # Resolve ingestion contract fields (spec §4.1).
            _ingestion = ingestion or {}
            _pipeline_mode: str | None = _ingestion.get("mode") if _ingestion else None

            # --- Phase 4: programmatic onboard fast-path ---
            # Activated only when ingestion.mode='programmatic': scan provided, no checkpoint,
            # no explicit ingest payload. Routes through the programmatic onboard
            # pipeline (graph build + validation), bypassing the checkpoint flow.
            if (
                _pipeline_mode == "programmatic"
                and scan is not None
                and checkpoint is None
                and ingest is None
            ):
                effective_scan_cfg = ScanConfig.model_validate(scan)
                scanned_files_raw, _snapshot = await _run_scan(
                    effective_scan_cfg,
                    active_project=_get_active_project(ctx),
                )
                file_entries = classify_scan_files_for_programmatic_mode(
                    scanned_files_raw,
                    base_path=effective_scan_cfg.root,
                )
                request = ProgrammaticOnboardRequest(
                    scope=scope or {},
                    files=file_entries,
                    mode="programmatic",
                    debug=debug,
                    provenance="onboard",
                    confidence=1.0,
                )
                if app_ctx.memory_backend is not None:
                    result = await run_programmatic_onboard_with_cycle_recording(
                        request,
                        memory_service=MemoryService(app_ctx.memory_backend, execution),
                    )
                else:
                    result = run_programmatic_onboard(request)
                if result.status == "completed":
                    persistence = await _persist_graph_payload_if_configured(
                        app_ctx=app_ctx,
                        execution=execution,
                        scope=result.scope,
                        graph=result.graph,
                        response=response,
                    )
                    candidate = _register_onboard_context_for_session(
                        ctx,
                        result.scope,
                        {"scope": result.scope, "scope_key": result.scope_key_value},
                    )
                    _set_active_context(ctx, candidate)
                    _enable_watcher_for_active_project_after_onboard(ctx)
                    payload = build_programmatic_onboard_response(result, debug=debug)
                    if persistence is not None:
                        payload["graph"]["persisted"] = {
                            "nodes": persistence["nodes"],
                            "corridors": persistence["corridors"],
                        }
                    return json_response(payload)
                return json_response(build_programmatic_onboard_response(result, debug=debug))

            # --- Phase 5: LLM onboard fast-path ---
            # Activated only when ingestion.mode='llm': scan provided, no checkpoint,
            # no explicit ingest payload. Routes through the LLM onboard pipeline
            # which validates the LLM profile and enforces reproducibility constraints.
            if (
                _pipeline_mode == "llm"
                and scan is not None
                and checkpoint is None
                and ingest is None
            ):
                llm_profile: str | None = _ingestion.get("llm_profile") or None
                if not llm_profile:
                    return json_response(
                        {
                            "error": {
                                "code": "INVALID_LLM_PROFILE",
                                "message": (
                                    "LLM mode requires 'llm_profile' in the ingestion config "
                                    "(e.g. ingestion={'mode': 'llm', 'llm_profile': 'standard'})."
                                ),
                                "retryable": False,
                                "stage": "profile_resolution",
                                "actionable_fix": ("Add 'llm_profile' key to the ingestion dict."),
                            }
                        }
                    )
                _reproducibility = _ingestion.get("reproducibility", "strict")
                llm_strict = _reproducibility != "relaxed"
                effective_scan_cfg = ScanConfig.model_validate(scan)
                scanned_files_raw, _snapshot = await _run_scan(
                    effective_scan_cfg,
                    active_project=_get_active_project(ctx),
                )
                file_entries = classify_scan_files_for_programmatic_mode(
                    scanned_files_raw,
                    base_path=effective_scan_cfg.root,
                )
                llm_request = LLMOnboardRequest(
                    scope=scope or {},
                    files=file_entries,
                    profile=llm_profile,
                    mode="llm",
                    strict=llm_strict,
                    debug=debug,
                    provenance="onboard",
                    confidence=1.0,
                )
                llm_result = run_llm_onboard(
                    llm_request,
                    loader=app_ctx.llm_config_loader,
                )
                if llm_result.status == "completed":
                    persistence = await _persist_graph_payload_if_configured(
                        app_ctx=app_ctx,
                        execution=execution,
                        scope=llm_result.scope,
                        graph=llm_result.graph,
                        response=response,
                    )
                    candidate = _register_onboard_context_for_session(
                        ctx,
                        llm_result.scope,
                        {"scope": llm_result.scope, "scope_key": llm_result.scope_key_value},
                    )
                    _set_active_context(ctx, candidate)
                    _enable_watcher_for_active_project_after_onboard(ctx)
                    payload = build_llm_onboard_response(llm_result, debug=debug)
                    if persistence is not None:
                        payload["graph"]["persisted"] = {
                            "nodes": persistence["nodes"],
                            "corridors": persistence["corridors"],
                        }
                    return json_response(payload)
                return json_response(build_llm_onboard_response(llm_result, debug=debug))

            # --- scan handling for new flows (checkpoint not provided) ---
            effective_scan: ScanConfig | None = None
            scan_snapshot: ScanSnapshot | None = None
            effective_ingest = ingest
            scan_ingested_paths: list[str] = []
            # Track whether auto-ingest was generated from scan (needs compartment injection).
            _scan_auto_ingest_active = False
            if scan is not None and checkpoint is None:
                effective_scan = ScanConfig.model_validate(scan)
                scanned_files, scan_snapshot = await _run_scan(
                    effective_scan,
                    active_project=_get_active_project(ctx),
                )
                auto_ingest, scan_ingested_paths = _build_ingest_from_files(scanned_files)
                if effective_ingest is None:
                    effective_ingest = auto_ingest
                    _scan_auto_ingest_active = True
                else:
                    # Explicit ingest wins on conflict; explicit caller controls path list
                    scan_ingested_paths = []
                    effective_ingest = _merge_dict_prefer_explicit(auto_ingest, effective_ingest)

            # Augment scope with a synthetic compartment for scan-driven auto-ingest.
            # Direct ingest (operation='ingest') requires scope.compartment — when the caller
            # provides only palace/wing/room (typical for a fresh onboard), the auto-generated
            # scan ingest payload would be rejected with COMPARTMENT_REQUIRED.  Injecting
            # compartment='scan' ensures the ingest step can proceed and records are queryable.
            # The caller's explicit compartment always wins; injection only applies when absent.
            # Build effective scope for the checkpoint plan, but only when we have an
            # actual scope or need to inject one for scan-driven auto-ingest.
            # Passing an empty dict (scope or {}) instead of None would falsely trigger
            # the MEM_CHECKPOINT_CONFLICT guard in normalize_project_checkpoint when a
            # checkpoint is being resumed without any explicit scope override.
            effective_scope_for_plan: dict[str, Any] | None = scope if scope else None
            if _scan_auto_ingest_active and not (effective_scope_for_plan or {}).get("wing"):
                # Inject wing='scan' to satisfy the scope hierarchy contract (compartment
                # requires room+wing; wing only requires palace).  This ensures scan-driven
                # auto-ingest records are queryable without violating hierarchy constraints.
                effective_scope_for_plan = {
                    **(effective_scope_for_plan or {}),
                    "wing": "scan",
                }

            resolved_scope, plan, next_index, completed = normalize_project_checkpoint(
                checkpoint=checkpoint,
                scope=effective_scope_for_plan,
                ingest=effective_ingest,
                supersede=supersede,
                archive=archive,
                maintain=maintain,
                require_ingest=True,
            )

            # Restore scan snapshot from existing checkpoint if present
            if scan_snapshot is None and checkpoint is not None:
                raw_scan = checkpoint.get("scan")
                if raw_scan is not None:
                    try:
                        effective_scan = ScanConfig.model_validate(raw_scan)
                    except Exception:
                        effective_scan = None
                raw_snap = checkpoint.get("scan_snapshot")
                if raw_snap is not None:
                    try:
                        scan_snapshot = ScanSnapshot.model_validate(raw_snap)
                    except Exception:
                        scan_snapshot = None

            state = FlowState(
                scope=resolved_scope,
                plan=plan,
                next_index=next_index,
                completed=completed,
                scan=effective_scan.model_dump() if effective_scan is not None else None,
                scan_snapshot=scan_snapshot.model_dump() if scan_snapshot is not None else None,
                on_ingest=_make_scan_snapshot_backfill(scan_snapshot, scan_ingested_paths),
            )
            advance_result = await _PROJECT_FLOW_SERVICE.advance(
                state,
                executor=_build_step_executor(
                    app_ctx=app_ctx, execution=execution, response=response
                ),
                error_shaper=lambda exc, stage: _tool_error_payload("onboard", exc, stage=stage),
                flow_name="onboard",
                max_operations=max_operations,
                debug=_is_debug_response(response),
            )
            if advance_result.completed:
                candidate = _register_onboard_context_for_session(
                    ctx,
                    resolved_scope,
                    advance_result.checkpoint,
                )
                _set_active_context(ctx, candidate)
                _enable_watcher_for_active_project_after_onboard(ctx)
            return json_response(advance_result.response)
        except Exception as e:
            return json_response(_tool_error_payload("onboard", e))

    @mcp_server.tool(
        description=(
            "Continue project memory synchronization from a checkpoint, or start a new sync plan. "
            "Call this after onboard returns status='checkpoint'."
        ),
        annotations=ToolAnnotations(
            title="Sync",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        ),
    )
    async def sync(
        checkpoint: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=("Checkpoint returned by onboard or sync."),
            ),
        ] = None,
        scope: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description="Scope for starting a new sync flow when checkpoint is omitted.",
            ),
        ] = None,
        ingestion: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Ingestion configuration: mode ('programmatic'|'llm'), "
                    "llm_profile (required when mode='llm'), "
                    "reproducibility ('strict'|'relaxed', default 'strict')."
                ),
            ),
        ] = None,
        ingest: Annotated[
            dict[str, Any] | None,
            Field(default=None, description="Optional ingest payload for new sync flows."),
        ] = None,
        supersede: Annotated[
            dict[str, Any] | None,
            Field(default=None, description="Optional supersede payload for new sync flows."),
        ] = None,
        archive: Annotated[
            dict[str, Any] | None,
            Field(default=None, description="Optional archive payload for new sync flows."),
        ] = None,
        maintain: Annotated[
            dict[str, Any] | None,
            Field(default=None, description="Optional maintain payload for new sync flows."),
        ] = None,
        scan: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Optional scan config override. When the resolved checkpoint carries a "
                    "scan_snapshot, re-scan is performed automatically using the stored config "
                    "(or this override). Delta (added/modified/deleted) drives auto-generated "
                    "ingest and deletion payloads. Explicit caller payloads win on conflict."
                ),
            ),
        ] = None,
        response: Annotated[
            dict[str, Any] | None,
            Field(default=None, description="Optional memory response shaping options."),
        ] = None,
        max_operations: Annotated[
            int,
            Field(
                default=1,
                ge=1,
                le=20,
                description="Maximum sync steps to execute before returning.",
            ),
        ] = 1,
        debug: Annotated[
            bool,
            Field(
                default=False,
                description=(
                    "When True, include full checkpoint internals and per-step results "
                    "in the response. Default (False): compact mode strips completed blobs."
                ),
            ),
        ] = False,
        *,
        ctx: AppContextType,
    ) -> CallToolResult:
        """Advance sync steps and return the next checkpoint or final result."""
        app_ctx = ctx.request_context.lifespan_context
        unavailable = _memory_backend_unavailable_envelope(app_ctx)
        if unavailable is not None:
            return json_response(unavailable)

        # Merge root-level debug flag into response shaping dict.
        if debug and response is None:
            response = {"debug": True}
        elif debug and isinstance(response, dict) and not response.get("debug"):
            response = {**response, "debug": True}
        execution = _create_memory_execution(ctx)
        try:
            MemoryResponseInput.model_validate(response or {})
            # Fast-path: if the checkpoint is already completed (next_index == len(plan)),
            # validate it first and then return a stable completed response without
            # replaying any operations.  Validation runs before the early return so that
            # corrupt/inconsistent completed-checkpoints still surface MEM_CHECKPOINT_INVALID.
            if checkpoint is not None:
                _raw_next = checkpoint.get("next_index", 0)
                _raw_plan = checkpoint.get("plan", [])
                # B: Accept integral floats (e.g. 1.0) by coercing safely; reject non-integral.
                if isinstance(_raw_next, float):
                    if _raw_next != int(_raw_next):
                        raise checkpoint_error(
                            "checkpoint.next_index must be an integer; "
                            f"non-integral float {_raw_next!r} is not allowed"
                        )
                    _raw_next = int(_raw_next)
                if (
                    isinstance(_raw_next, int)
                    and isinstance(_raw_plan, list)
                    and _raw_next > 0
                    and _raw_next == len(_raw_plan)
                ):
                    # C: Log to prove fast-path was taken (visible in server logs).
                    logger.info(
                        "sync fast-path: checkpoint already completed "
                        "(next_index=%d == len(plan)=%d); skipping replay",
                        _raw_next,
                        len(_raw_plan),
                    )
                    # Run full validation (raises MemoryContractError on corrupt checkpoint).
                    _scope, _plan, _ni, _done = restore_checkpoint(checkpoint, require_ingest=False)
                    # A: Include from_checkpoint flag and note in response.
                    _fp_debug = _is_debug_response(response)
                    _fp_checkpoint = (
                        build_project_checkpoint_payload(
                            scope=_scope,
                            plan=_plan,
                            next_index=_ni,
                            completed=_done,
                            scan=None,
                            scan_snapshot=None,
                        )
                        if checkpoint.get("scan") is None
                        else checkpoint
                    )
                    _fp_response: dict[str, Any] = {
                        "status": "completed",
                        "from_checkpoint": True,
                        "note": (
                            "Results returned from completed checkpoint cache; "
                            "no operations were re-executed."
                        ),
                        "completed_operations": [
                            str(item.get("operation")) for item in _done if isinstance(item, dict)
                        ],
                        "checkpoint": compact_checkpoint(_fp_checkpoint, debug=_fp_debug),
                    }
                    if _fp_debug:
                        _fp_response["results"] = _done
                    return json_response(_fp_response)

            effective_scan: ScanConfig | None = None
            prior_snapshot: ScanSnapshot | None = None
            new_scan_snapshot: ScanSnapshot | None = None
            sync_ingested_paths: list[str] = []
            sync_delta: SyncDelta | None = None

            if checkpoint is not None:
                raw_scan = checkpoint.get("scan")
                if raw_scan is not None:
                    effective_scan = ScanConfig.model_validate(raw_scan)

                raw_snap = checkpoint.get("scan_snapshot")
                if raw_snap is not None:
                    prior_snapshot = ScanSnapshot.model_validate(raw_snap)
                    if effective_scan is None:
                        effective_scan = prior_snapshot.scan_config

            if scan is not None:
                effective_scan = ScanConfig.model_validate(scan)

            if checkpoint is not None and effective_scan is None:
                resolved_scope, plan, next_index, completed = normalize_project_checkpoint(
                    checkpoint=checkpoint,
                    scope=scope,
                    ingest=ingest,
                    supersede=supersede,
                    archive=archive,
                    maintain=maintain,
                    require_ingest=False,
                )
            elif checkpoint is not None:
                if scope is not None:
                    raise MemoryContractError(
                        code="MEM_CHECKPOINT_CONFLICT",
                        message=(
                            "MEM_CHECKPOINT_CONFLICT: provide either checkpoint OR "
                            "scope/plan payloads, not both"
                        ),
                        retryable=False,
                    )

                resolved_scope, plan, next_index, completed = normalize_project_checkpoint(
                    checkpoint=checkpoint,
                    scope=None,
                    ingest=None,
                    supersede=None,
                    archive=None,
                    maintain=None,
                    require_ingest=False,
                )

                if effective_scan is None:
                    raise MemoryContractError(
                        code="MEM_SCAN_CONFIG_MISSING",
                        message=(
                            "MEM_SCAN_CONFIG_MISSING: checkpoint carries a scan_snapshot "
                            "but no resolvable scan config; provide scan override to proceed"
                        ),
                        retryable=False,
                    )
                scanned_files, new_scan_snapshot = await _run_scan(
                    effective_scan,
                    active_project=_get_active_project(ctx),
                )

                changed_files = scanned_files
                deleted: list[str] = []
                if prior_snapshot is not None:
                    sync_delta = _compute_full_scan_delta(prior_snapshot, new_scan_snapshot)
                    # UNCHANGED fast-path only applies when this is a fresh sync start
                    # (next_index == 0), meaning no operations have been executed yet from
                    # this checkpoint.  Mid-flight resumes (next_index > 0) must continue
                    # executing the remaining steps even when the file delta is empty —
                    # those steps were already committed to when the prior run paused.
                    _is_fresh_start = next_index == 0
                    if not sync_delta.has_semantic_delta and _is_fresh_start:
                        # Idempotency: no files changed — skip plan execution entirely.
                        _sync_debug = _is_debug_response(response)
                        unchanged_response: dict[str, Any] = {
                            "status": "UNCHANGED",
                            "scope": resolved_scope,
                            "message": "No files changed since last sync; nothing to do.",
                        }
                        if _sync_debug:
                            unchanged_response["delta"] = sync_delta.to_debug_dict()
                        return json_response(unchanged_response)
                    added = sync_delta.added
                    modified = sync_delta.modified
                    deleted = sync_delta.deleted
                    changed_paths = set(added) | set(modified)
                    changed_files = [f for f in scanned_files if f["path"] in changed_paths]

                auto_ingest_payload: dict[str, Any] | None = None
                if len(changed_files) > 0:
                    auto_ingest_payload, sync_ingested_paths = _build_ingest_from_files(
                        changed_files
                    )
                auto_ingest = auto_ingest_payload
                effective_ingest = (
                    _merge_dict_prefer_explicit(auto_ingest, ingest)
                    if auto_ingest is not None
                    else ingest
                )
                if ingest is not None:
                    # Explicit ingest overrides path tracking
                    sync_ingested_paths = []

                deleted_ids: list[str] = []
                if prior_snapshot is not None and len(deleted) > 0:
                    deleted_set = set(deleted)
                    deleted_ids = [
                        entry.memory_id
                        for entry in prior_snapshot.entries
                        if entry.path in deleted_set and entry.memory_id is not None
                    ]

                auto_archive: dict[str, Any] | None = None
                auto_supersede: dict[str, Any] | None = None
                if len(deleted_ids) > 0:
                    canonical_policy = classify_deletion_policy(effective_scan.deletion_policy)
                    if canonical_policy == "archive":
                        auto_archive = {"ids": deleted_ids}
                    elif canonical_policy == "supersede":
                        auto_supersede = {"ids": deleted_ids}
                    # canonical_policy == "ignore": no action on deleted files

                effective_archive = (
                    _merge_dict_prefer_explicit(auto_archive, archive)
                    if auto_archive is not None
                    else archive
                )
                effective_supersede = (
                    _merge_dict_prefer_explicit(auto_supersede, supersede)
                    if auto_supersede is not None
                    else supersede
                )

                rebuilt_plan = project_flow_plan(
                    ingest=effective_ingest,
                    supersede=effective_supersede,
                    archive=effective_archive,
                    maintain=maintain,
                )
                if len(rebuilt_plan) > 0:
                    plan = rebuilt_plan
                    next_index = 0
                    completed = []
            else:
                # sync({}) detection: no checkpoint, no scan, no plan payloads.
                # Resolution priority:
                # 1. Session active context (set by onboard/select)
                # 2. Explicit scope filter against onboard registry
                # 3. Registry-wide (ambiguous or no-context)
                # Return actionable error instead of MEM_PROJECT_FLOW_EMPTY.
                _no_plan = (
                    ingest is None and supersede is None and archive is None and maintain is None
                )
                _registry_resolved = False
                if effective_scan is None and _no_plan:
                    # Priority 1: use session active context when no explicit scope filter.
                    _active = _get_active_context(ctx) if scope is None else None
                    if _active is not None:
                        # Active context found — use it directly.
                        candidates: list[SyncContextCandidate] = [_active]
                        resolution = resolve_sync_context(candidates, requested_scope=None)
                    else:
                        # Fall back to this session's onboarded candidate registry.
                        candidates = _list_session_onboard_candidates(ctx)
                        resolution = resolve_sync_context(candidates, requested_scope=scope)

                    if resolution.status == "AMBIGUOUS_CONTEXT":
                        return json_response(
                            build_ambiguous_context_envelope(resolution.candidates)
                        )
                    if resolution.status == "success" and resolution.context is not None:
                        # Resolved: use the stored checkpoint to bootstrap the sync flow.
                        stored_cp = resolution.context.checkpoint_data
                        resolved_scope = resolution.context.scope
                        plan = []
                        next_index = 0
                        completed = []
                        # If the stored checkpoint carries a scan config, re-run the scan.
                        raw_scan_cfg = stored_cp.get("scan")
                        raw_snap = stored_cp.get("scan_snapshot")
                        if raw_scan_cfg is not None:
                            effective_scan = ScanConfig.model_validate(raw_scan_cfg)
                        if raw_snap is not None:
                            try:
                                prior_snapshot = ScanSnapshot.model_validate(raw_snap)
                                if effective_scan is None:
                                    effective_scan = prior_snapshot.scan_config
                            except Exception:
                                pass
                        if effective_scan is None:
                            # No scan config in stored checkpoint: return UNCHANGED so
                            # callers know onboard completed but nothing to re-sync.
                            return json_response(
                                {
                                    "status": "UNCHANGED",
                                    "scope": resolved_scope,
                                    "message": (
                                        "Onboard context found but no scan config is stored. "
                                        "Provide a scan argument to perform a delta sync."
                                    ),
                                }
                            )
                        _registry_resolved = True
                        # Fall through to scan + delta handling below.
                    else:
                        return json_response(_build_no_active_context_envelope())

                effective_ingest = ingest
                if effective_scan is not None:
                    scanned_files, new_scan_snapshot = await _run_scan(
                        effective_scan,
                        active_project=_get_active_project(ctx),
                    )
                    auto_ingest, sync_ingested_paths = _build_ingest_from_files(scanned_files)
                    effective_ingest = _merge_dict_prefer_explicit(auto_ingest, ingest)
                    if ingest is not None:
                        sync_ingested_paths = []

                # When resolved_scope/plan/next_index/completed were set by the registry
                # path above, skip re-building (they are already final).
                if not _registry_resolved:
                    resolved_scope, plan, next_index, completed = normalize_project_checkpoint(
                        checkpoint=None,
                        scope=scope,
                        ingest=effective_ingest,
                        supersede=supersede,
                        archive=archive,
                        maintain=maintain,
                        require_ingest=False,
                    )

            _sync_debug = _is_debug_response(response)
            state = FlowState(
                scope=resolved_scope,
                plan=plan,
                next_index=next_index,
                completed=completed,
                scan=effective_scan.model_dump() if effective_scan is not None else None,
                scan_snapshot=(
                    new_scan_snapshot.model_dump() if new_scan_snapshot is not None else None
                ),
                on_ingest=_make_scan_snapshot_backfill(new_scan_snapshot, sync_ingested_paths),
            )
            result = await _PROJECT_FLOW_SERVICE.advance(
                state,
                executor=_build_step_executor(
                    app_ctx=app_ctx, execution=execution, response=response
                ),
                error_shaper=lambda exc, stage: _tool_error_payload("sync", exc, stage=stage),
                flow_name="sync",
                max_operations=max_operations,
                debug=_sync_debug,
            )
            if _sync_debug and result.completed and sync_delta is not None:
                result.response["delta"] = sync_delta.to_debug_dict()
            return json_response(result.response)
        except Exception as e:
            return json_response(_tool_error_payload("sync", e))

    # -----------------------------------------------------------------------
    # Wire strict HTTP adapters (Task 6)
    #
    # These closures capture the inner onboard/sync tool functions defined
    # above.  They validate the raw HTTP payload through strict Pydantic models
    # before delegating to the existing orchestration, then JSON-decode the
    # CallToolResult text into a plain dict for the HTTP layer.
    # -----------------------------------------------------------------------

    async def _onboard_http_impl(payload: dict[str, Any], *, ctx: AppContextType) -> dict[str, Any]:
        """Validate payload strictly then delegate to onboard() orchestration."""
        request = OnboardRequest.model_validate(payload)  # raises ValidationError on violation
        result = await onboard(
            scope=request.scope.model_dump(exclude_none=True) if request.scope else None,
            ingestion=request.ingestion.model_dump(exclude_none=True)
            if request.ingestion
            else None,
            ingest=request.ingest,
            supersede=request.supersede,
            archive=request.archive,
            maintain=request.maintain,
            checkpoint=request.checkpoint.model_dump(exclude_none=True)
            if request.checkpoint
            else None,
            scan=request.scan,
            response=request.response.model_dump(exclude_none=True) if request.response else None,
            max_operations=request.max_operations,
            debug=request.debug,
            ctx=ctx,
        )
        decoded = json.loads(result.content[0].text)
        if not isinstance(decoded, dict):
            raise RuntimeError("onboard() must return a JSON object payload")
        return decoded

    async def _sync_http_impl(payload: dict[str, Any], *, ctx: AppContextType) -> dict[str, Any]:
        """Validate payload strictly then delegate to sync() orchestration."""
        request = SyncRequest.model_validate(payload)  # raises ValidationError on violation
        result = await sync(
            scope=request.scope.model_dump(exclude_none=True) if request.scope else None,
            ingest=request.ingest,
            supersede=request.supersede,
            archive=request.archive,
            maintain=request.maintain,
            checkpoint=request.checkpoint.model_dump(exclude_none=True)
            if request.checkpoint
            else None,
            scan=request.scan,
            response=request.response.model_dump(exclude_none=True) if request.response else None,
            max_operations=request.max_operations,
            debug=request.debug,
            ctx=ctx,
        )
        decoded = json.loads(result.content[0].text)
        if not isinstance(decoded, dict):
            raise RuntimeError("sync() must return a JSON object payload")
        return decoded

    # Publish to module-level names so the HTTP transport layer (Task 7)
    # and tests can import them directly from workflows_mcp.tools_memory.
    global onboard_http, sync_http
    onboard_http = _onboard_http_impl
    sync_http = _sync_http_impl

    @mcp_server.tool(
        description=(
            "Switch active memory context for this session. "
            "After selecting, memory() and sync() calls without explicit scope "
            "use the selected context automatically. "
            "Use onboard() first to create contexts."
        ),
        annotations=ToolAnnotations(
            title="Select",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def select(
        project: Annotated[
            str | None,
            Field(
                default=None,
                description=(
                    "Project selector (project_id, slug, or palace). "
                    "When provided, selects from session/token allowed projects."
                ),
            ),
        ] = None,
        scope: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Scope selector: at minimum provide 'palace'. "
                    "Optionally include 'wing', 'room', 'compartment' to narrow selection. "
                    "Example: {'palace': 'my-project'} or "
                    "{'palace': 'my-project', 'wing': 'backend'}."
                ),
            ),
        ] = None,
        *,
        ctx: AppContextType,
    ) -> CallToolResult:
        """Switch the session's active memory context to the specified scope."""
        app_ctx = ctx.request_context.lifespan_context
        unavailable = _memory_backend_unavailable_envelope(app_ctx)
        if unavailable is not None:
            return json_response(unavailable)

        if project is not None:
            session = _get_session(ctx)
            _prime_session_project_context_from_auth(ctx)
            allowed = app_ctx.list_allowed_projects(session)
            if not allowed:
                return json_response(
                    {
                        "error": {
                            "code": "MEM_NO_ACTIVE_CONTEXT",
                            "message": (
                                "No allowed projects found for this session. "
                                "Bind projects first, then call select(project=...)."
                            ),
                            "retryable": False,
                            "actionable_fix": (
                                "Register allowed projects for this session/token context, "
                                "then select by project id, slug, or palace."
                            ),
                        }
                    }
                )

            matches = [p for p in allowed if project in {p.project_id, p.slug, p.palace}]
            if len(matches) > 1:
                return json_response(
                    {
                        "error": {
                            "code": "MEM_SELECT_AMBIGUOUS_PROJECT",
                            "message": (
                                f"Project selector {project!r} matched multiple allowed projects."
                            ),
                            "retryable": False,
                            "actionable_fix": (
                                "Select by unique project_id, or narrow project bindings to avoid "
                                "duplicate slug/palace values in one session."
                            ),
                        }
                    }
                )
            if not matches:
                return json_response(
                    {
                        "error": {
                            "code": "MEM_SELECT_NOT_FOUND",
                            "message": f"No allowed project matched selector {project!r}.",
                            "retryable": False,
                            "actionable_fix": (
                                "Use a known project_id, slug, or palace from session bindings."
                            ),
                        }
                    }
                )

            selected = matches[0]
            app_ctx.set_active_project(session, selected)
            return json_response(
                {
                    "status": "selected",
                    "active_project": {
                        "project_id": selected.project_id,
                        "slug": selected.slug,
                        "palace": selected.palace,
                        "default_wing": selected.default_wing,
                        "default_room": selected.default_room,
                        "source": selected.source,
                    },
                    "message": (
                        "Active project set for this session. Subsequent memory() calls may "
                        "use project defaults when scope fields are omitted."
                    ),
                }
            )

        if scope is None:
            return json_response(
                {
                    "error": {
                        "code": "MEM_SELECT_INVALID_REQUEST",
                        "message": "select requires either 'project' or 'scope'.",
                        "retryable": False,
                        "actionable_fix": "Pass project='...' or scope={...}.",
                    }
                }
            )

        # Resolve against this session's onboard registry.
        candidates: list[SyncContextCandidate] = _list_session_onboard_candidates(ctx)
        if not candidates:
            return json_response(
                {
                    "error": {
                        "code": "MEM_NO_ACTIVE_CONTEXT",
                        "message": (
                            "No onboarded contexts found in this server session. "
                            "Run onboard() first to create a context before using select()."
                        ),
                        "retryable": False,
                        "actionable_fix": "Run onboard(scope={...}, ingest={...}) first.",
                    }
                }
            )

        resolution = resolve_sync_context(candidates, requested_scope=scope)

        if resolution.status == "AMBIGUOUS_CONTEXT":
            return json_response(build_ambiguous_context_envelope(resolution.candidates))

        if resolution.status == "success" and resolution.context is not None:
            _set_active_context(ctx, resolution.context)
            active = resolution.context
            return json_response(
                {
                    "status": "selected",
                    "active_context": {
                        "scope": active.scope,
                        "scope_key": active.scope_key_value,
                        "source": active.source,
                    },
                    "message": (
                        "Active context set. Subsequent memory() and sync() calls "
                        "without explicit scope will use this context."
                    ),
                }
            )

        # NO_CONTEXT — scope not found in registry.
        candidate_scopes = [c.scope for c in candidates]
        return json_response(
            {
                "error": {
                    "code": "MEM_SELECT_NOT_FOUND",
                    "message": (
                        f"No onboarded context matched scope {scope!r}. "
                        f"Known contexts: {candidate_scopes}"
                    ),
                    "retryable": False,
                    "actionable_fix": (
                        "Run onboard() with the desired scope first, "
                        "or adjust the scope selector to match a known context."
                    ),
                }
            }
        )

    @mcp_server.tool(
        description=(
            "Clear all ADR-013 ontology rows for an explicit palace scope and prepare "
            "for a fresh re-onboard. Requires an explicit 'palace' argument. "
            "Validates the target palace exists before any delete. "
            "All deletes execute in a single atomic transaction. "
            "No migration or compatibility layer is provided — this is a clean slate only."
        ),
        annotations=ToolAnnotations(
            title="Fresh Start",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=False,
        ),
    )
    async def fresh_start(
        palace: Annotated[
            str | None,
            Field(
                default=None,
                description=(
                    "Required. The palace name whose ADR-013 ontology rows will be cleared. "
                    "No global wipe path exists — this argument must be explicitly provided."
                ),
            ),
        ] = None,
        scope: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Optional sub-scope within the palace "
                    "(wing/room/compartment). Reserved for future narrowed clears; "
                    "currently only palace-level clearing is supported."
                ),
            ),
        ] = None,
        *,
        ctx: AppContextType,
    ) -> CallToolResult:
        """Clear all ADR-013 ontology rows for the given palace, atomically."""
        # --- guard: palace is required ---
        if not palace:
            return json_response(
                {
                    "error": {
                        "code": "MEM_FRESH_START_MISSING_SCOPE",
                        "message": (
                            "MEM_FRESH_START_MISSING_SCOPE: 'palace' is required and must be "
                            "a non-empty string. No global wipe path exists."
                        ),
                        "retryable": False,
                        "actionable_fix": (
                            "Provide an explicit palace name: "
                            "fresh_start(palace='<your-palace-name>')."
                        ),
                    }
                }
            )

        app_ctx = ctx.request_context.lifespan_context
        unavailable = _memory_backend_unavailable_envelope(app_ctx)
        if unavailable is not None:
            return json_response(unavailable)

        scope_identity: dict[str, Any] = {"palace": palace}
        if scope:
            scope_identity.update({k: v for k, v in scope.items() if v is not None})

        logger.info(
            "fresh_start: requested scope palace=%r scope_detail=%r",
            palace,
            scope,
        )

        backend: Any = getattr(app_ctx, "memory_backend", None)
        uses_ephemeral = backend is None
        if uses_ephemeral:
            config = (
                memory_connection_config_from_metadata(app_ctx)
                or memory_connection_config_from_env()
            )
            if config is None:
                return json_response(
                    {
                        "error": {
                            "code": "MEMORY_BACKEND_UNAVAILABLE",
                            "message": (
                                "MEMORY_BACKEND_UNAVAILABLE: no memory PostgreSQL backend "
                                "is configured. Save database settings or set MEMORY_DB_HOST."
                            ),
                            "retryable": False,
                        }
                    }
                )
            backend = PostgresBackend()
            try:
                await backend.connect(config)
                from .memory.knowledge.schema import ensure_schema

                await ensure_schema(backend)
            except Exception as conn_err:
                return json_response(_tool_error_payload("fresh_start", conn_err))

        try:
            # --- scope existence validation: reject unknown palace before any delete ---
            existence_rows = await backend.query(
                "SELECT COUNT(*) AS count FROM knowledge_verification_cycles WHERE palace = $1"
                " UNION ALL "
                "SELECT COUNT(*) FROM knowledge_structural_evidence WHERE palace = $1"
                " UNION ALL "
                "SELECT COUNT(*) FROM knowledge_semantic_claims WHERE palace = $1",
                (palace, palace, palace),
            )
            total_existing = sum(int(row.get("count", 0)) for row in existence_rows.rows)
            if total_existing == 0:
                logger.warning(
                    "fresh_start: palace=%r not found in any ontology table; rejecting",
                    palace,
                )
                return json_response(
                    {
                        "error": {
                            "code": "MEM_FRESH_START_SCOPE_NOT_FOUND",
                            "message": (
                                f"MEM_FRESH_START_SCOPE_NOT_FOUND: palace {palace!r} has no "
                                "rows in any ADR-013 ontology table. "
                                "Run onboard first, or check the palace name."
                            ),
                            "retryable": False,
                            "actionable_fix": (
                                "Verify the palace name matches an onboarded palace, "
                                "or run onboard() to initialise it first."
                            ),
                        }
                    }
                )

            # --- FK-safe deletion order (respects ON DELETE RESTRICT constraints) ---
            # 1. knowledge_wing_proof_bundles  → references semantic_claims RESTRICT
            # 2. knowledge_semantic_overrides   → references semantic_claims RESTRICT
            # 3. knowledge_claim_evidence_links → CASCADE from semantic_claims (delete explicitly)
            # 4. knowledge_semantic_claims
            # 5. knowledge_structural_evidence
            # 6. knowledge_verification_cycles
            #
            # All six deletes run inside a single explicit transaction.
            # On any failure, rollback() is called before returning the error envelope
            # so no partial-delete state can persist.

            per_table: dict[str, int] = {}

            await backend.begin_transaction()
            try:
                # knowledge_wing_proof_bundles
                r = await backend.execute(
                    "DELETE FROM knowledge_wing_proof_bundles WHERE palace = $1",
                    (palace,),
                )
                per_table["knowledge_wing_proof_bundles"] = getattr(r, "rowcount", 0) or 0
                logger.info(
                    "fresh_start: deleted %d rows from knowledge_wing_proof_bundles palace=%r",
                    per_table["knowledge_wing_proof_bundles"],
                    palace,
                )

                # knowledge_semantic_overrides: palace column may not exist directly;
                # join via knowledge_semantic_claims.
                r = await backend.execute(
                    "DELETE FROM knowledge_semantic_overrides"
                    " WHERE claim_id IN ("
                    "   SELECT id FROM knowledge_semantic_claims WHERE palace = $1"
                    " )",
                    (palace,),
                )
                per_table["knowledge_semantic_overrides"] = getattr(r, "rowcount", 0) or 0
                logger.info(
                    "fresh_start: deleted %d rows from knowledge_semantic_overrides palace=%r",
                    per_table["knowledge_semantic_overrides"],
                    palace,
                )

                # knowledge_claim_evidence_links: join via semantic_claims palace
                r = await backend.execute(
                    "DELETE FROM knowledge_claim_evidence_links"
                    " WHERE claim_id IN ("
                    "   SELECT id FROM knowledge_semantic_claims WHERE palace = $1"
                    " )",
                    (palace,),
                )
                per_table["knowledge_claim_evidence_links"] = getattr(r, "rowcount", 0) or 0
                logger.info(
                    "fresh_start: deleted %d rows from knowledge_claim_evidence_links palace=%r",
                    per_table["knowledge_claim_evidence_links"],
                    palace,
                )

                # knowledge_semantic_claims
                r = await backend.execute(
                    "DELETE FROM knowledge_semantic_claims WHERE palace = $1",
                    (palace,),
                )
                per_table["knowledge_semantic_claims"] = getattr(r, "rowcount", 0) or 0
                logger.info(
                    "fresh_start: deleted %d rows from knowledge_semantic_claims palace=%r",
                    per_table["knowledge_semantic_claims"],
                    palace,
                )

                # knowledge_structural_evidence
                r = await backend.execute(
                    "DELETE FROM knowledge_structural_evidence WHERE palace = $1",
                    (palace,),
                )
                per_table["knowledge_structural_evidence"] = getattr(r, "rowcount", 0) or 0
                logger.info(
                    "fresh_start: deleted %d rows from knowledge_structural_evidence palace=%r",
                    per_table["knowledge_structural_evidence"],
                    palace,
                )

                # knowledge_verification_cycles (last — no remaining dependents)
                r = await backend.execute(
                    "DELETE FROM knowledge_verification_cycles WHERE palace = $1",
                    (palace,),
                )
                per_table["knowledge_verification_cycles"] = getattr(r, "rowcount", 0) or 0
                logger.info(
                    "fresh_start: deleted %d rows from knowledge_verification_cycles palace=%r",
                    per_table["knowledge_verification_cycles"],
                    palace,
                )

                await backend.commit()

            except Exception as del_err:
                await backend.rollback()
                logger.error(
                    "fresh_start: delete failed for palace=%r error=%s; transaction rolled back",
                    palace,
                    del_err,
                    exc_info=True,
                )
                return json_response(_tool_error_payload("fresh_start", del_err))

            total_deleted = sum(per_table.values())
            logger.info(
                "fresh_start: completed palace=%r total_deleted=%d scope_identity=%r",
                palace,
                total_deleted,
                scope_identity,
            )

            return json_response(
                {
                    "status": "completed",
                    "scope_identity": scope_identity,
                    "total_deleted": total_deleted,
                    "deleted_rows": per_table,
                }
            )
        finally:
            if uses_ephemeral:
                await backend.disconnect()


# ---------------------------------------------------------------------------
# Strict HTTP-facing adapters (Task 6)
#
# onboard_http and sync_http are async callables that:
#   1. Parse the raw dict payload through the strict Pydantic contract model
#      (raises pydantic.ValidationError on unknown fields).
#   2. Delegate to the underlying onboard() / sync() MCP tool orchestration.
#   3. Return the JSON-decoded result dict.
#
# These names are assigned to module-level variables inside register_memory_tools
# (after the inner onboard/sync closures are in scope) so they are importable
# by the HTTP transport layer (Task 7) and by tests.
# ---------------------------------------------------------------------------

# Sentinel callables replaced by register_memory_tools at import time.


async def _onboard_http_not_ready(
    payload: dict[str, Any], *, ctx: AppContextType
) -> dict[str, Any]:
    raise RuntimeError("onboard_http is not available: call register_memory_tools() first.")


async def _sync_http_not_ready(payload: dict[str, Any], *, ctx: AppContextType) -> dict[str, Any]:
    raise RuntimeError("sync_http is not available: call register_memory_tools() first.")


onboard_http = _onboard_http_not_ready
sync_http = _sync_http_not_ready
