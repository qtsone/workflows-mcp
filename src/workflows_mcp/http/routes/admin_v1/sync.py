from __future__ import annotations

import logging
from importlib.resources import files
from pathlib import Path as FilePath
from sqlite3 import Connection
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, ValidationError

from workflows_mcp.engine.memory_service import MemoryContractError
from workflows_mcp.http.dependencies import (
    CurrentAdminSession,
    get_resources,
    require_admin_csrf,
    require_current_admin_session,
)
from workflows_mcp.http.lifespan import AppResources
from workflows_mcp.memory_runtime import refresh_memory_backend, register_memory_executors
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.repos.projects_repo import ProjectRecord, SQLiteProjectsRepository
from workflows_mcp.metadata.repos.watcher_repo import (
    DirtyQueueEntry,
    SQLiteWatcherRepository,
    UnknownProjectError,
)

router = APIRouter(prefix="/sync")
logger = logging.getLogger(__name__)

ProjectId = Annotated[str, Path(min_length=1, max_length=128)]


def _normalized_optional(value: str | None) -> str | None:
    normalized = (value or "").strip()
    return normalized or None


_TOPOLOGY_PLACEHOLDER_VALUES: frozenset[str] = frozenset(
    {"default-wing", "default-room", "default", "code"}
)


def _normalized_topology_optional(value: str | None) -> str:
    normalized = _normalized_optional(value)
    if normalized is None:
        return ""
    if normalized.lower() in _TOPOLOGY_PLACEHOLDER_VALUES:
        return ""
    return normalized


class SyncProjectSummary(BaseModel):
    project_id: str
    dirty_count: int
    requires_reconciliation: bool


class SyncListResponse(BaseModel):
    projects: list[SyncProjectSummary]


class SyncErrorDetail(BaseModel):
    code: str
    message: str


class SyncLogEntry(BaseModel):
    id: int
    project_id: str
    path: str
    event_type: str
    reason: str
    status: str
    enqueued_at: str
    updated_at: str
    processed_at: str | None


class SyncLogsResponse(BaseModel):
    project_id: str
    entries: list[SyncLogEntry]


class SyncExtractionCounts(BaseModel):
    source_items: int
    structural_evidence: int
    verification_cycles: int
    wings: int
    rooms: int
    compartments: int
    semantic_claims: int
    semantic_memories: int


class SyncProjectDetailsResponse(BaseModel):
    project_id: str
    memory_mode: str
    system1_enabled: bool
    system1_state: str
    system2_enabled: bool
    system2_state: str
    embedding_profile_required: bool
    embedding_profile: str
    memory_backend_ready: bool
    counts: SyncExtractionCounts


class SyncNowResponse(BaseModel):
    project_id: str
    status: str
    dirty_count: int
    action: str
    memory_mode: str
    job_id: str | None = None
    workflow: str | None = None
    error: SyncErrorDetail | None = None


def _repo(resources: AppResources) -> tuple[SQLiteWatcherRepository, Connection]:
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    return SQLiteWatcherRepository(conn), conn


def _assert_project_exists(repo: SQLiteWatcherRepository, project_id: str) -> None:
    if not repo.project_exists(project_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "project_not_found", "message": f"Project not found: {project_id}"},
        )


def _scope_for_project(project: ProjectRecord) -> dict[str, str]:
    return {
        "palace": project.palace,
        "wing": _normalized_topology_optional(project.default_wing),
        "room": _normalized_topology_optional(project.default_room),
        "compartment": project.slug,
    }


def _sync_error_detail(exc: Exception) -> SyncErrorDetail:
    if isinstance(exc, MemoryContractError):
        return SyncErrorDetail(code="project_system1_sync_failed", message=exc.message)
    if isinstance(exc, ValidationError):
        contract_message = _contract_message_from_validation_error(exc)
        if contract_message:
            return SyncErrorDetail(code="project_system1_sync_failed", message=contract_message)
    return SyncErrorDetail(
        code="project_system1_sync_failed",
        message="Project System 1 sync failed; check memory database readiness and server logs.",
    )


def _contract_message_from_validation_error(exc: ValidationError) -> str | None:
    for error in exc.errors(include_url=False):
        candidates: list[object] = []
        context = error.get("ctx")
        if isinstance(context, dict):
            candidates.append(context.get("error"))
        candidates.append(error.get("msg"))
        for candidate in candidates:
            if candidate is None:
                continue
            message = str(candidate)
            if "MEM_" in message or "SCOPE_" in message:
                return message.removeprefix("Value error, ")
    return None


async def _count_query(
    backend: Any,
    sql: str,
    palace: str,
    column: str = "n",
) -> int:
    result = await backend.query(sql, (palace,))
    if not result.rows:
        return 0
    value = result.rows[0].get(column, 0)
    return int(value or 0)


async def _memory_counts_for_project(
    *,
    resources: AppResources,
    palace: str,
) -> tuple[bool, SyncExtractionCounts]:
    empty_counts = SyncExtractionCounts(
        source_items=0,
        structural_evidence=0,
        verification_cycles=0,
        wings=0,
        rooms=0,
        compartments=0,
        semantic_claims=0,
        semantic_memories=0,
    )
    backend = getattr(resources.app_context, "memory_backend", None)
    if backend is None:
        try:
            await refresh_memory_backend(
                app_ctx=resources.app_context,
                executor_registry=resources.executor_registry,
                prefer_metadata=True,
            )
        except Exception:
            logger.warning("unable to refresh memory backend for sync details", exc_info=True)
        backend = getattr(resources.app_context, "memory_backend", None)
        if backend is None:
            return False, empty_counts

    async def _collect_counts() -> tuple[int, int, int, dict[str, Any], int, int]:
        source_items_local = await _count_query(
            backend,
            "SELECT COUNT(*)::int AS n FROM knowledge_items WHERE palace = $1",
            palace,
        )
        structural_evidence_local = await _count_query(
            backend,
            "SELECT COUNT(*)::int AS n FROM knowledge_structural_evidence WHERE palace = $1",
            palace,
        )
        verification_cycles_local = await _count_query(
            backend,
            "SELECT COUNT(*)::int AS n FROM knowledge_verification_cycles WHERE palace = $1",
            palace,
        )
        topology_result = await backend.query(
            """
            SELECT
                COUNT(DISTINCT NULLIF(wing, ''))::int AS wings,
                COUNT(DISTINCT NULLIF(room, ''))::int AS rooms,
                COUNT(DISTINCT NULLIF(compartment, ''))::int AS compartments
            FROM knowledge_structural_evidence
            WHERE palace = $1
            """,
            (palace,),
        )
        topology_local = topology_result.rows[0] if topology_result.rows else {}
        semantic_claims_local = await _count_query(
            backend,
            "SELECT COUNT(*)::int AS n FROM knowledge_semantic_claims WHERE palace = $1",
            palace,
        )
        semantic_memories_local = await _count_query(
            backend,
            "SELECT COUNT(*)::int AS n FROM knowledge_memories WHERE palace = $1",
            palace,
        )
        return (
            source_items_local,
            structural_evidence_local,
            verification_cycles_local,
            topology_local,
            semantic_claims_local,
            semantic_memories_local,
        )

    try:
        backend_lock = getattr(resources.app_context, "memory_backend_lock", None)
        if backend_lock is not None:
            async with backend_lock:
                (
                    source_items,
                    structural_evidence,
                    verification_cycles,
                    topology,
                    semantic_claims,
                    semantic_memories,
                ) = await _collect_counts()
        else:
            (
                source_items,
                structural_evidence,
                verification_cycles,
                topology,
                semantic_claims,
                semantic_memories,
            ) = await _collect_counts()
    except Exception:
        logger.exception("unable to collect memory sync details palace=%s", palace)
        return False, empty_counts

    return True, SyncExtractionCounts(
        source_items=source_items,
        structural_evidence=structural_evidence,
        verification_cycles=verification_cycles,
        wings=int(topology.get("wings", 0) or 0),
        rooms=int(topology.get("rooms", 0) or 0),
        compartments=int(topology.get("compartments", 0) or 0),
        semantic_claims=semantic_claims,
        semantic_memories=semantic_memories,
    )


def _system1_state(*, dirty_count: int, verification_cycles: int) -> str:
    if dirty_count > 0:
        return "queued"
    if verification_cycles > 0:
        return "completed"
    return "not_run"


def _system2_state(*, enabled: bool, semantic_claims: int, semantic_memories: int) -> str:
    if not enabled:
        return "disabled"
    if semantic_claims > 0 or semantic_memories > 0:
        return "completed"
    return "not_run"


def _memory_mode(project: ProjectRecord) -> str:
    return "advanced" if project.system2_enabled else "simple"


def _load_builtin_memory_workflows(resources: AppResources) -> None:
    builtin_path = FilePath(str(files("workflows_mcp").joinpath("templates").joinpath("memory")))
    resources.workflow_registry.load_from_directory(builtin_path)


def _ensure_workflow_loaded(resources: AppResources, workflow_name: str) -> None:
    if resources.workflow_registry.exists(workflow_name):
        return
    if resources.app_context.reload_workflows is not None:
        resources.app_context.reload_workflows()
    else:
        _load_builtin_memory_workflows(resources)
    if not resources.workflow_registry.exists(workflow_name):
        raise RuntimeError(f"Workflow '{workflow_name}' is not loaded")


async def _ensure_memory_runtime(resources: AppResources) -> None:
    if getattr(resources.app_context, "memory_backend", None) is None:
        try:
            await refresh_memory_backend(
                app_ctx=resources.app_context,
                executor_registry=resources.executor_registry,
                prefer_metadata=True,
            )
        except Exception:
            logger.warning("Unable to refresh memory backend before project sync", exc_info=True)
    if getattr(resources.app_context, "memory_backend", None) is None:
        raise MemoryContractError(
            code="MEMORY_BACKEND_UNAVAILABLE",
            message=(
                "MEMORY_BACKEND_UNAVAILABLE: no memory PostgreSQL database is connected. "
                "Save and test the PostgreSQL memory profile before running project sync."
            ),
            retryable=False,
        )
    register_memory_executors(resources.executor_registry)


def _project_sync_inputs(
    *,
    project: ProjectRecord,
    sync_scope: str,
    candidate_paths: list[str],
) -> dict[str, Any]:
    return {
        "project_root": project.fs_root,
        "fs_allowlist": project.fs_allowlist,
        "candidate_paths": candidate_paths,
        "palace": project.palace,
        "source_name": project.slug,
        "default_wing": _normalized_optional(project.default_wing),
        "default_room": _normalized_optional(project.default_room),
        "default_compartment": project.slug,
        "sync_scope": sync_scope,
        "memory_mode": _memory_mode(project),
    }


async def _queue_project_memory_sync(
    *,
    resources: AppResources,
    project: ProjectRecord,
    sync_scope: str,
    candidate_paths: list[str],
) -> tuple[str, str]:
    workflow_name = "project-memory-sync"
    _ensure_workflow_loaded(resources, workflow_name)
    await _ensure_memory_runtime(resources)
    if resources.job_queue is None:
        raise RuntimeError("Job queue is not enabled")
    if not getattr(resources.job_queue, "_running", False):
        await resources.job_queue.start()

    inputs = _project_sync_inputs(
        project=project,
        sync_scope=sync_scope,
        candidate_paths=candidate_paths,
    )
    job_id = await resources.job_queue.submit_job(
        workflow_name,
        inputs,
        project_id=project.id,
        token_id=None,
    )
    return job_id, workflow_name


def _sync_candidate_entries(entries: list[DirtyQueueEntry]) -> list[DirtyQueueEntry]:
    return [
        entry
        for entry in entries
        if entry.path != "." and entry.event_type in {"created", "modified"}
    ]


async def process_project_sync_now(
    *,
    project_id: str,
    resources: AppResources,
) -> SyncNowResponse:
    repo, conn = _repo(resources)
    open_conn: Connection | None = conn
    try:
        _assert_project_exists(repo, project_id)
        projects_repo = SQLiteProjectsRepository(conn)
        project = projects_repo.get_by_id(project_id)
        if project is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "project_not_found", "message": f"Project not found: {project_id}"},
            )

        active_entries = repo.list_active_dirty(project_id=project_id)
        candidate_entries = _sync_candidate_entries(active_entries)
        if not candidate_entries:
            dirty_count = repo.count_active_dirty(project_id=project_id)
            return SyncNowResponse(
                project_id=project_id,
                status="idle",
                dirty_count=dirty_count,
                action="sync",
                memory_mode=_memory_mode(project),
            )
        assert open_conn is not None
        open_conn.close()
        open_conn = None
        candidate_paths = [entry.path for entry in candidate_entries]
        try:
            job_id, workflow_name = await _queue_project_memory_sync(
                resources=resources,
                project=project,
                sync_scope="dirty",
                candidate_paths=candidate_paths,
            )
        except Exception as exc:
            repo2, conn2 = _repo(resources)
            try:
                dirty_count = repo2.count_active_dirty(project_id=project_id)
            finally:
                conn2.close()
            logger.exception(
                "project sync enqueue failed project_id=%s dirty_count=%s error_type=%s",
                project_id,
                dirty_count,
                type(exc).__name__,
            )
            return SyncNowResponse(
                project_id=project_id,
                status="failed",
                dirty_count=dirty_count,
                action="sync",
                memory_mode=_memory_mode(project),
                error=_sync_error_detail(exc),
            )

        repo2, conn2 = _repo(resources)
        try:
            repo2.mark_dirty_entries_processed(
                project_id=project_id,
                entry_ids=[entry.id for entry in candidate_entries],
            )
            dirty_count = repo2.count_active_dirty(project_id=project_id)
        finally:
            conn2.close()
        return SyncNowResponse(
            project_id=project_id,
            status="queued",
            dirty_count=dirty_count,
            action="sync",
            memory_mode=_memory_mode(project),
            job_id=job_id,
            workflow=workflow_name,
        )
    finally:
        if open_conn is not None:
            open_conn.close()


async def queue_project_rebuild(
    *,
    project_id: str,
    resources: AppResources,
    reason: str = "reconciliation_required:manual_rebuild",
) -> SyncNowResponse:
    """Queue a non-destructive full project rescan through the async workflow runner."""
    return await _enqueue_project_reconciliation(
        project_id=project_id,
        action="rebuild",
        sync_scope="rebuild",
        event_type="rebuild",
        reason=reason,
        resources=resources,
    )


@router.get(
    "",
    response_model=SyncListResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def list_sync_queue(
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> SyncListResponse:
    repo, conn = _repo(resources)
    try:
        summaries = [
            SyncProjectSummary(
                project_id=row.project_id,
                dirty_count=row.dirty_count,
                requires_reconciliation=row.requires_reconciliation,
            )
            for row in repo.list_queue_summaries()
        ]
        return SyncListResponse(projects=summaries)
    finally:
        conn.close()


@router.get(
    "/{project_id}/logs",
    response_model=SyncLogsResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def list_sync_logs(
    project_id: ProjectId,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> SyncLogsResponse:
    repo, conn = _repo(resources)
    try:
        _assert_project_exists(repo, project_id)
        entries = [
            SyncLogEntry(
                id=entry.id,
                project_id=entry.project_id,
                path=entry.path,
                event_type=entry.event_type,
                reason=entry.reason,
                status="processed" if entry.processed_at is not None else "queued",
                enqueued_at=entry.enqueued_at,
                updated_at=entry.updated_at,
                processed_at=entry.processed_at,
            )
            for entry in repo.list_dirty_history(project_id=project_id, limit=limit)
        ]
        return SyncLogsResponse(project_id=project_id, entries=entries)
    finally:
        conn.close()


@router.get(
    "/{project_id}/details",
    response_model=SyncProjectDetailsResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def get_sync_details(
    project_id: ProjectId,
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> SyncProjectDetailsResponse:
    repo, conn = _repo(resources)
    try:
        _assert_project_exists(repo, project_id)
        project = SQLiteProjectsRepository(conn).get_by_id(project_id)
        if project is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "project_not_found", "message": f"Project not found: {project_id}"},
            )
        memory_backend_ready, counts = await _memory_counts_for_project(
            resources=resources,
            palace=project.palace,
        )
        dirty_count = repo.count_active_dirty(project_id=project_id)
        return SyncProjectDetailsResponse(
            project_id=project_id,
            memory_mode=_memory_mode(project),
            system1_enabled=True,
            system1_state=_system1_state(
                dirty_count=dirty_count,
                verification_cycles=counts.verification_cycles,
            ),
            system2_enabled=project.system2_enabled,
            system2_state=_system2_state(
                enabled=project.system2_enabled,
                semantic_claims=counts.semantic_claims,
                semantic_memories=counts.semantic_memories,
            ),
            embedding_profile_required=project.system2_enabled,
            embedding_profile="embedding",
            memory_backend_ready=memory_backend_ready,
            counts=counts,
        )
    finally:
        conn.close()


@router.post(
    "/{project_id}/now",
    response_model=SyncNowResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def sync_now(
    project_id: ProjectId,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> SyncNowResponse:
    return await process_project_sync_now(project_id=project_id, resources=resources)


async def _enqueue_project_reconciliation(
    *,
    project_id: str,
    action: str,
    sync_scope: str,
    event_type: str,
    reason: str,
    resources: AppResources,
) -> SyncNowResponse:
    repo, conn = _repo(resources)
    try:
        try:
            repo.enqueue_dirty(
                project_id=project_id,
                path=".",
                event_type=event_type,
                reason=reason,
            )
        except UnknownProjectError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "project_not_found", "message": f"Project not found: {project_id}"},
            ) from None

        projects_repo = SQLiteProjectsRepository(conn)
        project = projects_repo.get_by_id(project_id)
        if project is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "project_not_found", "message": f"Project not found: {project_id}"},
            )
    finally:
        conn.close()

    try:
        job_id, workflow_name = await _queue_project_memory_sync(
            resources=resources,
            project=project,
            sync_scope=sync_scope,
            candidate_paths=[],
        )
    except Exception as exc:
        repo2, conn2 = _repo(resources)
        try:
            dirty_count = repo2.count_active_dirty(project_id=project_id)
        finally:
            conn2.close()
        logger.exception(
            "project %s enqueue failed project_id=%s dirty_count=%s error_type=%s",
            action,
            project_id,
            dirty_count,
            type(exc).__name__,
        )
        return SyncNowResponse(
            project_id=project_id,
            status="failed",
            dirty_count=dirty_count,
            action=action,
            memory_mode=_memory_mode(project),
            error=_sync_error_detail(exc),
        )

    repo3, conn3 = _repo(resources)
    try:
        repo3.mark_active_dirty_processed(project_id=project_id)
        dirty_count = repo3.count_active_dirty(project_id=project_id)
    finally:
        conn3.close()
    return SyncNowResponse(
        project_id=project_id,
        status="queued",
        dirty_count=dirty_count,
        action=action,
        memory_mode=_memory_mode(project),
        job_id=job_id,
        workflow=workflow_name,
    )


@router.post(
    "/{project_id}/reconcile",
    response_model=SyncNowResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def sync_reconcile(
    project_id: ProjectId,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> SyncNowResponse:
    return await _enqueue_project_reconciliation(
        project_id=project_id,
        action="reconcile",
        sync_scope="reconcile",
        event_type="reconcile",
        reason="reconciliation_required:manual_reconcile",
        resources=resources,
    )


@router.post(
    "/{project_id}/rebuild",
    response_model=SyncNowResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def sync_rebuild(
    project_id: ProjectId,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> SyncNowResponse:
    return await queue_project_rebuild(
        project_id=project_id,
        reason="reconciliation_required:manual_rebuild",
        resources=resources,
    )
