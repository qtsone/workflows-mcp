from __future__ import annotations

import logging
from pathlib import Path as FilePath
from sqlite3 import Connection
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, ValidationError

from workflows_mcp.engine.execution import Execution
from workflows_mcp.engine.memory_onboard_sync_orchestrator import (
    ProgrammaticOnboardRequest,
    classify_scan_files_for_programmatic_mode,
    run_programmatic_onboard,
)
from workflows_mcp.engine.memory_service import MemoryContractError
from workflows_mcp.http.dependencies import (
    CurrentAdminSession,
    get_resources,
    require_admin_csrf,
    require_current_admin_session,
)
from workflows_mcp.http.lifespan import AppResources
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.repos.projects_repo import ProjectRecord, SQLiteProjectsRepository
from workflows_mcp.metadata.repos.watcher_repo import SQLiteWatcherRepository, UnknownProjectError
from workflows_mcp.tools_memory import persist_graph_payload
from workflows_mcp.watcher.scanner import scan_project_files

router = APIRouter(prefix="/sync")
logger = logging.getLogger(__name__)

ProjectId = Annotated[str, Path(min_length=1, max_length=128)]
DEFAULT_PROJECT_WING = "default-wing"
DEFAULT_PROJECT_ROOM = "default-room"


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


class SyncNowResponse(BaseModel):
    project_id: str
    status: str
    dirty_count: int
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
        "wing": project.default_wing or DEFAULT_PROJECT_WING,
        "room": project.default_room or DEFAULT_PROJECT_ROOM,
        "compartment": project.slug,
    }


def _scan_entries_for_project(project: ProjectRecord) -> list[dict[str, Any]]:
    project_root = FilePath(project.fs_root)
    entries: list[dict[str, Any]] = []
    for relative_path in scan_project_files(project_root):
        absolute_path = project_root / relative_path
        try:
            stat = absolute_path.stat()
            size_bytes = stat.st_size
        except OSError:
            size_bytes = 0
        entries.append(
            {
                "path": relative_path.as_posix(),
                "content": "",
                "size_bytes": size_bytes,
            }
        )
    return entries


def _admin_memory_execution(resources: AppResources) -> Execution:
    execution = Execution()
    execution.set_execution_context(
        resources.app_context.create_execution_context(auth_method="ADMIN_SESSION")
    )
    return execution


def _sync_error_detail(exc: Exception) -> SyncErrorDetail:
    if isinstance(exc, MemoryContractError):
        return SyncErrorDetail(code="project_graph_sync_failed", message=exc.message)
    if isinstance(exc, ValidationError):
        contract_message = _contract_message_from_validation_error(exc)
        if contract_message:
            return SyncErrorDetail(code="project_graph_sync_failed", message=contract_message)
    return SyncErrorDetail(
        code="project_graph_sync_failed",
        message="Project graph sync failed; check memory database readiness and server logs.",
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


async def _persist_project_graph_from_scan(
    *,
    resources: AppResources,
    project: ProjectRecord,
) -> dict[str, int]:
    scanned_entries = _scan_entries_for_project(project)
    if not scanned_entries:
        return {"nodes": 0, "corridors": 0}

    scope = _scope_for_project(project)
    file_entries = classify_scan_files_for_programmatic_mode(
        scanned_entries,
        base_path=project.fs_root,
    )
    result = run_programmatic_onboard(
        ProgrammaticOnboardRequest(
            scope=scope,
            files=file_entries,
            mode="programmatic",
            provenance="admin_sync",
            confidence=1.0,
        )
    )
    if result.status != "completed" or result.graph is None:
        message = "Project graph onboarding failed"
        if result.error is not None:
            error = result.error.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                message = str(error["message"])
        raise RuntimeError(message)

    persisted = await persist_graph_payload(
        app_ctx=resources.app_context,
        execution=_admin_memory_execution(resources),
        scope=result.scope,
        graph=result.graph,
        response=None,
    )
    return {"nodes": int(persisted["nodes"]), "corridors": int(persisted["corridors"])}


async def process_project_sync_now(
    *,
    project_id: str,
    resources: AppResources,
) -> SyncNowResponse:
    repo, conn = _repo(resources)
    try:
        _assert_project_exists(repo, project_id)
        projects_repo = SQLiteProjectsRepository(conn)
        project = projects_repo.get_by_id(project_id)
        if project is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "project_not_found", "message": f"Project not found: {project_id}"},
            )

        try:
            await _persist_project_graph_from_scan(resources=resources, project=project)
        except Exception as exc:
            dirty_count = repo.count_active_dirty(project_id=project_id)
            logger.exception(
                "project graph sync failed project_id=%s dirty_count=%s error_type=%s",
                project_id,
                dirty_count,
                type(exc).__name__,
            )
            return SyncNowResponse(
                project_id=project_id,
                status="failed",
                dirty_count=dirty_count,
                error=_sync_error_detail(exc),
            )

        repo.mark_active_dirty_processed(project_id=project_id)

        dirty_count = repo.count_active_dirty(project_id=project_id)
        return SyncNowResponse(
            project_id=project_id,
            status="queued" if dirty_count > 0 else "idle",
            dirty_count=dirty_count,
        )
    finally:
        conn.close()


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

    finally:
        conn.close()
    return await process_project_sync_now(project_id=project_id, resources=resources)


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
    return await _enqueue_project_reconciliation(
        project_id=project_id,
        event_type="rebuild",
        reason="reconciliation_required:manual_rebuild",
        resources=resources,
    )
