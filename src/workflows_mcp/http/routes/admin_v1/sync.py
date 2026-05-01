from __future__ import annotations

from pathlib import Path as FilePath
from sqlite3 import Connection
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel

from workflows_mcp.http.dependencies import (
    CurrentAdminSession,
    get_resources,
    require_admin_csrf,
    require_current_admin_session,
)
from workflows_mcp.http.lifespan import AppResources
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.repos.projects_repo import SQLiteProjectsRepository
from workflows_mcp.metadata.repos.watcher_repo import SQLiteWatcherRepository, UnknownProjectError
from workflows_mcp.watcher.scanner import scan_project_files

router = APIRouter(prefix="/sync")

ProjectId = Annotated[str, Path(min_length=1, max_length=128)]


class SyncProjectSummary(BaseModel):
    project_id: str
    dirty_count: int
    requires_reconciliation: bool


class SyncListResponse(BaseModel):
    projects: list[SyncProjectSummary]


class SyncNowResponse(BaseModel):
    project_id: str
    status: str
    dirty_count: int


class SyncReconcileResponse(BaseModel):
    project_id: str
    dirty_count: int
    requires_reconciliation: bool


def _repo(resources: AppResources) -> tuple[SQLiteWatcherRepository, Connection]:
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    return SQLiteWatcherRepository(conn), conn


def _assert_project_exists(repo: SQLiteWatcherRepository, project_id: str) -> None:
    if not repo.project_exists(project_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "project_not_found", "message": f"Project not found: {project_id}"},
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
            scanned_paths = scan_project_files(FilePath(project.fs_root))
        except FileNotFoundError:
            scanned_paths = []

        for scanned_path in scanned_paths:
            repo.enqueue_dirty(
                project_id=project_id,
                path=scanned_path.as_posix(),
                event_type="modified",
                reason="scan_now",
            )

        dirty_count = repo.count_active_dirty(project_id=project_id)
        return SyncNowResponse(
            project_id=project_id,
            status="queued" if dirty_count > 0 else "idle",
            dirty_count=dirty_count,
        )
    finally:
        conn.close()


def _enqueue_project_reconciliation(
    *,
    project_id: str,
    event_type: str,
    reason: str,
    resources: AppResources,
) -> SyncReconcileResponse:
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

        return SyncReconcileResponse(
            project_id=project_id,
            dirty_count=repo.count_active_dirty(project_id=project_id),
            requires_reconciliation=True,
        )
    finally:
        conn.close()


@router.post(
    "/{project_id}/reconcile",
    response_model=SyncReconcileResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def sync_reconcile(
    project_id: ProjectId,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> SyncReconcileResponse:
    return _enqueue_project_reconciliation(
        project_id=project_id,
        event_type="reconcile",
        reason="reconciliation_required:manual_reconcile",
        resources=resources,
    )


@router.post(
    "/{project_id}/rebuild",
    response_model=SyncReconcileResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def sync_rebuild(
    project_id: ProjectId,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> SyncReconcileResponse:
    return _enqueue_project_reconciliation(
        project_id=project_id,
        event_type="rebuild",
        reason="reconciliation_required:manual_rebuild",
        resources=resources,
    )
