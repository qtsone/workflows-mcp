from __future__ import annotations

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
from workflows_mcp.metadata.repos.watcher_repo import (
    SQLiteWatcherRepository,
    UnknownProjectError,
    WatcherStatusRecord,
)

router = APIRouter(prefix="/watchers")

ProjectId = Annotated[str, Path(min_length=1, max_length=128)]


class WatcherStatusResponse(BaseModel):
    project_id: str
    state: str
    dirty_count: int
    requires_reconciliation: bool
    last_event_at: str | None
    updated_at: str


class WatchersListResponse(BaseModel):
    watchers: list[WatcherStatusResponse]


def _repo(resources: AppResources) -> tuple[SQLiteWatcherRepository, Connection]:
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    return SQLiteWatcherRepository(conn), conn


def _to_status_response(
    *,
    record: WatcherStatusRecord,
    dirty_count: int,
    requires_reconciliation: bool,
) -> WatcherStatusResponse:
    return WatcherStatusResponse(
        project_id=record.project_id,
        state=record.state,
        dirty_count=dirty_count,
        requires_reconciliation=requires_reconciliation,
        last_event_at=record.last_event_at,
        updated_at=record.updated_at,
    )


def _not_found(project_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "code": "watcher_status_not_found",
            "message": f"Watcher status not found: {project_id}",
        },
    )


@router.get(
    "",
    response_model=WatchersListResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def list_watchers(
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> WatchersListResponse:
    repo, conn = _repo(resources)
    try:
        statuses = repo.list_statuses()
        payload = [
            _to_status_response(
                record=status,
                dirty_count=repo.count_active_dirty(project_id=status.project_id),
                requires_reconciliation=repo.requires_reconciliation(project_id=status.project_id),
            )
            for status in statuses
        ]
        return WatchersListResponse(watchers=payload)
    finally:
        conn.close()


@router.get(
    "/{project_id}",
    response_model=WatcherStatusResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def get_watcher_status(
    project_id: ProjectId,
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> WatcherStatusResponse:
    repo, conn = _repo(resources)
    try:
        record = repo.get_status(project_id)
        if record is None:
            raise _not_found(project_id)
        return _to_status_response(
            record=record,
            dirty_count=repo.count_active_dirty(project_id=project_id),
            requires_reconciliation=repo.requires_reconciliation(project_id=project_id),
        )
    finally:
        conn.close()


def _persist_state(
    *,
    project_id: str,
    state: str,
    resources: AppResources,
) -> WatcherStatusResponse:
    repo, conn = _repo(resources)
    try:
        try:
            record = repo.set_status(project_id=project_id, state=state)
        except UnknownProjectError:
            raise _not_found(project_id) from None
        return _to_status_response(
            record=record,
            dirty_count=repo.count_active_dirty(project_id=project_id),
            requires_reconciliation=repo.requires_reconciliation(project_id=project_id),
        )
    finally:
        conn.close()


@router.post(
    "/{project_id}/pause",
    response_model=WatcherStatusResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def pause_watcher(
    project_id: ProjectId,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> WatcherStatusResponse:
    response = _persist_state(project_id=project_id, state="paused", resources=resources)
    resources.watcher_manager.stop_project_watcher(project_id)
    return response


@router.post(
    "/{project_id}/resume",
    response_model=WatcherStatusResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def resume_watcher(
    project_id: ProjectId,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> WatcherStatusResponse:
    response = _persist_state(project_id=project_id, state="enabled", resources=resources)
    if resources.watcher_manager.is_started:
        resources.watcher_manager.start_project_watcher(project_id)
    return response


@router.post(
    "/{project_id}/disable",
    response_model=WatcherStatusResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def disable_watcher(
    project_id: ProjectId,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> WatcherStatusResponse:
    response = _persist_state(project_id=project_id, state="disabled", resources=resources)
    resources.watcher_manager.stop_project_watcher(project_id)
    return response
