from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from sqlite3 import Connection

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from workflows_mcp.http.dependencies import (
    CurrentAdminSession,
    get_resources,
    require_current_admin_session,
)
from workflows_mcp.http.lifespan import AppResources
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.repos.watcher_repo import SQLiteWatcherRepository

router = APIRouter(prefix="/api/events/v1", tags=["events-v1"])


class WatcherEventItem(BaseModel):
    project_id: str
    state: str
    dirty_count: int
    requires_reconciliation: bool
    last_event_at: str | None
    updated_at: str


class WatcherStatusPayload(BaseModel):
    version: int
    items: list[WatcherEventItem]


class SyncEventItem(BaseModel):
    project_id: str
    dirty_count: int
    requires_reconciliation: bool


class SyncStatusPayload(BaseModel):
    version: int
    items: list[SyncEventItem]


def _repo(resources: AppResources) -> tuple[SQLiteWatcherRepository, Connection]:
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    return SQLiteWatcherRepository(conn), conn


def _watcher_status_payload(resources: AppResources) -> WatcherStatusPayload:
    repo, conn = _repo(resources)
    try:
        statuses = repo.list_statuses()
        items = [
            WatcherEventItem(
                project_id=status.project_id,
                state=status.state,
                dirty_count=repo.count_active_dirty(project_id=status.project_id),
                requires_reconciliation=repo.requires_reconciliation(project_id=status.project_id),
                last_event_at=status.last_event_at,
                updated_at=status.updated_at,
            )
            for status in statuses
        ]
        return WatcherStatusPayload(version=1, items=items)
    finally:
        conn.close()


def _sync_status_payload(resources: AppResources) -> SyncStatusPayload:
    repo, conn = _repo(resources)
    try:
        summaries = repo.list_queue_summaries()
        items = [
            SyncEventItem(
                project_id=summary.project_id,
                dirty_count=summary.dirty_count,
                requires_reconciliation=summary.requires_reconciliation,
            )
            for summary in summaries
        ]
        return SyncStatusPayload(version=1, items=items)
    finally:
        conn.close()


async def _live_status_event_stream(
    event_name: str,
    payload_factory: Callable[[], dict[str, object]],
    *,
    interval_seconds: float = 15.0,
    max_events: int | None = None,
) -> AsyncIterator[bytes]:
    emitted = 0
    while max_events is None or emitted < max_events:
        payload = payload_factory()
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        yield f"event: {event_name}\ndata: {encoded}\n\n".encode()
        emitted += 1
        if max_events is not None and emitted >= max_events:
            return
        await asyncio.sleep(interval_seconds)


@router.get(
    "/system",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {
                "text/event-stream": {},
            }
        }
    },
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def system_events(
    _: CurrentAdminSession = Depends(require_current_admin_session),
) -> StreamingResponse:
    """Return minimal SSE stream for v1 events namespace."""

    async def _stream() -> AsyncIterator[bytes]:
        yield b"event: system\ndata: {\"status\":\"ok\"}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.get(
    "/watchers",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def watcher_events(
    _: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> StreamingResponse:
    return StreamingResponse(
        _live_status_event_stream(
            "watcher.status",
            lambda: _watcher_status_payload(resources).model_dump(mode="json"),
        ),
        media_type="text/event-stream",
    )


@router.get(
    "/watchers/state",
    response_class=JSONResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def watcher_events_state(
    _: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> JSONResponse:
    payload = _watcher_status_payload(resources).model_dump(mode="json")
    return JSONResponse(payload)


@router.get(
    "/sync",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def sync_events(
    _: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> StreamingResponse:
    return StreamingResponse(
        _live_status_event_stream(
            "sync.status",
            lambda: _sync_status_payload(resources).model_dump(mode="json"),
        ),
        media_type="text/event-stream",
    )


@router.get(
    "/sync/state",
    response_class=JSONResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def sync_events_state(
    _: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> JSONResponse:
    payload = _sync_status_payload(resources).model_dump(mode="json")
    return JSONResponse(payload)
