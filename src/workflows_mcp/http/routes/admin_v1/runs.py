from __future__ import annotations

from sqlite3 import Connection
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, ConfigDict, StringConstraints

from workflows_mcp.http.dependencies import (
    CurrentAdminSession,
    get_resources,
    require_admin_csrf,
    require_current_admin_session,
)
from workflows_mcp.http.lifespan import AppResources
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.repos.run_history_repo import (
    InvalidRunHistoryPaginationError,
    RunRecord,
    SQLiteRunHistoryRepository,
)

router = APIRouter(prefix="/runs")

RunId = Annotated[str, Path(min_length=1, max_length=128)]
RunStatus = Annotated[str, StringConstraints(min_length=1, max_length=64)]


class ErrorDetail(BaseModel):
    code: str
    message: str


class RunRowResponse(BaseModel):
    run_id: str
    job_id: str
    workflow_name: str
    status: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    updated_at: str
    cancellable: bool
    project_id: str | None
    token_id: str | None


class RunsListResponse(BaseModel):
    runs: list[RunRowResponse]


class RunDetailResponse(RunRowResponse):
    result_summary: str | None
    error_summary: str | None


class ResumeRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: str = ""


def _repo(resources: AppResources) -> tuple[SQLiteRunHistoryRepository, Connection]:
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    return SQLiteRunHistoryRepository(conn), conn


def _to_row(run: RunRecord) -> RunRowResponse:
    return RunRowResponse(
        run_id=run.run_id,
        job_id=run.run_id,
        workflow_name=run.workflow_name,
        status=run.status,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        updated_at=run.updated_at,
        cancellable=run.cancellable,
        project_id=run.project_id,
        token_id=run.token_id,
    )


def _not_found(run_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=ErrorDetail(code="run_not_found", message=f"Run not found: {run_id}").model_dump(),
    )


def _conflict(code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=ErrorDetail(code=code, message=message).model_dump(),
    )


def _bad_request(code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=ErrorDetail(code=code, message=message).model_dump(),
    )


@router.get(
    "",
    response_model=RunsListResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def list_runs(
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
    status_filter: RunStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> RunsListResponse:
    repo, conn = _repo(resources)
    try:
        try:
            runs = repo.list_runs(limit=limit, offset=offset, status=status_filter)
        except InvalidRunHistoryPaginationError as exc:
            raise _bad_request("invalid_pagination", str(exc)) from exc
        return RunsListResponse(runs=[_to_row(run) for run in runs])
    finally:
        conn.close()


@router.get(
    "/{run_id}",
    response_model=RunDetailResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def get_run_detail(
    run_id: RunId,
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> RunDetailResponse:
    repo, conn = _repo(resources)
    try:
        run = repo.get_run(run_id)
        if run is None:
            raise _not_found(run_id)
        row = _to_row(run)
        return RunDetailResponse(
            **row.model_dump(),
            result_summary=run.result_summary,
            error_summary=run.error_summary,
        )
    finally:
        conn.close()


@router.post(
    "/{run_id}/cancel",
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def cancel_run(
    run_id: RunId,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> dict[str, Any]:
    repo, conn = _repo(resources)
    try:
        run = repo.get_run(run_id)
        if run is None:
            raise _not_found(run_id)
    finally:
        conn.close()

    if resources.job_queue is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(
                code="job_queue_unavailable",
                message="Job queue unavailable",
            ).model_dump(),
        )

    return await resources.job_queue.cancel_job(run_id)


@router.post(
    "/{run_id}/resume",
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def resume_run(
    run_id: RunId,
    body: ResumeRunRequest | None = None,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> dict[str, Any]:
    repo, conn = _repo(resources)
    try:
        run = repo.get_run(run_id)
        if run is None:
            raise _not_found(run_id)
    finally:
        conn.close()

    if resources.job_queue is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(
                code="job_queue_unavailable",
                message="Job queue unavailable",
            ).model_dump(),
        )

    resume_input = body.response if body is not None else ""
    resume_result = await resources.job_queue.resume_job(run_id, response=resume_input)

    outcome = resume_result.get("outcome")
    if outcome == "not_found":
        raise _conflict(
            "run_not_resumable",
            str(resume_result.get("message", "Run is not resumable")),
        )
    if outcome == "not_resumable":
        raise _conflict(
            "run_not_resumable",
            str(resume_result.get("message", "Run is not resumable")),
        )
    return resume_result
