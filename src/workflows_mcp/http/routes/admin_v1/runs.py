from __future__ import annotations

import json
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
    execution_mode: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    updated_at: str
    duration_ms: int | None
    cancellable: bool
    project_id: str | None
    token_id: str | None


class RunsListResponse(BaseModel):
    runs: list[RunRowResponse]
    total: int
    limit: int
    offset: int


class RunBlockResponse(BaseModel):
    block_id: str
    block_type: str | None
    status: str | None
    outcome: str | None
    duration_ms: int | None
    message: str | None
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    metadata: dict[str, Any]


class RunDetailResponse(RunRowResponse):
    result_summary: str | None
    error_summary: str | None
    inputs: dict[str, Any]
    outputs: Any
    error: str | None
    metadata: dict[str, Any]
    blocks: list[RunBlockResponse]
    technical_json: dict[str, Any]


class ResumeRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: str = ""


def _repo(resources: AppResources) -> tuple[SQLiteRunHistoryRepository, Connection]:
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    return SQLiteRunHistoryRepository(conn), conn


def _parse_json_object(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _duration_ms(run: RunRecord, execution_json: dict[str, Any] | None = None) -> int | None:
    metadata = execution_json.get("metadata") if execution_json else None
    if isinstance(metadata, dict):
        seconds = metadata.get("execution_time_seconds")
        if isinstance(seconds, int | float):
            return max(0, int(seconds * 1000))
    return None


def _to_row(run: RunRecord, execution_json: dict[str, Any] | None = None) -> RunRowResponse:
    return RunRowResponse(
        run_id=run.run_id,
        job_id=run.run_id,
        workflow_name=run.workflow_name,
        status=run.status,
        execution_mode=run.execution_mode,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        updated_at=run.updated_at,
        duration_ms=_duration_ms(run, execution_json),
        cancellable=run.cancellable,
        project_id=run.project_id,
        token_id=run.token_id,
    )


def _to_block_rows(execution_json: dict[str, Any]) -> list[RunBlockResponse]:
    blocks = execution_json.get("blocks")
    if not isinstance(blocks, dict):
        return []
    rows: list[RunBlockResponse] = []
    for block_id, raw_block in blocks.items():
        block = raw_block if isinstance(raw_block, dict) else {}
        metadata = block.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        inputs = block.get("inputs")
        outputs = block.get("outputs")
        duration_ms = metadata_dict.get("duration_ms")
        rows.append(
            RunBlockResponse(
                block_id=str(block_id),
                block_type=(
                    str(metadata_dict["type"])
                    if isinstance(metadata_dict.get("type"), str)
                    else None
                ),
                status=(
                    str(metadata_dict["status"])
                    if isinstance(metadata_dict.get("status"), str)
                    else None
                ),
                outcome=(
                    str(metadata_dict["outcome"])
                    if isinstance(metadata_dict.get("outcome"), str)
                    else None
                ),
                duration_ms=duration_ms if isinstance(duration_ms, int) else None,
                message=(
                    str(metadata_dict["message"])
                    if isinstance(metadata_dict.get("message"), str)
                    else None
                ),
                inputs=inputs if isinstance(inputs, dict) else {},
                outputs=outputs if isinstance(outputs, dict) else {},
                metadata=metadata_dict,
            )
        )
    return rows


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
    mode_filter: RunStatus | None = Query(default=None, alias="mode"),
    workflow_filter: RunStatus | None = Query(default=None, alias="workflow"),
    project_filter: RunStatus | None = Query(default=None, alias="project_id"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> RunsListResponse:
    repo, conn = _repo(resources)
    try:
        try:
            runs = repo.list_runs(
                limit=limit,
                offset=offset,
                status=status_filter,
                execution_mode=mode_filter,
                workflow_name=workflow_filter,
                project_id=project_filter,
            )
            total = repo.count_runs(
                status=status_filter,
                execution_mode=mode_filter,
                workflow_name=workflow_filter,
                project_id=project_filter,
            )
        except InvalidRunHistoryPaginationError as exc:
            raise _bad_request("invalid_pagination", str(exc)) from exc
        return RunsListResponse(
            runs=[_to_row(run, _parse_json_object(run.execution_json)) for run in runs],
            total=total,
            limit=limit,
            offset=offset,
        )
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
        execution_json = _parse_json_object(run.execution_json)
        row = _to_row(run, execution_json)
        inputs = _parse_json_object(run.inputs_json)
        return RunDetailResponse(
            **row.model_dump(),
            result_summary=run.result_summary,
            error_summary=run.error_summary,
            inputs=inputs,
            outputs=execution_json.get("outputs"),
            error=(
                str(execution_json["error"])
                if isinstance(execution_json.get("error"), str)
                else run.error_summary
            ),
            metadata=(
                execution_json["metadata"]
                if isinstance(execution_json.get("metadata"), dict)
                else {}
            ),
            blocks=_to_block_rows(execution_json),
            technical_json=execution_json,
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
