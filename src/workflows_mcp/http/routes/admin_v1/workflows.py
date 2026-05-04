from __future__ import annotations

from pathlib import Path as FilePath
from sqlite3 import Connection
from typing import Annotated, Any

import yaml
from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel, ConfigDict, StringConstraints

from workflows_mcp.engine.registry import WorkflowRegistry
from workflows_mcp.engine.workflow_source_loader import (
    WorkflowSourceReloadError,
    reload_registry_from_source_paths,
)
from workflows_mcp.http.dependencies import (
    CurrentAdminSession,
    get_resources,
    require_admin_csrf,
    require_current_admin_session,
)
from workflows_mcp.http.lifespan import AppResources
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.repos.workflow_sources_repo import (
    DuplicateWorkflowSourceError,
    InvalidWorkflowSourcePathError,
    SQLiteWorkflowSourcesRepository,
    WorkflowSourceCreate,
    WorkflowSourceNotFoundError,
    WorkflowSourceProjectNotFoundError,
    WorkflowSourceRecord,
)

router = APIRouter(prefix="/workflows")

SourceId = Annotated[str, Path(min_length=1, max_length=128)]
WorkflowName = Annotated[str, Path(min_length=1, max_length=256)]
NonEmptyString = Annotated[str, StringConstraints(min_length=1, max_length=1024)]


class ErrorDetail(BaseModel):
    code: str
    message: str


class WorkflowSourceResponse(BaseModel):
    source_id: str
    project_id: str
    source_path: str
    checksum: str | None
    discovered_at: str
    last_loaded_at: str | None
    status: str | None
    error_message: str | None


class WorkflowSourcesListResponse(BaseModel):
    sources: list[WorkflowSourceResponse]


class CreateWorkflowSourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: NonEmptyString
    source_path: NonEmptyString
    checksum: str | None = None


class DeleteWorkflowSourceResponse(BaseModel):
    deleted: bool


class WorkflowReloadResponse(BaseModel):
    status: str
    total: int
    source_count: int
    workflow_names: list[str]


class WorkflowSummaryResponse(BaseModel):
    name: str
    description: str
    version: str
    tags: list[str]
    source_path: str | None


class WorkflowsListResponse(BaseModel):
    workflows: list[WorkflowSummaryResponse]


class WorkflowValidateResponse(BaseModel):
    valid: bool
    workflow_names: list[str]
    total: int


def _workflow_yaml_candidates(source_dir: FilePath) -> list[FilePath]:
    candidates: list[FilePath] = []
    for pattern in ("*.yaml", "*.yml"):
        try:
            paths = source_dir.rglob(pattern)
            for path in paths:
                try:
                    if path.is_file():
                        candidates.append(path)
                except OSError:
                    continue
        except OSError:
            continue
    return sorted(candidates)


def _workflow_yaml_detail(workflow_name: str, source: FilePath | None) -> dict[str, Any]:
    if source is None:
        return {
            "raw_yaml": None,
            "yaml_path": None,
            "load_logs": ["Workflow YAML not found because no source path is registered."],
        }

    source_dir = FilePath(source)
    try:
        source_available = source_dir.exists() and source_dir.is_dir()
    except OSError:
        source_available = False

    if not source_available:
        return {
            "raw_yaml": None,
            "yaml_path": None,
            "load_logs": [
                f"Workflow YAML not found because source path is unavailable: {source_dir}"
            ],
        }

    load_logs: list[str] = []
    for yaml_path in _workflow_yaml_candidates(source_dir):
        try:
            raw_yaml = yaml_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            load_logs.append(f"Skipped unreadable YAML candidate: {yaml_path}")
            continue

        try:
            parsed = yaml.safe_load(raw_yaml)
        except yaml.YAMLError:
            load_logs.append(f"Skipped unparsable YAML candidate: {yaml_path}")
            continue

        if isinstance(parsed, dict):
            name = parsed.get("name")
            if isinstance(name, str) and name == workflow_name:
                return {
                    "raw_yaml": raw_yaml,
                    "yaml_path": str(yaml_path),
                    "load_logs": [f"Workflow YAML loaded from: {yaml_path}"],
                }

    load_logs.append(f"Workflow YAML not found for workflow: {workflow_name}")

    return {
        "raw_yaml": None,
        "yaml_path": None,
        "load_logs": load_logs,
    }


def _repo(resources: AppResources) -> tuple[SQLiteWorkflowSourcesRepository, Connection]:
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    return SQLiteWorkflowSourcesRepository(conn), conn


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=ErrorDetail(code=code, message=message).model_dump(),
    )


def _to_source_response(source: WorkflowSourceRecord) -> WorkflowSourceResponse:
    return WorkflowSourceResponse(
        source_id=source.source_id,
        project_id=source.project_id,
        source_path=source.source_path,
        checksum=source.checksum,
        discovered_at=source.discovered_at,
        last_loaded_at=source.last_loaded_at,
        status=source.status,
        error_message=source.error_message,
    )


def _map_reload_error(exc: WorkflowSourceReloadError) -> HTTPException:
    if exc.code == "workflow_duplicate_name":
        return _error(status.HTTP_409_CONFLICT, exc.code, exc.message)
    if exc.code in {"workflow_invalid_definition", "workflow_source_invalid_path"}:
        return _error(status.HTTP_422_UNPROCESSABLE_CONTENT, exc.code, exc.message)
    return _error(status.HTTP_422_UNPROCESSABLE_CONTENT, "workflow_reload_failed", str(exc))


def _reload_from_sqlite_sources(resources: AppResources) -> WorkflowReloadResponse:
    if resources.app_context.reload_workflows is not None:
        try:
            summary = resources.app_context.reload_workflows()
        except WorkflowSourceReloadError as exc:
            raise _map_reload_error(exc) from exc
        return WorkflowReloadResponse(
            status="ok",
            total=summary.workflow_count,
            source_count=summary.source_count,
            workflow_names=summary.workflow_names,
        )

    repo, conn = _repo(resources)
    try:
        sources = repo.list()
        source_paths = [source.source_path for source in sources]
        try:
            summary = reload_registry_from_source_paths(resources.workflow_registry, source_paths)
        except WorkflowSourceReloadError as exc:
            for source in sources:
                repo.update_reload_state(
                    source.source_id,
                    status="failed",
                    error_message=exc.message,
                )
            raise _map_reload_error(exc) from exc

        for source in sources:
            repo.update_reload_state(source.source_id, status="loaded", error_message=None)

        return WorkflowReloadResponse(
            status="ok",
            total=summary.workflow_count,
            source_count=summary.source_count,
            workflow_names=summary.workflow_names,
        )
    finally:
        conn.close()


@router.get(
    "/sources",
    response_model=WorkflowSourcesListResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def list_workflow_sources(
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> WorkflowSourcesListResponse:
    repo, conn = _repo(resources)
    try:
        sources = [_to_source_response(item) for item in repo.list()]
        return WorkflowSourcesListResponse(sources=sources)
    finally:
        conn.close()


@router.post(
    "/sources",
    status_code=status.HTTP_201_CREATED,
    response_model=WorkflowSourceResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def create_workflow_source(
    body: CreateWorkflowSourceRequest,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> WorkflowSourceResponse:
    repo, conn = _repo(resources)
    try:
        try:
            source = repo.create(
                WorkflowSourceCreate(
                    project_id=body.project_id,
                    source_path=body.source_path,
                    checksum=body.checksum,
                )
            )
        except WorkflowSourceProjectNotFoundError as exc:
            raise _error(status.HTTP_404_NOT_FOUND, "project_not_found", str(exc)) from exc
        except DuplicateWorkflowSourceError as exc:
            raise _error(status.HTTP_409_CONFLICT, "workflow_source_conflict", str(exc)) from exc
        except InvalidWorkflowSourcePathError as exc:
            raise _error(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "workflow_source_invalid_path",
                str(exc),
            ) from exc
        return _to_source_response(source)
    finally:
        conn.close()


@router.delete(
    "/sources/{source_id}",
    response_model=DeleteWorkflowSourceResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def delete_workflow_source(
    source_id: SourceId,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> DeleteWorkflowSourceResponse:
    repo, conn = _repo(resources)
    try:
        try:
            repo.delete(source_id)
        except WorkflowSourceNotFoundError as exc:
            raise _error(status.HTTP_404_NOT_FOUND, "workflow_source_not_found", str(exc)) from exc
        return DeleteWorkflowSourceResponse(deleted=True)
    finally:
        conn.close()


@router.post(
    "/reload",
    response_model=WorkflowReloadResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def reload_workflows(
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> WorkflowReloadResponse:
    return _reload_from_sqlite_sources(resources)


@router.get(
    "",
    response_model=WorkflowsListResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def list_workflows(
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> WorkflowsListResponse:
    workflows: list[WorkflowSummaryResponse] = []
    for name in resources.workflow_registry.list_names():
        metadata = resources.workflow_registry.get_workflow_metadata(name)
        source = resources.workflow_registry.get_workflow_source(name)
        workflows.append(
            WorkflowSummaryResponse(
                name=str(metadata.get("name", name)),
                description=str(metadata.get("description", "")),
                version=str(metadata.get("version", "")),
                tags=list(metadata.get("tags", [])),
                source_path=str(source) if source is not None else None,
            )
        )
    return WorkflowsListResponse(workflows=workflows)


@router.get(
    "/schema",
    response_model=dict[str, Any],
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def workflow_schema(
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> dict[str, Any]:
    return resources.executor_registry.generate_workflow_schema()


@router.get(
    "/{workflow_name}",
    response_model=dict[str, Any],
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def workflow_detail(
    workflow_name: WorkflowName,
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> dict[str, Any]:
    try:
        metadata = resources.workflow_registry.get_workflow_metadata(workflow_name, detailed=True)
    except KeyError as exc:
        raise _error(
            status.HTTP_404_NOT_FOUND,
            "workflow_not_found",
            f"Workflow not found: {workflow_name}",
        ) from exc
    source = resources.workflow_registry.get_workflow_source(workflow_name)
    metadata["source_path"] = str(source) if source is not None else None
    metadata.update(_workflow_yaml_detail(workflow_name, source))
    return metadata


@router.post(
    "/sources/{source_id}/validate",
    response_model=WorkflowValidateResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def validate_workflow_source(
    source_id: SourceId,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> WorkflowValidateResponse:
    repo, conn = _repo(resources)
    try:
        try:
            source = repo.get(source_id)
        except WorkflowSourceNotFoundError as exc:
            raise _error(status.HTTP_404_NOT_FOUND, "workflow_source_not_found", str(exc)) from exc

        temp_registry = WorkflowRegistry()
        try:
            summary = reload_registry_from_source_paths(temp_registry, [source.source_path])
        except WorkflowSourceReloadError as exc:
            raise _map_reload_error(exc) from exc

        return WorkflowValidateResponse(
            valid=True,
            workflow_names=summary.workflow_names,
            total=summary.workflow_count,
        )
    finally:
        conn.close()
