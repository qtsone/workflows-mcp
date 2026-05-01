from __future__ import annotations

from sqlite3 import Connection
from typing import Annotated, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel, ConfigDict, StringConstraints

from workflows_mcp.http.dependencies import (
    CurrentAdminSession,
    get_resources,
    require_admin_csrf,
    require_current_admin_session,
)
from workflows_mcp.http.lifespan import AppResources
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.repos.projects_repo import (
    DuplicateProjectPalaceError,
    DuplicateProjectSlugError,
    ProjectCreate,
    ProjectNotFoundError,
    ProjectPalaceImmutableError,
    ProjectRecord,
    ProjectUpdate,
    SQLiteProjectsRepository,
)

router = APIRouter(prefix="/projects")

ProjectId = Annotated[str, Path(min_length=1, max_length=128)]
NonEmptyString = Annotated[str, StringConstraints(min_length=1, max_length=1024)]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ProjectResponse(BaseModel):
    id: str
    name: str
    slug: str
    palace: str
    default_wing: str
    default_room: str
    fs_root: str
    fs_allowlist: list[str]
    created_at: str
    updated_at: str


class ProjectsListResponse(BaseModel):
    projects: list[ProjectResponse]


class CreateProjectRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "Workflow Service",
                    "slug": "workflow-service",
                    "palace": "wf-palace",
                    "default_wing": "platform",
                    "default_room": "runtime",
                    "fs_root": "/workspace/workflows",
                    "fs_allowlist": ["/workspace/workflows", "/workspace/shared"],
                }
            ]
        }
    )

    name: NonEmptyString
    slug: NonEmptyString
    palace: NonEmptyString
    default_wing: NonEmptyString
    default_room: NonEmptyString
    fs_root: NonEmptyString
    fs_allowlist: list[NonEmptyString] | None = None


class UpdateProjectRequest(BaseModel):
    name: NonEmptyString | None = None
    slug: NonEmptyString | None = None
    palace: NonEmptyString | None = None
    default_wing: NonEmptyString | None = None
    default_room: NonEmptyString | None = None
    fs_root: NonEmptyString | None = None
    fs_allowlist: list[NonEmptyString] | None = None


class DeleteProjectResponse(BaseModel):
    deleted: bool


def _repo(resources: AppResources) -> tuple[SQLiteProjectsRepository, Connection]:
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    return SQLiteProjectsRepository(conn), conn


def _to_response(project: ProjectRecord) -> ProjectResponse:
    return ProjectResponse(
        id=project.id,
        name=project.name,
        slug=project.slug,
        palace=project.palace,
        default_wing=project.default_wing,
        default_room=project.default_room,
        fs_root=project.fs_root,
        fs_allowlist=project.fs_allowlist,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


def _raise_conflict(code: str, message: str) -> NoReturn:
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=ErrorDetail(code=code, message=message).model_dump(),
    )


def _raise_not_found(project_id: str) -> NoReturn:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=ErrorDetail(
            code="project_not_found",
            message=f"Project not found: {project_id}",
        ).model_dump(),
    )


@router.get(
    "",
    response_model=ProjectsListResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def list_projects(
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> ProjectsListResponse:
    repo, conn = _repo(resources)
    try:
        projects = repo.list_all()
        return ProjectsListResponse(projects=[_to_response(project) for project in projects])
    finally:
        conn.close()


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=ProjectResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def create_project(
    body: CreateProjectRequest,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> ProjectResponse:
    repo, conn = _repo(resources)
    try:
        try:
            created = repo.create(
                ProjectCreate(
                    name=body.name,
                    slug=body.slug,
                    palace=body.palace,
                    default_wing=body.default_wing,
                    default_room=body.default_room,
                    fs_root=body.fs_root,
                    fs_allowlist=list(body.fs_allowlist) if body.fs_allowlist is not None else None,
                )
            )
        except DuplicateProjectSlugError:
            _raise_conflict("project_slug_conflict", "Project slug already exists")
        except DuplicateProjectPalaceError:
            _raise_conflict("project_palace_conflict", "Project palace already exists")

        try:
            resources.watcher_manager.enable_project_by_default(created.id)
        except Exception:
            repo.delete(created.id)
            raise

        return _to_response(created)
    finally:
        conn.close()


@router.get(
    "/{project_id}",
    response_model=ProjectResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def get_project(
    project_id: ProjectId,
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> ProjectResponse:
    repo, conn = _repo(resources)
    try:
        project = repo.get_by_id(project_id)
        if project is None:
            _raise_not_found(project_id)
        return _to_response(project)
    finally:
        conn.close()


@router.patch(
    "/{project_id}",
    response_model=ProjectResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def patch_project(
    project_id: ProjectId,
    body: UpdateProjectRequest,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> ProjectResponse:
    repo, conn = _repo(resources)
    try:
        current = repo.get_by_id(project_id)
        if current is None:
            _raise_not_found(project_id)
        if body.palace is not None and body.palace != current.palace:
            _raise_conflict("project_palace_immutable", "Project palace is immutable")

        try:
            updated = repo.update(
                project_id,
                ProjectUpdate(
                    name=body.name,
                    slug=body.slug,
                    palace=body.palace,
                    default_wing=body.default_wing,
                    default_room=body.default_room,
                    fs_root=body.fs_root,
                    fs_allowlist=list(body.fs_allowlist) if body.fs_allowlist is not None else None,
                ),
            )
        except ProjectNotFoundError:
            _raise_not_found(project_id)
        except DuplicateProjectSlugError:
            _raise_conflict("project_slug_conflict", "Project slug already exists")
        except DuplicateProjectPalaceError:
            _raise_conflict("project_palace_immutable", "Project palace is immutable")
        except ProjectPalaceImmutableError:
            _raise_conflict("project_palace_immutable", "Project palace is immutable")
        return _to_response(updated)
    finally:
        conn.close()


@router.delete(
    "/{project_id}",
    response_model=DeleteProjectResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def delete_project(
    project_id: ProjectId,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> DeleteProjectResponse:
    repo, conn = _repo(resources)
    try:
        deleted = repo.delete(project_id)
        if not deleted:
            _raise_not_found(project_id)
        return DeleteProjectResponse(deleted=True)
    finally:
        conn.close()
