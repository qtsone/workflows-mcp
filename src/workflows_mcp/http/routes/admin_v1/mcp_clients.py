from __future__ import annotations

from sqlite3 import Connection
from typing import Annotated, Any

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
from workflows_mcp.metadata.repos.tokens_repo import (
    DuplicateTokenLabelError,
    InvalidTokenProjectBindingError,
    SQLiteTokensRepository,
    TokenIntegrityError,
    TokenRecord,
    UnknownTokenError,
)

router = APIRouter(prefix="/mcp-clients")

TokenId = Annotated[str, Path(min_length=1, max_length=128)]
NonEmptyString = Annotated[str, StringConstraints(min_length=1, max_length=1024)]


class ErrorDetail(BaseModel):
    code: str
    message: str


class MCPClientResponse(BaseModel):
    id: str
    label: str
    capabilities: dict[str, Any]
    project_ids: list[str]
    created_at: str
    last_used_at: str | None
    revoked_at: str | None


class MCPClientCreateResponse(MCPClientResponse):
    token: str
    config_snippet: str


class MCPClientsListResponse(BaseModel):
    mcp_clients: list[MCPClientResponse]


class MCPClientCreateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "label": "ci-agent",
                    "project_ids": ["project-id"],
                    "capabilities": {"scopes": ["read:workflows"]},
                }
            ]
        }
    )

    label: NonEmptyString
    project_ids: list[NonEmptyString]
    capabilities: dict[str, Any] | None = None


class MCPClientRevokeResponse(BaseModel):
    revoked: bool


def _repo(resources: AppResources) -> tuple[SQLiteTokensRepository, Connection]:
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    return SQLiteTokensRepository(conn), conn


def _to_response(record: TokenRecord) -> MCPClientResponse:
    return MCPClientResponse(
        id=record.id,
        label=record.label,
        capabilities=record.capabilities,
        project_ids=record.project_ids,
        created_at=record.created_at,
        last_used_at=record.last_used_at,
        revoked_at=record.revoked_at,
    )


def _raise_http(status_code: int, code: str, message: str) -> None:
    raise HTTPException(
        status_code=status_code,
        detail=ErrorDetail(code=code, message=message).model_dump(),
    )


def _config_snippet(token: str) -> str:
    return (
        "{\n"
        '  "transport": "streamable-http",\n'
        '  "url": "https://<your-workflows-host>/mcp",\n'
        '  "headers": {\n'
        '    "Authorization": "Bearer '
        + token
        + '"\n'
        "  }\n"
        "}"
    )


@router.get(
    "",
    response_model=MCPClientsListResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def list_mcp_clients(
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> MCPClientsListResponse:
    repo, conn = _repo(resources)
    try:
        tokens = repo.list_tokens()
        return MCPClientsListResponse(mcp_clients=[_to_response(token) for token in tokens])
    finally:
        conn.close()


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=MCPClientCreateResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def create_mcp_client(
    body: MCPClientCreateRequest,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> MCPClientCreateResponse:
    repo, conn = _repo(resources)
    try:
        try:
            created = repo.create(
                label=body.label,
                project_ids=list(body.project_ids),
                capabilities=body.capabilities,
            )
        except InvalidTokenProjectBindingError:
            _raise_http(
                status.HTTP_400_BAD_REQUEST,
                "invalid_project_binding",
                "One or more project ids are invalid",
            )
        except DuplicateTokenLabelError:
            _raise_http(
                status.HTTP_409_CONFLICT,
                "token_label_conflict",
                "Token label already exists",
            )
        except TokenIntegrityError:
            _raise_http(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "token_integrity_error",
                "Token operation failed",
            )
        return MCPClientCreateResponse(
            id=created.id,
            label=created.label,
            token=created.token_secret,
            config_snippet=_config_snippet(created.token_secret),
            capabilities=created.capabilities,
            project_ids=created.project_ids,
            created_at=created.created_at,
            last_used_at=created.last_used_at,
            revoked_at=created.revoked_at,
        )
    finally:
        conn.close()


@router.delete(
    "/{token_id}",
    response_model=MCPClientRevokeResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def revoke_mcp_client(
    token_id: TokenId,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> MCPClientRevokeResponse:
    repo, conn = _repo(resources)
    try:
        try:
            repo.revoke(token_id)
        except UnknownTokenError:
            _raise_http(
                status.HTTP_404_NOT_FOUND,
                "token_not_found",
                "Token not found",
            )
        except TokenIntegrityError:
            _raise_http(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "token_integrity_error",
                "Token operation failed",
            )
        return MCPClientRevokeResponse(revoked=True)
    finally:
        conn.close()


@router.post(
    "/{token_id}/regenerate",
    response_model=MCPClientCreateResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def regenerate_mcp_client(
    token_id: TokenId,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> MCPClientCreateResponse:
    repo, conn = _repo(resources)
    try:
        try:
            regenerated = repo.regenerate(token_id)
        except UnknownTokenError:
            _raise_http(
                status.HTTP_404_NOT_FOUND,
                "token_not_found",
                "Token not found",
            )
        except TokenIntegrityError:
            _raise_http(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "token_integrity_error",
                "Token operation failed",
            )
        return MCPClientCreateResponse(
            id=regenerated.id,
            label=regenerated.label,
            token=regenerated.token_secret,
            config_snippet=_config_snippet(regenerated.token_secret),
            capabilities=regenerated.capabilities,
            project_ids=regenerated.project_ids,
            created_at=regenerated.created_at,
            last_used_at=regenerated.last_used_at,
            revoked_at=regenerated.revoked_at,
        )
    finally:
        conn.close()
