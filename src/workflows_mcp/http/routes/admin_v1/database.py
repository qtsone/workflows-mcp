from __future__ import annotations

from sqlite3 import Connection
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, StringConstraints

from workflows_mcp.http.dependencies import (
    CurrentAdminSession,
    get_resources,
    require_admin_csrf,
    require_current_admin_session,
)
from workflows_mcp.http.lifespan import AppResources
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.repos.postgres_repo import (
    PostgresSettingsMetadata,
    SQLitePostgresSettingsRepository,
)
from workflows_mcp.postgres_probe import PostgresProbe
from workflows_mcp.security.crypto import SecretCryptoError, SecretKeyError

router = APIRouter(prefix="/database")

_PGVECTOR_IMAGE = "pgvector/pgvector:pg17"


class DatabaseSetupResponse(BaseModel):
    image: str
    docker: str
    podman: str
    notes: list[str]


class SavePostgresSettingsRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "enabled": True,
                    "dsn": "postgresql://wf_user:password@127.0.0.1:5432/workflows",
                }
            ]
        }
    )

    enabled: bool = True
    dsn: Annotated[str, StringConstraints(min_length=1, max_length=65536)] | None = None


class DatabaseSettingsResponse(BaseModel):
    enabled: bool
    configured: bool
    updated_at: str


class DatabaseConnectionTestResponse(BaseModel):
    ok: bool
    status: str
    configured: bool
    blockers: list[str]
    actionable: list[str]


def _repo(resources: AppResources) -> tuple[SQLitePostgresSettingsRepository, Connection]:
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    return (
        SQLitePostgresSettingsRepository(
            conn=conn,
            key_path=resources.metadata.base_dir / "secrets.key",
        ),
        conn,
    )


def _settings_response(metadata: PostgresSettingsMetadata) -> DatabaseSettingsResponse:
    return DatabaseSettingsResponse(
        enabled=metadata.enabled,
        configured=metadata.configured,
        updated_at=metadata.updated_at,
    )


def _actionable_for_blockers(blockers: list[str]) -> list[str]:
    steps: list[str] = []
    mapping = {
        "postgresql_dsn_missing": "Save PostgreSQL DSN in /api/admin/v1/database/settings.",
        "postgresql_connectivity": "Verify host, port, credentials, and network reachability.",
        "postgresql_version_unsupported": "Use PostgreSQL 14+ (recommended image uses pgvector).",
        "pgvector_missing": "Install/enable pgvector extension in your database.",
        "postgresql_readwrite_failed": "Ensure DB user can create temp tables and write/read data.",
    }
    for blocker in blockers:
        if blocker in mapping:
            steps.append(mapping[blocker])
    if not steps:
        steps.append("Review PostgreSQL settings and retry the connection test.")
    return steps


@router.get(
    "/setup",
    response_model=DatabaseSetupResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def get_database_setup_guidance(
    _current: CurrentAdminSession = Depends(require_current_admin_session),
) -> DatabaseSetupResponse:
    return DatabaseSetupResponse(
        image=_PGVECTOR_IMAGE,
        docker=(
            "docker run --name workflows-postgres -e POSTGRES_DB=workflows "
            "-e POSTGRES_USER=workflows -e POSTGRES_PASSWORD=<set-password> "
            f"-p 5432:5432 -d {_PGVECTOR_IMAGE}"
        ),
        podman=(
            "podman run --name workflows-postgres -e POSTGRES_DB=workflows "
            "-e POSTGRES_USER=workflows -e POSTGRES_PASSWORD=<set-password> "
            f"-p 5432:5432 -d {_PGVECTOR_IMAGE}"
        ),
        notes=[
            "Use a unique password and store it in your password manager.",
            "Do not paste production DSNs or plaintext credentials into shared logs.",
        ],
    )


@router.get(
    "/settings",
    response_model=DatabaseSettingsResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def get_postgres_settings(
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> DatabaseSettingsResponse:
    repo, conn = _repo(resources)
    try:
        return _settings_response(repo.load_settings())
    finally:
        conn.close()


@router.put(
    "/settings",
    response_model=DatabaseSettingsResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def put_postgres_settings(
    body: SavePostgresSettingsRequest,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> DatabaseSettingsResponse:
    normalized_dsn = body.dsn.strip() if isinstance(body.dsn, str) else None
    if body.enabled and not normalized_dsn:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="enabled PostgreSQL settings require a non-blank dsn",
        )

    repo, conn = _repo(resources)
    try:
        saved = repo.save_settings(enabled=body.enabled, dsn=normalized_dsn)
        return _settings_response(saved)
    finally:
        conn.close()


@router.post(
    "/connection-test",
    response_model=DatabaseConnectionTestResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def test_postgres_connection(
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> DatabaseConnectionTestResponse:
    repo, conn = _repo(resources)
    try:
        settings = repo.load_settings()
        if not settings.configured:
            blockers = ["postgresql_dsn_missing"]
            return DatabaseConnectionTestResponse(
                ok=False,
                status="degraded",
                configured=False,
                blockers=blockers,
                actionable=_actionable_for_blockers(blockers),
            )

        try:
            dsn = repo.load_dsn()
        except (SecretKeyError, SecretCryptoError):
            blockers = ["postgresql_dsn_missing"]
            return DatabaseConnectionTestResponse(
                ok=False,
                status="degraded",
                configured=False,
                blockers=blockers,
                actionable=_actionable_for_blockers(blockers),
            )

        if not dsn:
            blockers = ["postgresql_dsn_missing"]
            return DatabaseConnectionTestResponse(
                ok=False,
                status="degraded",
                configured=False,
                blockers=blockers,
                actionable=_actionable_for_blockers(blockers),
            )

        probe = PostgresProbe(dsn=dsn, require_pgvector=True)
        ok, blockers = await probe.check()
        if not ok:
            return DatabaseConnectionTestResponse(
                ok=False,
                status="degraded",
                configured=True,
                blockers=blockers,
                actionable=_actionable_for_blockers(blockers),
            )

        return DatabaseConnectionTestResponse(
            ok=True,
            status="ready",
            configured=True,
            blockers=[],
            actionable=[],
        )
    finally:
        conn.close()
