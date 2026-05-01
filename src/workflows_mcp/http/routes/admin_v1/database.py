from __future__ import annotations

import shlex
from sqlite3 import Connection
from typing import Annotated, Literal
from urllib.parse import parse_qsl, urlencode, urlparse

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from workflows_mcp.http.dependencies import (
    CurrentAdminSession,
    get_resources,
    require_admin_csrf,
    require_current_admin_session,
)
from workflows_mcp.http.lifespan import AppResources
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.repos.postgres_repo import (
    PostgresProfileInput,
    PostgresSettingsMetadata,
    SQLitePostgresSettingsRepository,
)
from workflows_mcp.postgres_probe import PostgresProbe
from workflows_mcp.security.crypto import SecretCryptoError, SecretKeyError

router = APIRouter(prefix="/database")

_PGVECTOR_IMAGE = "pgvector/pgvector:pg17"
_NAME_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$"


class DatabaseSetupResponse(BaseModel):
    image: str
    docker: str
    podman: str
    notes: list[str]


class SavePostgresSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    host: Annotated[str, StringConstraints(min_length=1, max_length=253)] = "127.0.0.1"
    port: int = Field(default=5432, ge=1, le=65535)
    database: Annotated[str, StringConstraints(min_length=1, max_length=63)] = "workflows"
    username: Annotated[str, StringConstraints(min_length=1, max_length=63)] = "workflows"
    password: Annotated[str, StringConstraints(max_length=1024)] | None = None
    password_clear: bool = False
    ssl_mode: Literal["disable", "prefer", "require", "verify-ca", "verify-full"] = "disable"
    extra_params: Annotated[str, StringConstraints(max_length=2048)] = ""
    container_name: Annotated[str, StringConstraints(pattern=_NAME_PATTERN)] = "workflows-postgres"
    container_image: Annotated[
        str,
        StringConstraints(min_length=1, max_length=255),
    ] = _PGVECTOR_IMAGE
    container_host_port: int = Field(default=5432, ge=1, le=65535)
    volume_name: Annotated[
        str,
        StringConstraints(pattern=_NAME_PATTERN),
    ] = "workflows-postgres-data"
    dsn_import: str | None = None

    @field_validator("extra_params")
    @classmethod
    def _validate_extra_params(cls, value: str) -> str:
        stripped = value.strip()
        if stripped.startswith("?"):
            raise ValueError("extra_params must not start with '?'")
        parse_qsl(stripped, keep_blank_values=True)
        return stripped

    @field_validator(
        "host",
        "database",
        "username",
        "container_name",
        "container_image",
        "volume_name",
        "extra_params",
        mode="after",
    )
    @classmethod
    def _reject_control_chars(cls, value: str) -> str:
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
            raise ValueError("control characters are not allowed")
        return value

    @model_validator(mode="after")
    def _validate_enabled_fields(self) -> SavePostgresSettingsRequest:
        if self.enabled:
            if not self.host.strip() or not self.database.strip() or not self.username.strip():
                raise ValueError("enabled settings require host, database, and username")
        return self


class DatabaseSettingsResponse(BaseModel):
    enabled: bool
    configured: bool
    password_configured: bool
    legacy_profile_reentry_required: bool
    host: str
    port: int
    database: str
    username: str
    ssl_mode: str
    extra_params: str
    container_name: str
    container_image: str
    container_host_port: int
    volume_name: str
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
        password_configured=metadata.password_configured,
        legacy_profile_reentry_required=metadata.legacy_profile_reentry_required,
        host=metadata.host,
        port=metadata.port,
        database=metadata.database,
        username=metadata.username,
        ssl_mode=metadata.ssl_mode,
        extra_params=metadata.extra_params,
        container_name=metadata.container_name,
        container_image=metadata.container_image,
        container_host_port=metadata.container_host_port,
        volume_name=metadata.volume_name,
        updated_at=metadata.updated_at,
    )


def _actionable_for_blockers(blockers: list[str]) -> list[str]:
    steps: list[str] = []
    mapping = {
        "postgresql_dsn_missing": "Save PostgreSQL settings in /api/admin/v1/database/settings.",
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


def _parse_dsn_import(dsn: str) -> dict[str, str | int | None]:
    parsed = urlparse(dsn)
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise ValueError("dsn_import must use postgres/postgresql scheme")
    if parsed.hostname is None or parsed.port is None:
        raise ValueError("dsn_import must include host and port")
    if parsed.username is None:
        raise ValueError("dsn_import must include username")
    database = parsed.path.lstrip("/")
    if not database:
        raise ValueError("dsn_import must include database")

    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    ssl_mode = "disable"
    extra: list[tuple[str, str]] = []
    for key, value in query_pairs:
        if key == "sslmode":
            ssl_mode = value
        else:
            extra.append((key, value))
    return {
        "host": parsed.hostname,
        "port": parsed.port,
        "database": database,
        "username": parsed.username,
        "password": parsed.password,
        "ssl_mode": ssl_mode,
        "extra_params": urlencode(extra, doseq=True),
    }


def _quote(value: str) -> str:
    return shlex.quote(value)


def _setup_command(engine: str, metadata: PostgresSettingsMetadata) -> str:
    password_value = "<configured-password>" if metadata.password_configured else "<password>"
    parts = [
        engine,
        "run",
        "--name",
        _quote(metadata.container_name),
        "-e",
        _quote(f"POSTGRES_DB={metadata.database}"),
        "-e",
        _quote(f"POSTGRES_USER={metadata.username}"),
        "-e",
        _quote(f"POSTGRES_PASSWORD={password_value}"),
        "-p",
        _quote(f"{metadata.container_host_port}:5432"),
        "-v",
        _quote(f"{metadata.volume_name}:/var/lib/postgresql/data"),
        "-d",
        _quote(metadata.container_image),
    ]
    return " ".join(parts)


@router.get(
    "/setup",
    response_model=DatabaseSetupResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def get_database_setup_guidance(
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> DatabaseSetupResponse:
    repo, conn = _repo(resources)
    try:
        metadata = repo.load_settings()
    finally:
        conn.close()
    return DatabaseSetupResponse(
        image=metadata.container_image,
        docker=_setup_command("docker", metadata),
        podman=_setup_command("podman", metadata),
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
    input_host = body.host
    input_port = body.port
    input_database = body.database
    input_username = body.username
    input_password = body.password
    input_ssl_mode: str = body.ssl_mode
    input_extra_params = body.extra_params
    imported_password_was_provided = False

    if body.dsn_import is not None:
        try:
            imported = _parse_dsn_import(body.dsn_import)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        input_host = str(imported["host"])
        imported_port = imported["port"]
        if not isinstance(imported_port, int):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="dsn_import must include numeric port",
            )
        input_port = imported_port
        input_database = str(imported["database"])
        input_username = str(imported["username"])
        imported_password = imported["password"]
        if isinstance(imported_password, str):
            input_password = imported_password
            imported_password_was_provided = True
        parsed_ssl_mode = str(imported["ssl_mode"])
        if parsed_ssl_mode in {"disable", "prefer", "require", "verify-ca", "verify-full"}:
            input_ssl_mode = parsed_ssl_mode
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="dsn_import sslmode is invalid",
            )
        input_extra_params = str(imported["extra_params"])

    password_was_provided = (
        (("password" in body.model_fields_set) and input_password is not None)
        or imported_password_was_provided
    )

    repo, conn = _repo(resources)
    try:
        saved = repo.save_settings(
            PostgresProfileInput(
                enabled=body.enabled,
                host=input_host,
                port=input_port,
                database=input_database,
                username=input_username,
                password=input_password,
                password_was_provided=password_was_provided,
                password_clear=body.password_clear,
                ssl_mode=input_ssl_mode,
                extra_params=input_extra_params,
                container_name=body.container_name,
                container_image=body.container_image,
                container_host_port=body.container_host_port,
                volume_name=body.volume_name,
                dsn_import=body.dsn_import,
            )
        )
        return _settings_response(saved)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
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
        try:
            dsn = repo.load_dsn()
        except (SecretKeyError, SecretCryptoError):
            dsn = None
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
