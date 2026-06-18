"""Runtime wiring for the PostgreSQL-backed memory database."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlparse

from workflows_mcp.context import MemoryBackendUnavailableError
from workflows_mcp.engine.sql import ConnectionConfig, DatabaseEngine
from workflows_mcp.engine.sql.postgres_backend import PostgresBackend
from workflows_mcp.memory.knowledge.schema import ensure_schema
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.repos.postgres_repo import SQLitePostgresSettingsRepository


def memory_connection_config_from_dsn(dsn: str) -> ConnectionConfig:
    parsed = urlparse(dsn)
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise ValueError("configured PostgreSQL DSN is invalid")
    if parsed.hostname is None or not parsed.path.strip("/"):
        raise ValueError("configured PostgreSQL DSN is incomplete")

    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    ssl_mode = query.get("sslmode", "disable")
    return ConnectionConfig(
        dialect=DatabaseEngine.POSTGRESQL,
        host=parsed.hostname,
        port=parsed.port or 5432,
        database=unquote(parsed.path.strip("/")),
        username=unquote(parsed.username) if parsed.username else None,
        password=unquote(parsed.password) if parsed.password else None,
        ssl=False if ssl_mode == "disable" else ssl_mode,
    )


def memory_connection_config_from_metadata(app_ctx: Any) -> ConnectionConfig | None:
    metadata_db_path = getattr(app_ctx, "metadata_db_path", None)
    metadata_base_dir = getattr(app_ctx, "metadata_base_dir", None)
    if not isinstance(metadata_db_path, (str, Path)) or not isinstance(
        metadata_base_dir, (str, Path)
    ):
        return None

    try:
        conn = connect_metadata_db(Path(metadata_db_path))
    except Exception:
        return None
    try:
        repo = SQLitePostgresSettingsRepository(
            conn=conn,
            key_path=Path(metadata_base_dir) / "secrets.key",
        )
        try:
            dsn = repo.load_dsn()
        except Exception:
            dsn = None
    finally:
        conn.close()
    if not dsn:
        return None
    return memory_connection_config_from_dsn(dsn)


def memory_connection_config_from_env() -> ConnectionConfig | None:
    raw_host = os.environ.get("MEMORY_DB_HOST")
    if not raw_host:
        return None
    return ConnectionConfig(
        dialect=DatabaseEngine.POSTGRESQL,
        host=raw_host,
        port=int(os.environ.get("MEMORY_DB_PORT", "5432")),
        database=os.environ.get("MEMORY_DB_NAME", "memory_db"),
        username=os.environ.get("MEMORY_DB_USER"),
        password=os.environ.get("MEMORY_DB_PASSWORD"),
    )


def resolve_memory_connection_config(
    app_ctx: Any,
    *,
    prefer_metadata: bool,
) -> ConnectionConfig | None:
    metadata_config = memory_connection_config_from_metadata(app_ctx)
    env_config = memory_connection_config_from_env()
    if prefer_metadata:
        return metadata_config or env_config
    return env_config or metadata_config


async def connect_memory_backend(config: ConnectionConfig) -> PostgresBackend:
    backend = PostgresBackend()
    connected = False
    try:
        await backend.connect(config)
        connected = True
        await ensure_schema(backend)
    except Exception:
        if connected:
            await backend.disconnect()
        raise
    return backend


def register_memory_executors(executor_registry: Any) -> None:
    """Register memory-gated block executors on a registry, idempotently.

    These executors operate directly against the knowledge tables and only make
    sense once a memory backend is connected. They are deliberately kept out of
    the default registry (general-purpose blocks only) and registered here at the
    same seam that connects the backend.
    """

    from workflows_mcp.memory.executors_memory import MemoryExecutor
    from workflows_mcp.memory.executors_system2_planner import System2PlannerExecutor

    if not executor_registry.has("Memory"):
        executor_registry.register(MemoryExecutor())
    if not executor_registry.has("System2Planner"):
        executor_registry.register(System2PlannerExecutor())


def _memory_backend_unavailable(message: str) -> MemoryBackendUnavailableError:
    return MemoryBackendUnavailableError(
        code="MEMORY_BACKEND_UNAVAILABLE",
        message=message,
        retryable=False,
        actionable_fix=(
            "Configure PostgreSQL/pgvector via /api/admin/v1/database/settings and "
            "/api/admin/v1/database/connection-test, then verify /ready reports healthy "
            "before retrying."
        ),
    )


async def refresh_memory_backend(
    *,
    app_ctx: Any,
    executor_registry: Any,
    prefer_metadata: bool = True,
) -> None:
    """Reconnect the shared memory backend from the active runtime profile."""

    config = resolve_memory_connection_config(app_ctx, prefer_metadata=prefer_metadata)
    old_backend = getattr(app_ctx, "memory_backend", None)
    app_ctx.memory_backend = None
    app_ctx.memory_backend_lock = None
    if old_backend is not None:
        await old_backend.disconnect()

    if config is None:
        app_ctx.memory_backend_unavailable_error = _memory_backend_unavailable(
            "Memory backend is unavailable because no PostgreSQL memory database is configured."
        )
        return

    try:
        new_backend = await connect_memory_backend(config)
    except Exception:
        app_ctx.memory_backend_unavailable_error = _memory_backend_unavailable(
            "Memory backend is unavailable because the configured PostgreSQL memory "
            "database is not reachable."
        )
        raise
    app_ctx.memory_backend = new_backend
    app_ctx.memory_backend_lock = asyncio.Lock()
    app_ctx.memory_backend_unavailable_error = None

    register_memory_executors(executor_registry)
