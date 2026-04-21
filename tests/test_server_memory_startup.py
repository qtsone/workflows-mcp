from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from workflows_mcp import server


class _FakeBackend:
    def __init__(self, connect_side_effects: list[BaseException | None] | None = None) -> None:
        self._connect_side_effects = connect_side_effects or []
        self.connect_calls = 0
        self.disconnect_calls = 0

    async def connect(self, _config: Any) -> None:
        effect = None
        if self.connect_calls < len(self._connect_side_effects):
            effect = self._connect_side_effects[self.connect_calls]

        self.connect_calls += 1
        if effect is not None:
            raise effect

    async def disconnect(self) -> None:
        self.disconnect_calls += 1


class _FakeAdminConn:
    def __init__(self, execute_error: BaseException | None = None) -> None:
        self.execute_error = execute_error
        self.closed = False
        self.executed_sql: str | None = None

    async def execute(self, sql: str) -> None:
        self.executed_sql = sql
        if self.execute_error is not None:
            raise self.execute_error

    async def close(self) -> None:
        self.closed = True


class _FakeLifespanBackend:
    def __init__(self) -> None:
        self.disconnect_calls = 0

    async def disconnect(self) -> None:
        self.disconnect_calls += 1


@pytest.mark.asyncio
async def test_prepare_memory_schema_connects_and_initializes_without_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()

    class _FakePostgresBackend:
        def __new__(cls) -> _FakeBackend:
            return backend

    ensure_schema_calls = {"count": 0}

    async def _fake_ensure_schema(_backend: Any) -> None:
        ensure_schema_calls["count"] += 1

    import workflows_mcp.engine.knowledge.schema as schema_mod
    import workflows_mcp.engine.sql.postgres_backend as postgres_backend_mod

    monkeypatch.setattr(schema_mod, "ensure_schema", _fake_ensure_schema)
    monkeypatch.setattr(postgres_backend_mod, "PostgresBackend", _FakePostgresBackend)

    backend_result = await server._prepare_memory_schema("localhost")

    assert backend_result is backend
    assert backend.connect_calls == 1
    assert ensure_schema_calls["count"] == 1
    assert backend.disconnect_calls == 0


@pytest.mark.asyncio
async def test_app_lifespan_reuses_single_memory_backend_and_disconnects_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backend = _FakeLifespanBackend()
    prepare_calls = {"count": 0}
    register_calls = {"count": 0}

    async def _fake_prepare_memory_schema(_memory_db_host: str) -> _FakeLifespanBackend:
        prepare_calls["count"] += 1
        return fake_backend

    def _fake_register_memory_tools(_mcp_server: Any) -> None:
        register_calls["count"] += 1

    class _FakeMemoryExecutor:
        type_name = "Memory"

    import workflows_mcp.engine.executors_memory as executors_memory_mod
    import workflows_mcp.tools_memory as tools_memory_mod

    monkeypatch.setattr(server, "_prepare_memory_schema", _fake_prepare_memory_schema)
    monkeypatch.setattr(server, "load_workflows", lambda _registry: None)
    monkeypatch.setattr(executors_memory_mod, "MemoryExecutor", _FakeMemoryExecutor)
    monkeypatch.setattr(tools_memory_mod, "register_memory_tools", _fake_register_memory_tools)
    monkeypatch.setenv("MEMORY_DB_HOST", "localhost")
    monkeypatch.setenv("WORKFLOWS_IO_QUEUE_ENABLED", "false")
    monkeypatch.setenv("WORKFLOWS_JOB_QUEUE_ENABLED", "false")

    async with server.app_lifespan(server.mcp) as app_context:
        assert prepare_calls["count"] == 1
        assert register_calls["count"] == 1
        assert app_context.memory_backend is fake_backend
        assert fake_backend.disconnect_calls == 0

    assert fake_backend.disconnect_calls == 1


@pytest.mark.asyncio
async def test_prepare_memory_schema_bootstraps_missing_db_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class InvalidCatalogNameError(Exception):
        pass

    first_error = InvalidCatalogNameError("database does not exist")
    backend = _FakeBackend(connect_side_effects=[first_error, None])
    admin_conn = _FakeAdminConn()

    class _FakePostgresBackend:
        def __new__(cls) -> _FakeBackend:
            return backend

    ensure_schema_calls = {"count": 0}

    async def _fake_ensure_schema(_backend: Any) -> None:
        ensure_schema_calls["count"] += 1

    async def _fake_admin_connect(**_kwargs: Any) -> _FakeAdminConn:
        return admin_conn

    fake_asyncpg = SimpleNamespace(
        InvalidCatalogNameError=InvalidCatalogNameError,
        connect=_fake_admin_connect,
    )

    import workflows_mcp.engine.knowledge.schema as schema_mod
    import workflows_mcp.engine.sql.postgres_backend as postgres_backend_mod

    monkeypatch.setattr(schema_mod, "ensure_schema", _fake_ensure_schema)
    monkeypatch.setattr(postgres_backend_mod, "PostgresBackend", _FakePostgresBackend)
    monkeypatch.setitem(sys.modules, "asyncpg", fake_asyncpg)
    monkeypatch.setenv("MEMORY_DB_AUTO_CREATE", "true")
    monkeypatch.setenv("MEMORY_DB_ADMIN_DATABASE", "postgres")
    monkeypatch.setenv("MEMORY_DB_NAME", "memory_db")

    backend_result = await server._prepare_memory_schema("localhost")

    assert backend_result is backend
    assert backend.connect_calls == 2
    assert ensure_schema_calls["count"] == 1
    assert backend.disconnect_calls == 0
    assert admin_conn.executed_sql == 'CREATE DATABASE "memory_db"'
    assert "Memory DB is missing" in caplog.text
    assert "Memory DB bootstrap succeeded" in caplog.text


@pytest.mark.asyncio
async def test_prepare_memory_schema_skips_bootstrap_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class InvalidCatalogNameError(Exception):
        pass

    first_error = InvalidCatalogNameError("database does not exist")
    backend = _FakeBackend(connect_side_effects=[first_error])

    class _FakePostgresBackend:
        def __new__(cls) -> _FakeBackend:
            return backend

    async def _fake_ensure_schema(_backend: Any) -> None:
        raise AssertionError("ensure_schema should not be called when connect fails")

    async def _fake_admin_connect(**_kwargs: Any) -> _FakeAdminConn:
        raise AssertionError("admin connect should not be called when bootstrap is disabled")

    fake_asyncpg = SimpleNamespace(
        InvalidCatalogNameError=InvalidCatalogNameError,
        connect=_fake_admin_connect,
    )

    import workflows_mcp.engine.knowledge.schema as schema_mod
    import workflows_mcp.engine.sql.postgres_backend as postgres_backend_mod

    monkeypatch.setattr(schema_mod, "ensure_schema", _fake_ensure_schema)
    monkeypatch.setattr(postgres_backend_mod, "PostgresBackend", _FakePostgresBackend)
    monkeypatch.setitem(sys.modules, "asyncpg", fake_asyncpg)
    monkeypatch.setenv("MEMORY_DB_AUTO_CREATE", "false")

    with pytest.raises(InvalidCatalogNameError):
        await server._prepare_memory_schema("localhost")

    assert "Memory DB bootstrap skipped (MEMORY_DB_AUTO_CREATE=false)" in caplog.text


@pytest.mark.asyncio
async def test_prepare_memory_schema_tolerates_duplicate_database_race(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class InvalidCatalogNameError(Exception):
        pass

    class DuplicateDatabaseError(Exception):
        pass

    first_error = InvalidCatalogNameError("database does not exist")
    backend = _FakeBackend(connect_side_effects=[first_error, None])
    admin_conn = _FakeAdminConn(execute_error=DuplicateDatabaseError("already exists"))

    class _FakePostgresBackend:
        def __new__(cls) -> _FakeBackend:
            return backend

    ensure_schema_calls = {"count": 0}

    async def _fake_ensure_schema(_backend: Any) -> None:
        ensure_schema_calls["count"] += 1

    async def _fake_admin_connect(**_kwargs: Any) -> _FakeAdminConn:
        return admin_conn

    fake_asyncpg = SimpleNamespace(
        InvalidCatalogNameError=InvalidCatalogNameError,
        connect=_fake_admin_connect,
    )

    import workflows_mcp.engine.knowledge.schema as schema_mod
    import workflows_mcp.engine.sql.postgres_backend as postgres_backend_mod

    monkeypatch.setattr(schema_mod, "ensure_schema", _fake_ensure_schema)
    monkeypatch.setattr(postgres_backend_mod, "PostgresBackend", _FakePostgresBackend)
    monkeypatch.setitem(sys.modules, "asyncpg", fake_asyncpg)

    backend_result = await server._prepare_memory_schema("localhost")

    assert backend_result is backend
    assert backend.connect_calls == 2
    assert ensure_schema_calls["count"] == 1
    assert backend.disconnect_calls == 0
    assert "Memory DB bootstrap detected concurrent create; continuing" in caplog.text


@pytest.mark.asyncio
async def test_prepare_memory_schema_raises_when_bootstrap_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class InvalidCatalogNameError(Exception):
        pass

    first_error = InvalidCatalogNameError("database does not exist")
    backend = _FakeBackend(connect_side_effects=[first_error])
    admin_conn = _FakeAdminConn(execute_error=RuntimeError("permission denied"))

    class _FakePostgresBackend:
        def __new__(cls) -> _FakeBackend:
            return backend

    async def _fake_ensure_schema(_backend: Any) -> None:
        raise AssertionError("ensure_schema should not be called when bootstrap fails")

    async def _fake_admin_connect(**_kwargs: Any) -> _FakeAdminConn:
        return admin_conn

    fake_asyncpg = SimpleNamespace(
        InvalidCatalogNameError=InvalidCatalogNameError,
        connect=_fake_admin_connect,
    )

    import workflows_mcp.engine.knowledge.schema as schema_mod
    import workflows_mcp.engine.sql.postgres_backend as postgres_backend_mod

    monkeypatch.setattr(schema_mod, "ensure_schema", _fake_ensure_schema)
    monkeypatch.setattr(postgres_backend_mod, "PostgresBackend", _FakePostgresBackend)
    monkeypatch.setitem(sys.modules, "asyncpg", fake_asyncpg)
    monkeypatch.setenv("MEMORY_DB_AUTO_CREATE", "true")

    with pytest.raises(RuntimeError, match="permission denied"):
        await server._prepare_memory_schema("localhost")

    assert "Memory DB bootstrap failed" in caplog.text


@pytest.mark.asyncio
async def test_prepare_memory_schema_disconnects_on_schema_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()

    class _FakePostgresBackend:
        def __new__(cls) -> _FakeBackend:
            return backend

    async def _fake_ensure_schema(_backend: Any) -> None:
        raise RuntimeError("schema failure")

    import workflows_mcp.engine.knowledge.schema as schema_mod
    import workflows_mcp.engine.sql.postgres_backend as postgres_backend_mod

    monkeypatch.setattr(schema_mod, "ensure_schema", _fake_ensure_schema)
    monkeypatch.setattr(postgres_backend_mod, "PostgresBackend", _FakePostgresBackend)

    with pytest.raises(RuntimeError, match="schema failure"):
        await server._prepare_memory_schema("localhost")

    assert backend.connect_calls == 1
    assert backend.disconnect_calls == 1
