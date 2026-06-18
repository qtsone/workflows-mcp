from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP

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


class _FakeMemoryRequestContext:
    def __init__(self, lifespan_context: Any) -> None:
        self.lifespan_context = lifespan_context
        self.session = object()


class _FakeMemoryCtx:
    def __init__(self, lifespan_context: Any) -> None:
        self.request_context = _FakeMemoryRequestContext(lifespan_context)


def _tool_fn(mcp_server: FastMCP, name: str) -> Any:
    tool = mcp_server._tool_manager._tools.get(name)
    assert tool is not None, f"Tool {name!r} should be registered"
    return tool.fn


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

    import workflows_mcp.engine.sql.postgres_backend as postgres_backend_mod
    import workflows_mcp.memory.knowledge.schema as schema_mod

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
    local_mcp = FastMCP("memory-backend-reuse-test", lifespan=server.app_lifespan)
    fake_backend = _FakeLifespanBackend()
    prepare_calls = {"count": 0}
    register_calls = {"count": 0}

    async def _fake_prepare_memory_schema(_memory_db_host: str) -> _FakeLifespanBackend:
        prepare_calls["count"] += 1
        return fake_backend

    def _fake_register_memory_tools(
        _mcp_server: Any,
        *,
        enable_project_tools: bool = True,
    ) -> None:
        assert isinstance(enable_project_tools, bool)
        register_calls["count"] += 1

    class _FakeMemoryExecutor:
        type_name = "Memory"

    import workflows_mcp.memory.executors_memory as executors_memory_mod
    import workflows_mcp.tools_memory as tools_memory_mod

    monkeypatch.setattr(server, "_prepare_memory_schema", _fake_prepare_memory_schema)
    monkeypatch.setattr(server, "load_workflows", lambda _registry: None)
    monkeypatch.setattr(executors_memory_mod, "MemoryExecutor", _FakeMemoryExecutor)
    monkeypatch.setattr(tools_memory_mod, "register_memory_tools", _fake_register_memory_tools)
    monkeypatch.setenv("MEMORY_DB_HOST", "localhost")
    monkeypatch.setenv("WORKFLOWS_IO_QUEUE_ENABLED", "false")
    monkeypatch.setenv("WORKFLOWS_JOB_QUEUE_ENABLED", "false")

    async with server.app_lifespan(local_mcp) as app_context:
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

    import workflows_mcp.engine.sql.postgres_backend as postgres_backend_mod
    import workflows_mcp.memory.knowledge.schema as schema_mod

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

    import workflows_mcp.engine.sql.postgres_backend as postgres_backend_mod
    import workflows_mcp.memory.knowledge.schema as schema_mod

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

    import workflows_mcp.engine.sql.postgres_backend as postgres_backend_mod
    import workflows_mcp.memory.knowledge.schema as schema_mod

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

    import workflows_mcp.engine.sql.postgres_backend as postgres_backend_mod
    import workflows_mcp.memory.knowledge.schema as schema_mod

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

    import workflows_mcp.engine.sql.postgres_backend as postgres_backend_mod
    import workflows_mcp.memory.knowledge.schema as schema_mod

    monkeypatch.setattr(schema_mod, "ensure_schema", _fake_ensure_schema)
    monkeypatch.setattr(postgres_backend_mod, "PostgresBackend", _FakePostgresBackend)

    with pytest.raises(RuntimeError, match="schema failure"):
        await server._prepare_memory_schema("localhost")

    assert backend.connect_calls == 1
    assert backend.disconnect_calls == 1


@pytest.mark.asyncio
async def test_app_lifespan_registers_memory_tools_when_backend_unavailable_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_mcp = FastMCP("memory-unavailable-test", lifespan=server.app_lifespan)

    async def _fake_prepare_memory_schema(_memory_db_host: str) -> Any:
        raise ConnectionError("backend down")

    monkeypatch.setattr(server, "_prepare_memory_schema", _fake_prepare_memory_schema)
    monkeypatch.setattr(server, "load_workflows", lambda _registry: None)
    monkeypatch.setenv("MEMORY_DB_HOST", "localhost")
    monkeypatch.setenv("WORKFLOWS_IO_QUEUE_ENABLED", "false")
    monkeypatch.setenv("WORKFLOWS_JOB_QUEUE_ENABLED", "false")

    async with server.app_lifespan(local_mcp) as app_context:
        for name in ("memory", "onboard", "sync", "select"):
            assert local_mcp._tool_manager._tools.get(name) is not None

        memory = _tool_fn(local_mcp, "memory")
        onboard = _tool_fn(local_mcp, "onboard")
        sync = _tool_fn(local_mcp, "sync")
        select = _tool_fn(local_mcp, "select")
        ctx = _FakeMemoryCtx(app_context)

        memory_result = await memory(operation="query", query={"text": "hello"}, ctx=ctx)
        onboard_result = await onboard(
            scope={"palace": "forge"},
            ingest={"format": "structured", "memories": [{"content": "x"}]},
            ctx=ctx,
        )
        sync_result = await sync(
            scope={"palace": "forge"},
            ingest={"format": "structured"},
            ctx=ctx,
        )
        select_result = await select(scope={"palace": "forge"}, ctx=ctx)

        for result in (memory_result, onboard_result, sync_result, select_result):
            payload = result.structuredContent
            assert isinstance(payload, dict)
            assert "error" in payload
            err = payload["error"]
            assert err["code"] == "MEMORY_BACKEND_UNAVAILABLE"
            assert "/api/admin/v1/database/settings" in (err.get("actionable_fix") or "")
            assert "/ready" in (err.get("actionable_fix") or "")
            serialized = json.dumps(payload)
            assert "postgres://" not in serialized
            assert "MEMORY_DB_PASSWORD" not in serialized


@pytest.mark.asyncio
async def test_app_lifespan_registers_memory_tools_when_schema_incompatible_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_mcp = FastMCP("memory-incompatible-test", lifespan=server.app_lifespan)

    async def _fake_prepare_memory_schema(_memory_db_host: str) -> Any:
        raise RuntimeError("Incompatible knowledge schema detected: expected v3")

    monkeypatch.setattr(server, "_prepare_memory_schema", _fake_prepare_memory_schema)
    monkeypatch.setattr(server, "load_workflows", lambda _registry: None)
    monkeypatch.setenv("MEMORY_DB_HOST", "localhost")
    monkeypatch.setenv("WORKFLOWS_IO_QUEUE_ENABLED", "false")
    monkeypatch.setenv("WORKFLOWS_JOB_QUEUE_ENABLED", "false")

    async with server.app_lifespan(local_mcp) as app_context:
        memory = _tool_fn(local_mcp, "memory")
        payload = (
            await memory(
                operation="query",
                query={"text": "hello"},
                ctx=_FakeMemoryCtx(app_context),
            )
        ).structuredContent
        assert isinstance(payload, dict)
        assert payload.get("error", {}).get("code") == "MEMORY_BACKEND_UNAVAILABLE"


@pytest.mark.asyncio
async def test_app_lifespan_registers_memory_tools_only_once_per_mcp_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_mcp = FastMCP("memory-idempotent-registration-test", lifespan=server.app_lifespan)

    async def _fake_prepare_memory_schema(_memory_db_host: str) -> Any:
        raise ConnectionError("backend down")

    monkeypatch.setattr(server, "_prepare_memory_schema", _fake_prepare_memory_schema)
    monkeypatch.setattr(server, "load_workflows", lambda _registry: None)
    monkeypatch.setenv("MEMORY_DB_HOST", "localhost")
    monkeypatch.setenv("WORKFLOWS_IO_QUEUE_ENABLED", "false")
    monkeypatch.setenv("WORKFLOWS_JOB_QUEUE_ENABLED", "false")

    async with server.app_lifespan(local_mcp):
        first_memory_tool = local_mcp._tool_manager._tools.get("memory")
        assert first_memory_tool is not None

    async with server.app_lifespan(local_mcp):
        second_memory_tool = local_mcp._tool_manager._tools.get("memory")
        assert second_memory_tool is first_memory_tool

    assert len(local_mcp._tool_manager._tools) >= 1
