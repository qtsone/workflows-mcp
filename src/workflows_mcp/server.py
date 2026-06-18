"""MCP server initialization and HTTP app builder for workflows-mcp.

This module provides:
- The FastMCP server and its lifespan context manager (MCP protocol).
- ``build_app()``: composes the full FastAPI HTTP application from auth,
  readiness, config, and app components (HTTP transport).
- ``main()``: HTTP-only entry point that starts the Uvicorn server.

All tool implementations are in the tools module.

Following the official Anthropic Python SDK patterns:
- Lifespan context manager for resource initialization and cleanup
- Context injection for tool access to shared resources
"""

import asyncio
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from fastapi import FastAPI

from .context import AppContext, AppContextType, MemoryBackendUnavailableError
from .engine.workflow_source_loader import (
    WorkflowSourceReloadError,
    WorkflowSourceReloadSummary,
    reload_registry_from_source_paths,
)
from .http.lifespan import AppResources, build_resources, start_resources, stop_resources
from .metadata.repos.workflow_sources_repo import SQLiteWorkflowSourcesRepository

logger = logging.getLogger(__name__)


def _has_registered_memory_tools(mcp_server: FastMCP) -> bool:
    """Return True when memory MCP tools are already registered."""
    tools = mcp_server._tool_manager._tools
    return "memory" in tools


def _resolve_base_dir(base_dir: Path | None = None) -> Path:
    """Resolve runtime base directory from explicit value or environment."""
    if base_dir is not None:
        return base_dir

    config_dir = os.getenv("WORKFLOWS_CONFIG_DIR")
    if config_dir and config_dir.strip():
        return Path(config_dir).expanduser()

    return Path.home() / ".workflows"


# =============================================================================
# Shared Resources and Lifespan Management
# =============================================================================


def _is_enabled_env_flag(name: str, *, default: bool) -> bool:
    """Parse a boolean environment flag with a default value."""
    raw = os.getenv(name)
    if raw is None:
        return default

    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _quote_pg_identifier(identifier: str) -> str:
    """Safely quote a PostgreSQL identifier for CREATE DATABASE statements."""
    return f'"{identifier.replace('"', '""')}"'


def _is_duplicate_database_error(exc: BaseException) -> bool:
    """Return True when the error indicates CREATE DATABASE raced with another startup."""
    return (
        exc.__class__.__name__ == "DuplicateDatabaseError"
        or getattr(exc, "sqlstate", None) == "42P04"
    )


async def _bootstrap_memory_database(
    asyncpg_module: Any,
    *,
    host: str,
    port: int,
    admin_database: str,
    target_database: str,
    username: str | None,
    password: str | None,
) -> None:
    """Create the memory database once using an admin database connection."""
    logger.warning(
        "Memory DB bootstrap attempt starting",
        extra={
            "memory_db_name": target_database,
            "admin_database": admin_database,
        },
    )

    admin_conn = await asyncpg_module.connect(
        host=host,
        port=port,
        database=admin_database,
        user=username,
        password=password,
    )
    try:
        sql = f"CREATE DATABASE {_quote_pg_identifier(target_database)}"
        await admin_conn.execute(sql)
        logger.warning(
            "Memory DB bootstrap succeeded",
            extra={"memory_db_name": target_database},
        )
    except Exception as exc:
        if _is_duplicate_database_error(exc):
            logger.warning(
                "Memory DB bootstrap detected concurrent create; continuing",
                extra={"memory_db_name": target_database},
            )
            return

        logger.error(
            "Memory DB bootstrap failed",
            extra={"memory_db_name": target_database},
            exc_info=True,
        )
        raise
    finally:
        await admin_conn.close()


async def _prepare_memory_schema(memory_db_host: str) -> Any:
    """Connect to memory DB, ensure schema, and return a connected backend."""
    from .engine.knowledge.schema import ensure_schema
    from .engine.sql.backend import ConnectionConfig, DatabaseEngine
    from .engine.sql.postgres_backend import PostgresBackend

    memory_db_port = int(os.getenv("MEMORY_DB_PORT", "5432"))
    memory_db_name = os.getenv("MEMORY_DB_NAME", "memory_db")
    memory_db_user = os.getenv("MEMORY_DB_USER")
    memory_db_password = os.getenv("MEMORY_DB_PASSWORD")
    memory_db_auto_create = _is_enabled_env_flag("MEMORY_DB_AUTO_CREATE", default=True)
    memory_db_admin_database = os.getenv("MEMORY_DB_ADMIN_DATABASE", "postgres")

    config = ConnectionConfig(
        dialect=DatabaseEngine.POSTGRESQL,
        host=memory_db_host,
        port=memory_db_port,
        database=memory_db_name,
        username=memory_db_user,
        password=memory_db_password,
    )

    backend = PostgresBackend()
    connected = False

    try:
        await backend.connect(config)
        connected = True
    except Exception as exc:
        asyncpg_module: Any | None = None
        try:
            asyncpg_module = import_module("asyncpg")
        except ImportError:
            asyncpg_module = None

        invalid_catalog_error: type[BaseException] | None = None
        if asyncpg_module is not None:
            invalid_catalog_error = getattr(asyncpg_module, "InvalidCatalogNameError", None)

        if invalid_catalog_error is not None and isinstance(exc, invalid_catalog_error):
            logger.warning(
                "Memory DB is missing",
                extra={"memory_db_name": memory_db_name},
            )

            if not memory_db_auto_create:
                logger.warning(
                    "Memory DB bootstrap skipped (MEMORY_DB_AUTO_CREATE=false)",
                    extra={"memory_db_name": memory_db_name},
                )
                raise

            if asyncpg_module is None:
                logger.error(
                    "Memory DB bootstrap failed (asyncpg unavailable)",
                    extra={"memory_db_name": memory_db_name},
                )
                raise

            await _bootstrap_memory_database(
                asyncpg_module,
                host=memory_db_host,
                port=memory_db_port,
                admin_database=memory_db_admin_database,
                target_database=memory_db_name,
                username=memory_db_user,
                password=memory_db_password,
            )

            logger.warning(
                "Retrying memory DB connection after bootstrap",
                extra={"memory_db_name": memory_db_name},
            )
            await backend.connect(config)
            connected = True
        else:
            raise

    try:
        await ensure_schema(backend)
        return backend
    except Exception:
        if connected:
            await backend.disconnect()
        raise


def get_max_recursion_depth() -> int:
    """Get maximum workflow recursion depth from environment.

    Reads WORKFLOWS_MAX_RECURSION_DEPTH environment variable.
    Default: 50, Valid range: 1-10000 (clamped automatically)

    Returns:
        Maximum recursion depth (1-10000)
    """
    try:
        depth = int(os.getenv("WORKFLOWS_MAX_RECURSION_DEPTH", "50"))
        return max(1, min(10000, depth))
    except ValueError:
        return 50


def get_graceful_shutdown_timeout() -> int:
    """Get Uvicorn graceful shutdown timeout from environment."""
    raw = os.getenv("WORKFLOWS_GRACEFUL_SHUTDOWN_TIMEOUT", "5")
    try:
        timeout = int(raw)
    except ValueError:
        logger.warning(
            "Invalid WORKFLOWS_GRACEFUL_SHUTDOWN_TIMEOUT value %r; using 5 seconds.",
            raw,
        )
        return 5

    if timeout < 0:
        logger.warning(
            "Invalid WORKFLOWS_GRACEFUL_SHUTDOWN_TIMEOUT value %r; using 5 seconds.",
            raw,
        )
        return 5

    return timeout


def _builtin_workflow_path() -> Path:
    """Resolve the packaged builtin workflow directory (templates/memory)."""
    from importlib.resources import files

    return Path(str(files("workflows_mcp").joinpath("templates").joinpath("memory")))


def load_workflows(resources: AppResources) -> WorkflowSourceReloadSummary:
    """Load workflows from packaged built-ins and SQLite-managed user sources.

    Built-in/system workflows are loaded from the package-derived path returned
    by ``_builtin_workflow_path()`` at runtime.  No rows are written to SQLite
    for the built-in source.
    """
    repo = SQLiteWorkflowSourcesRepository(resources.metadata_db_conn)
    sources = repo.list()
    source_paths = [source.source_path for source in sources]

    builtin_path = _builtin_workflow_path()

    try:
        summary = reload_registry_from_source_paths(
            resources.workflow_registry,
            source_paths,
            builtin_paths=[builtin_path],
        )
    except WorkflowSourceReloadError as exc:
        for source in sources:
            repo.update_reload_state(
                source.source_id,
                status="failed",
                error_message=exc.message,
            )
        logger.warning(
            "Workflow registry reload from SQLite sources failed",
            extra={
                "source_count": len(sources),
                "error_code": exc.code,
            },
        )
        raise

    for source in sources:
        repo.update_reload_state(source.source_id, status="loaded", error_message=None)

    logger.info(
        "Workflow registry reloaded from SQLite sources",
        extra={
            "source_count": summary.source_count,
            "workflow_count": summary.workflow_count,
        },
    )
    return summary


@asynccontextmanager
async def app_lifespan(_server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with resource initialization and cleanup.

    This lifespan context manager:
    1. Initializes shared resources (workflow registry, executor registry, checkpoint store)
    2. Reloads workflow registry from SQLite-backed workflow source records
    3. Initializes secret management system
    4. Yields context to make resources available to tools
    5. Cleans up resources on shutdown

    Environment Variables:
        WORKFLOWS_MAX_RECURSION_DEPTH: Maximum workflow recursion depth
            (default: 50, range: 1-10000)
        WORKFLOW_SECRET_*: Secret environment variables for workflows

    Args:
        _server: FastMCP server instance (unused, required by FastMCP signature)

    Yields:
        AppContext with initialized resources (ADR-008 pattern)
    """
    # Startup: initialize resources
    logger.info("Initializing MCP server resources...")

    resources = build_resources(base_dir=_resolve_base_dir())
    app_context = resources.app_context
    executor_registry = resources.executor_registry
    try:
        await start_resources(resources)

        # Read max recursion depth from environment
        max_recursion_depth = resources.max_recursion_depth
        if max_recursion_depth != 50:
            logger.info(f"Using max recursion depth: {max_recursion_depth}")

        # Initialize secret provider and check for configured secrets
        from .engine.secrets import EnvVarSecretProvider

        secret_provider = app_context.secret_provider or EnvVarSecretProvider()
        secret_keys = await secret_provider.list_secret_keys()

        logger.info(f"Secret provider: {secret_provider.__class__.__name__}")
        logger.info(f"Available secrets: {len(secret_keys)}")

        if len(secret_keys) == 0:
            logger.warning(
                "No secrets configured. "
                "Use WORKFLOW_SECRET_* environment variables to provide secrets."
            )
        else:
            # Log secret keys (not values!) for debugging
            logger.debug(f"Secret keys: {', '.join(sorted(secret_keys))}")

        llm_config = resources.llm_config_loader.load_config()

        logger.info(
            "LLM config: "
            f"{len(llm_config.providers)} providers, {len(llm_config.profiles)} profiles"
        )
        if llm_config.default_profile:
            logger.info(f"Default LLM profile: {llm_config.default_profile}")

        app_context.reload_workflows = lambda: load_workflows(resources)

        # Load workflows into registry
        load_workflows(resources)

        from .tools_memory import register_memory_tools

        oss_mode_enabled = _is_enabled_env_flag("WORKFLOWS_OSS_MODE", default=True)
        project_tools_enabled = _is_enabled_env_flag("WORKFLOWS_ENABLE_PROJECT_TOOLS", default=True)
        if "WORKFLOWS_ENABLE_PROJECT_TOOLS" not in os.environ and (
            "WORKFLOWS_ENABLE_TEMP_PROJECT_TOOLS" in os.environ
        ):
            project_tools_enabled = _is_enabled_env_flag(
                "WORKFLOWS_ENABLE_TEMP_PROJECT_TOOLS", default=True
            )
        expose_project_tools = oss_mode_enabled and project_tools_enabled
        if not _has_registered_memory_tools(_server):
            register_memory_tools(
                _server,
                enable_project_tools=expose_project_tools,
            )

        app_context.memory_backend_unavailable_error = MemoryBackendUnavailableError(
            code="MEMORY_BACKEND_UNAVAILABLE",
            message="Memory backend is unavailable.",
            retryable=False,
            actionable_fix=(
                "Configure PostgreSQL/pgvector via /api/admin/v1/database/settings and "
                "/api/admin/v1/database/connection-test, then verify /ready reports "
                "healthy before retrying."
            ),
        )

        # Initialize memory features if memory DB is configured.
        from .memory_runtime import refresh_memory_backend

        memory_db_host = os.getenv("MEMORY_DB_HOST")
        if memory_db_host:
            try:
                app_context.memory_backend = await _prepare_memory_schema(memory_db_host)
                app_context.memory_backend_lock = asyncio.Lock()
                app_context.memory_backend_unavailable_error = None

                # Schema OK — register memory block executor.
                from .engine.executors_memory import MemoryExecutor

                if not executor_registry.has("Memory"):
                    executor_registry.register(MemoryExecutor())
                from .engine.memory_service import AUDIT_FAIL_CLOSED

                logger.info(
                    "Memory features enabled (DB ready)",
                    extra={
                        "audit_fail_closed": AUDIT_FAIL_CLOSED,
                        "oss_mode_enabled": oss_mode_enabled,
                        "project_tools_enabled": project_tools_enabled,
                        "expose_project_tools": expose_project_tools,
                    },
                )
            except Exception as exc:
                if app_context.memory_backend is not None:
                    try:
                        await app_context.memory_backend.disconnect()
                    finally:
                        app_context.memory_backend = None
                        app_context.memory_backend_lock = None

                incompatible_schema = isinstance(exc, RuntimeError) and (
                    "Incompatible knowledge schema detected" in str(exc)
                )
                if incompatible_schema:
                    app_context.memory_backend_unavailable_error = MemoryBackendUnavailableError(
                        code="MEMORY_BACKEND_UNAVAILABLE",
                        message=(
                            "Memory backend is unavailable due to incompatible knowledge schema."
                        ),
                        retryable=False,
                        actionable_fix=(
                            "Apply the documented knowledge schema migration, then "
                            "configure/verify "
                            "database readiness via /api/admin/v1/database/settings and /ready."
                        ),
                    )
                    logger.warning(
                        "Memory features disabled (schema incompatible). "
                        "Apply the documented migration path, then restart.",
                        exc_info=True,
                    )
                else:
                    app_context.memory_backend_unavailable_error = MemoryBackendUnavailableError(
                        code="MEMORY_BACKEND_UNAVAILABLE",
                        message="Memory backend is unavailable.",
                        retryable=False,
                        actionable_fix=(
                            "Configure PostgreSQL/pgvector via /api/admin/v1/database/settings and "
                            "/api/admin/v1/database/connection-test, then verify /ready reports "
                            "healthy before retrying."
                        ),
                    )
                    logger.warning(
                        "Memory features disabled (DB unreachable)",
                        exc_info=True,
                    )
        else:
            try:
                await refresh_memory_backend(
                    app_ctx=app_context,
                    executor_registry=executor_registry,
                    prefer_metadata=True,
                )
                if app_context.memory_backend is not None:
                    logger.info("Memory features enabled from PostgreSQL profile")
                else:
                    logger.info("Memory features disabled (PostgreSQL profile not configured)")
            except Exception:
                app_context.memory_backend_unavailable_error = MemoryBackendUnavailableError(
                    code="MEMORY_BACKEND_UNAVAILABLE",
                    message="Memory backend is unavailable.",
                    retryable=False,
                    actionable_fix=(
                        "Configure PostgreSQL/pgvector via /api/admin/v1/database/settings and "
                        "/api/admin/v1/database/connection-test, then verify /ready reports "
                        "healthy before retrying."
                    ),
                )
                logger.warning(
                    "Memory features disabled (PostgreSQL profile unreachable)",
                    exc_info=True,
                )

        # Make resources available to tools via AppContext
        yield app_context
    finally:
        # Shutdown: cleanup resources (reverse order)
        logger.info("Shutting down MCP server...")

        await stop_resources(resources)

        # Close shared memory backend if enabled
        if app_context.memory_backend is not None:
            try:
                await app_context.memory_backend.disconnect()
            except Exception:
                logger.warning("Failed to disconnect shared memory backend", exc_info=True)
            finally:
                app_context.memory_backend = None
                app_context.memory_backend_lock = None

        # No other explicit cleanup required because:
        # - WorkflowRegistry: In-memory only, no persistent state
        # - ExecutorRegistry: Stateless executor instances
        # - No file handles, network connections, or external resources to close


# Initialize MCP server with lifespan management
# Following Python MCP naming convention: {service}_mcp
mcp = FastMCP("workflows_mcp", lifespan=app_lifespan)


# =============================================================================
# Server Entry Point
# =============================================================================


def build_app(*, base_dir: Path | None = None) -> "FastAPI":
    """Compose and return the FastAPI HTTP application.

    Wires together authentication, readiness, config, and route layers.

    Parameters
    ----------
    base_dir:
        Directory used for token store and readiness checks.  Defaults to
        ``~/.workflows``.  Override in tests to avoid touching the real
        home directory.

    Returns
    -------
    fastapi.FastAPI
        Fully configured application ready for ASGI / Uvicorn.
    """
    from .auth import TokenStore
    from .http_app import create_app
    from .postgres_probe import ConfiguredPostgresProbe
    from .readiness import ReadinessService

    resolved_base = _resolve_base_dir(base_dir)
    frontend_mode = os.getenv("WORKFLOWS_FRONTEND_MODE", "").strip().lower()
    require_frontend_assets = frontend_mode == "production"

    token_store = TokenStore(resolved_base / "auth.json")

    readiness_service = ReadinessService(
        base_dir=resolved_base,
        probe=ConfiguredPostgresProbe(base_dir=resolved_base),
    )
    resources = build_resources(base_dir=resolved_base)

    @asynccontextmanager
    async def _http_lifespan(_app: "FastAPI") -> AsyncIterator[None]:
        await start_resources(resources)
        resources.app_context.reload_workflows = lambda: load_workflows(resources)
        load_workflows(resources)
        from .memory_runtime import refresh_memory_backend

        try:
            await refresh_memory_backend(
                app_ctx=resources.app_context,
                executor_registry=resources.executor_registry,
                prefer_metadata=True,
            )
        except Exception:
            logger.warning("HTTP startup memory backend refresh failed", exc_info=True)
        transport_mount = getattr(_app.state, "mcp_transport_mount", None)
        transport_started = False
        try:
            if transport_mount is not None:
                await transport_mount.startup()
                transport_started = True
            yield
        finally:
            if transport_mount is not None and transport_started:
                await transport_mount.shutdown()
            await stop_resources(resources)

    app = create_app(
        readiness_service=readiness_service,
        token_store=token_store,
        require_frontend_assets=require_frontend_assets,
        lifespan=_http_lifespan,
    )
    app.state.resources = resources

    return app


def main() -> None:
    """HTTP-only entry point.

    Starts a Uvicorn server hosting the FastAPI application built by
    :func:`build_app`.  Bind address and port are controlled via environment
    variables.

    Environment Variables
    ---------------------
    WORKFLOWS_BIND_HOST:
        Interface to bind to (default: ``127.0.0.1``).
    WORKFLOWS_PORT:
        Port number (default: ``8000``).
    WORKFLOWS_LOG_LEVEL:
        Logging level (default: ``INFO``).
    """
    import uvicorn

    valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    log_level_str = os.getenv("WORKFLOWS_LOG_LEVEL", "INFO").upper()

    if log_level_str not in valid_log_levels:
        print(
            f"Warning: Invalid WORKFLOWS_LOG_LEVEL '{log_level_str}'. "
            f"Valid levels: {', '.join(sorted(valid_log_levels))}. "
            "Using INFO.",
            file=sys.stderr,
        )
        log_level_str = "INFO"

    log_level = getattr(logging, log_level_str)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    logger.info("Starting HTTP server (press Ctrl+C to stop)...")

    port_str = os.getenv("WORKFLOWS_PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        logger.error("Invalid WORKFLOWS_PORT value %r — must be an integer. Exiting.", port_str)
        sys.exit(1)

    try:
        uvicorn.run(
            build_app(),
            host=os.getenv("WORKFLOWS_BIND_HOST", "127.0.0.1"),
            port=port,
            log_level=log_level_str.lower(),
            timeout_graceful_shutdown=get_graceful_shutdown_timeout(),
        )
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down gracefully...")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)

    logger.info("Server shutdown complete")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Server infrastructure
    "mcp",
    "main",
    "build_app",
    "AppContext",
    "AppContextType",
    # Workflow loading (exposed for testing)
    "load_workflows",
]
