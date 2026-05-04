"""Shared HTTP/FastMCP resource construction primitives."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from workflows_mcp.context import AppContext
from workflows_mcp.engine import WorkflowRegistry
from workflows_mcp.engine.executor_base import ExecutorRegistry, create_default_registry
from workflows_mcp.engine.io_queue import IOQueue
from workflows_mcp.engine.job_queue import JobQueue
from workflows_mcp.engine.llm_config import LLMConfigLoader
from workflows_mcp.engine.secrets import (
    CompositeSecretProvider,
    EnvVarSecretProvider,
    SQLiteSecretProvider,
)
from workflows_mcp.metadata.migrations import migrate_metadata_db
from workflows_mcp.metadata.repos.watcher_repo import SQLiteWatcherRepository
from workflows_mcp.watcher.manager import WatcherManager


@dataclass(frozen=True)
class MetadataResources:
    """Resource metadata shared by HTTP and MCP entrypoints."""

    base_dir: Path
    io_queue_enabled: bool
    job_queue_enabled: bool
    job_queue_workers: int


@dataclass
class AppResources:
    """Unified container for foundational runtime resources."""

    workflow_registry: WorkflowRegistry
    executor_registry: ExecutorRegistry
    llm_config_loader: LLMConfigLoader
    io_queue: IOQueue | None
    job_queue: JobQueue | None
    max_recursion_depth: int
    metadata: MetadataResources
    app_context: AppContext
    metadata_db_conn: sqlite3.Connection
    watcher_manager: WatcherManager


def build_resources(*, base_dir: Path) -> AppResources:
    """Build shared foundational resources without starting background queues."""
    llm_config_loader = LLMConfigLoader(metadata_db_path=base_dir / "server.db")
    # Keep existing eager validation behavior.
    llm_config_loader.load_config()

    executor_registry = create_default_registry()
    workflow_registry = WorkflowRegistry()

    io_queue_enabled = os.getenv("WORKFLOWS_IO_QUEUE_ENABLED", "true").lower() == "true"
    job_queue_enabled = os.getenv("WORKFLOWS_JOB_QUEUE_ENABLED", "true").lower() == "true"
    num_workers = int(os.getenv("WORKFLOWS_JOB_QUEUE_WORKERS", "3"))

    io_queue = IOQueue() if io_queue_enabled else None
    max_recursion_depth = _get_max_recursion_depth()

    metadata_db_path = base_dir / "server.db"
    metadata_db_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_db_conn = sqlite3.connect(metadata_db_path, check_same_thread=False)
    os.chmod(metadata_db_path, 0o600)
    metadata_db_conn.row_factory = sqlite3.Row
    metadata_db_conn.execute("PRAGMA foreign_keys = ON")
    metadata_db_conn.execute("PRAGMA journal_mode = WAL")
    metadata_db_conn.execute("PRAGMA busy_timeout = 5000")
    migrate_metadata_db(metadata_db_conn)

    secret_provider = CompositeSecretProvider(
        [
            EnvVarSecretProvider(),
            SQLiteSecretProvider(
                db_path=metadata_db_path,
                key_path=base_dir / "secrets.key",
            ),
        ]
    )

    app_context = AppContext(
        registry=workflow_registry,
        executor_registry=executor_registry,
        llm_config_loader=llm_config_loader,
        io_queue=io_queue,
        job_queue=None,
        max_recursion_depth=max_recursion_depth,
        metadata_base_dir=base_dir,
        metadata_db_path=metadata_db_path,
        secret_provider=secret_provider,
    )

    job_queue = (
        JobQueue(app_context, num_workers=num_workers, db_path=str(metadata_db_path))
        if job_queue_enabled
        else None
    )
    app_context.job_queue = job_queue
    watcher_manager = WatcherManager(SQLiteWatcherRepository(metadata_db_conn))
    app_context.watcher_manager = watcher_manager

    metadata = MetadataResources(
        base_dir=base_dir,
        io_queue_enabled=io_queue_enabled,
        job_queue_enabled=job_queue_enabled,
        job_queue_workers=num_workers,
    )

    return AppResources(
        workflow_registry=workflow_registry,
        executor_registry=executor_registry,
        llm_config_loader=llm_config_loader,
        io_queue=io_queue,
        job_queue=job_queue,
        max_recursion_depth=max_recursion_depth,
        metadata=metadata,
        app_context=app_context,
        metadata_db_conn=metadata_db_conn,
        watcher_manager=watcher_manager,
    )


async def start_resources(resources: AppResources) -> None:
    resources.watcher_manager.start()


async def stop_resources(resources: AppResources) -> None:
    resources.watcher_manager.stop()
    resources.metadata_db_conn.close()


def _get_max_recursion_depth() -> int:
    try:
        depth = int(os.getenv("WORKFLOWS_MAX_RECURSION_DEPTH", "50"))
        return max(1, min(10000, depth))
    except ValueError:
        return 50
