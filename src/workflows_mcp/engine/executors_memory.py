"""Memory workflow block executor using unified memory envelope."""

from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, Literal

from pydantic import Field

from .block import BlockInput, BlockOutput
from .execution import Execution
from .executor_base import BlockExecutor, ExecutorCapabilities, ExecutorSecurityLevel
from .memory_service import MemoryRequest, MemoryService

logger = logging.getLogger(__name__)


class MemoryInput(BlockInput):
    """Unified memory block input."""

    operation: Literal[
        "query",
        "ingest",
        "validate",
        "supersede",
        "archive",
        "maintain",
        "graph_upsert",
        "graph_delete",
    ] = Field(description="Memory operation")

    host: str = Field(
        default_factory=lambda: os.environ.get("MEMORY_DB_HOST", "localhost"),
        description="Memory PostgreSQL host. Defaults to MEMORY_DB_HOST or localhost.",
    )
    port: int = Field(
        default_factory=lambda: int(os.environ.get("MEMORY_DB_PORT", "5432")),
        description="Memory PostgreSQL port. Defaults to MEMORY_DB_PORT or 5432.",
    )
    database: str = Field(
        default_factory=lambda: os.environ.get("MEMORY_DB_NAME", "memory_db"),
        description="Memory PostgreSQL database name. Defaults to MEMORY_DB_NAME or memory_db.",
    )
    username: str | None = Field(
        default_factory=lambda: os.environ.get("MEMORY_DB_USER"),
        description="Memory PostgreSQL username. Defaults to MEMORY_DB_USER when set.",
    )
    password: str | None = Field(
        default_factory=lambda: os.environ.get("MEMORY_DB_PASSWORD"),
        description="Memory PostgreSQL password. Defaults to MEMORY_DB_PASSWORD when set.",
    )

    scope: dict[str, Any] | None = Field(
        default=None,
        description="Scope envelope (room/corridor/global and retrieval boundaries).",
    )
    query: dict[str, Any] | None = Field(
        default=None,
        description="Query payload for retrieval operations (text, strategy, and limits).",
    )
    record: dict[str, Any] | None = Field(
        default=None,
        description="Record payload for ingest/update operations (memory content and metadata).",
    )
    graph: dict[str, Any] | None = Field(
        default=None,
        description="Graph mutation payload (entities/relations for upsert or delete).",
    )
    maintenance: dict[str, Any] | None = Field(
        default=None,
        description="Maintenance payload for validate/supersede/archive/maintain operations.",
    )
    response: dict[str, Any] | None = Field(
        default=None,
        description="Response shaping controls (include sections, verbosity, and formatting flags).",
    )


class MemoryOutput(BlockOutput):
    """Unified memory block output."""

    success: bool = Field(default=False)
    error: str | None = Field(default=None, description="Error message when the operation fails.")
    operation: str = Field(default="")
    result: dict[str, Any] = Field(
        default_factory=dict,
        description="Operation result envelope returned by the memory service.",
    )


class MemoryExecutor(BlockExecutor):
    """Unified memory block executor."""

    type_name: ClassVar[str] = "Memory"
    input_type: ClassVar[type[BlockInput]] = MemoryInput
    output_type: ClassVar[type[BlockOutput]] = MemoryOutput

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(can_network=True)

    async def execute(self, inputs: MemoryInput, context: Execution) -> MemoryOutput:  # type: ignore[override]
        try:
            backend = self._create_backend()
            config = self._create_config(inputs)
        except Exception as e:
            return MemoryOutput(success=False, error=str(e), operation=inputs.operation)

        try:
            await backend.connect(config)
            service = MemoryService(backend, context)
            request = MemoryRequest.model_validate(
                {
                    "operation": inputs.operation,
                    "scope": inputs.scope or {},
                    "query": inputs.query,
                    "record": inputs.record,
                    "graph": inputs.graph,
                    "maintenance": inputs.maintenance,
                    "response": inputs.response or {},
                }
            )
            result = await service.execute(request)
            return MemoryOutput(
                success=True,
                operation=inputs.operation,
                result=result.model_dump(by_alias=True),
            )
        except Exception as e:
            logger.error("Memory operation '%s' failed: %s", inputs.operation, e)
            return MemoryOutput(success=False, error=str(e), operation=inputs.operation)
        finally:
            await backend.disconnect()

    def _create_backend(self) -> Any:
        try:
            from .sql.postgres_backend import PostgresBackend
        except ImportError as e:
            raise ImportError(
                "PostgreSQL backend requires 'asyncpg'. "
                "Install with: pip install workflows-mcp[postgresql]"
            ) from e
        return PostgresBackend()

    def _create_config(self, inputs: MemoryInput) -> Any:
        from .sql import ConnectionConfig, DatabaseEngine

        return ConnectionConfig(
            dialect=DatabaseEngine.POSTGRESQL,
            host=inputs.host,
            port=inputs.port,
            database=inputs.database,
            username=inputs.username,
            password=inputs.password,
        )
