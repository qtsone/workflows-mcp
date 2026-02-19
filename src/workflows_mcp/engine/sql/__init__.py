"""SQL database backend module for the workflow engine.

This module provides a unified interface for executing SQL against multiple
database backends: SQLite, PostgreSQL, and MariaDB/MySQL.

Features:
    - Pluggable backend architecture
    - Automatic parameter placeholder conversion between engines
    - Connection pooling for remote databases
    - Transaction support with isolation levels
    - Schema management (CREATE TABLE IF NOT EXISTS)
    - Model-based CRUD operations (Active Record-style)

Usage:
    from workflows_mcp.engine.sql import (
        SqliteBackend,
        PostgresBackend,
        MariaDBBackend,
        ConnectionConfig,
        DatabaseEngine,
        QueryResult,
        ParamConverter,
        ModelSchema,
        QueryBuilder,
    )

    # SQLite (always available)
    backend = SqliteBackend()
    await backend.connect(ConnectionConfig(
        dialect=DatabaseEngine.SQLITE,
        path="/data/app.db"
    ))

    # Model-based CRUD
    schema = ModelSchema.from_dict({
        "table": "tasks",
        "columns": {
            "id": {"type": "text", "primary": True},
            "name": {"type": "text", "required": True},
        }
    })
    builder = QueryBuilder(schema, DatabaseEngine.SQLITE)
    sql, params = builder.insert({"name": "My Task"})
"""

from .backend import (
    ConnectionConfig,
    DatabaseBackend,
    DatabaseBackendBase,
    DatabaseEngine,
    Params,
    QueryResult,
)
from .model import ColumnDef, IndexDef, ModelSchema
from .param_converter import ParamConverter, convert_sql_for_dialect
from .query_builder import QueryBuilder
from .sqlite_backend import SqliteBackend

# PostgreSQL backend (optional dependency)
try:
    from .postgres_backend import PostgresBackend
except ImportError:
    PostgresBackend = None  # type: ignore[misc,assignment]

# MariaDB backend (optional dependency)
try:
    from .mariadb_backend import MariaDBBackend
except ImportError:
    MariaDBBackend = None  # type: ignore[misc,assignment]

__all__ = [
    # Core types
    "ConnectionConfig",
    "DatabaseBackend",
    "DatabaseBackendBase",
    "DatabaseEngine",
    "Params",
    "QueryResult",
    # Model-based CRUD
    "ModelSchema",
    "ColumnDef",
    "IndexDef",
    "QueryBuilder",
    # Parameter conversion
    "ParamConverter",
    "convert_sql_for_dialect",
    # Backends
    "SqliteBackend",
    "PostgresBackend",
    "MariaDBBackend",
]
