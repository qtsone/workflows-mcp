"""SQL database backend module for the workflow engine.

This module provides a unified interface for executing SQL against multiple
database backends: SQLite, PostgreSQL, and MariaDB/MySQL.

Features:
    - Pluggable backend architecture
    - Automatic parameter placeholder conversion between dialects
    - Connection pooling for remote databases
    - Transaction support with isolation levels
    - Schema management (CREATE TABLE IF NOT EXISTS)

Usage:
    from workflows_mcp.engine.sql import (
        SqliteBackend,
        PostgresBackend,
        MariaDBBackend,
        ConnectionConfig,
        DatabaseDialect,
        QueryResult,
        ParamConverter,
    )

    # SQLite (always available)
    backend = SqliteBackend()
    await backend.connect(ConnectionConfig(
        dialect=DatabaseDialect.SQLITE,
        path="/data/app.db"
    ))

    # PostgreSQL (requires asyncpg)
    backend = PostgresBackend()
    await backend.connect(ConnectionConfig(
        dialect=DatabaseDialect.POSTGRESQL,
        host="localhost",
        database="mydb",
        username="user",
        password="pass"
    ))
"""

from .backend import (
    ConnectionConfig,
    DatabaseBackend,
    DatabaseBackendBase,
    DatabaseDialect,
    Params,
    QueryResult,
)
from .param_converter import ParamConverter, convert_sql_for_dialect
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
    "DatabaseDialect",
    "Params",
    "QueryResult",
    # Parameter conversion
    "ParamConverter",
    "convert_sql_for_dialect",
    # Backends
    "SqliteBackend",
    "PostgresBackend",
    "MariaDBBackend",
]
