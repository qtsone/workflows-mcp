"""Database backend protocol and data classes for SQL executor.

This module defines the abstract interface that all database backends must implement,
along with shared data structures for configuration and query results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class DatabaseEngine(Enum):
    """Supported database engines."""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MARIADB = "mariadb"


@dataclass
class ConnectionConfig:
    """Database connection configuration.

    Attributes:
        dialect: Database dialect (sqlite, postgresql, mariadb)
        path: SQLite database file path (or ":memory:" for in-memory)
        host: Database server host (PostgreSQL/MariaDB)
        port: Database server port
        database: Database name
        username: Database username
        password: Database password
        ssl: SSL/TLS configuration (bool or sslmode string)
        timeout: Query execution timeout in seconds
        connect_timeout: Connection establishment timeout in seconds
        pool_size: Connection pool size (remote DBs only)
        pool_timeout: Max seconds to wait for pool connection
        options: Backend-specific options (e.g., sqlite_pragmas)
    """

    dialect: DatabaseEngine
    path: str | None = None
    host: str | None = None
    port: int | None = None
    database: str | None = None
    username: str | None = None
    password: str | None = None
    ssl: bool | str = False
    timeout: int = 30
    connect_timeout: int = 10
    pool_size: int = 5
    pool_timeout: int = 30
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration based on dialect."""
        if self.dialect == DatabaseEngine.SQLITE:
            if not self.path:
                raise ValueError("SQLite requires 'path' parameter")
        else:
            if not self.host:
                raise ValueError(f"{self.dialect.value} requires 'host' parameter")
            if not self.database:
                raise ValueError(f"{self.dialect.value} requires 'database' parameter")

            # Set default ports
            if self.port is None:
                if self.dialect == DatabaseEngine.POSTGRESQL:
                    self.port = 5432
                elif self.dialect == DatabaseEngine.MARIADB:
                    self.port = 3306


@dataclass
class QueryResult:
    """Unified query result across backends.

    Attributes:
        rows: Result rows as list of dicts (for SELECT queries)
        row_count: Number of rows returned (SELECT) or affected (INSERT/UPDATE/DELETE)
        columns: Column names from result set
        last_insert_id: Last inserted row ID (for INSERT operations)
        affected_rows: Number of rows affected by INSERT/UPDATE/DELETE
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    columns: list[str] = field(default_factory=list)
    last_insert_id: int | None = None
    affected_rows: int = 0


# Type alias for query parameters
Params = tuple[Any, ...] | list[Any] | dict[str, Any] | None


@runtime_checkable
class DatabaseBackend(Protocol):
    """Protocol defining the interface for database backend implementations.

    All database backends must implement this interface to be usable by the
    SQL executor. Backends should be stateful (hold connection/pool) and
    support async operations.

    Example implementation:
        class SqliteBackend:
            dialect = DatabaseEngine.SQLITE

            async def connect(self, config: ConnectionConfig) -> None:
                self._conn = sqlite3.connect(config.path)

            async def query(self, sql: str, params: Params) -> QueryResult:
                cursor = self._conn.execute(sql, params or ())
                rows = [dict(row) for row in cursor.fetchall()]
                return QueryResult(rows=rows, row_count=len(rows))
    """

    dialect: DatabaseEngine

    async def connect(self, config: ConnectionConfig) -> None:
        """Establish database connection or create connection pool.

        Args:
            config: Connection configuration

        Raises:
            SqlConnectionError: If connection fails
        """
        ...

    async def disconnect(self) -> None:
        """Close database connection or pool gracefully.

        Should be called when the backend is no longer needed.
        Safe to call multiple times.
        """
        ...

    async def query(self, sql: str, params: Params = None) -> QueryResult:
        """Execute a SELECT query and return results.

        Args:
            sql: SQL SELECT statement
            params: Query parameters (positional or named)

        Returns:
            QueryResult with rows and metadata

        Raises:
            SqlQueryError: If query execution fails
        """
        ...

    async def execute(self, sql: str, params: Params = None) -> QueryResult:
        """Execute an INSERT/UPDATE/DELETE statement.

        Args:
            sql: SQL statement
            params: Query parameters (positional or named)

        Returns:
            QueryResult with affected_rows and last_insert_id

        Raises:
            SqlQueryError: If execution fails
        """
        ...

    async def execute_many(
        self, sql: str, params_list: list[tuple[Any, ...] | dict[str, Any]]
    ) -> QueryResult:
        """Execute the same statement with multiple parameter sets.

        Useful for batch inserts/updates.

        Args:
            sql: SQL statement
            params_list: List of parameter sets

        Returns:
            QueryResult with total affected_rows

        Raises:
            SqlQueryError: If execution fails
        """
        ...

    async def begin_transaction(self, isolation_level: str | None = None) -> None:
        """Begin a transaction with optional isolation level.

        Args:
            isolation_level: Transaction isolation level. Values depend on dialect:
                - SQLite: "deferred", "immediate", "exclusive"
                - PostgreSQL/MariaDB: "read_uncommitted", "read_committed",
                  "repeatable_read", "serializable"

        Raises:
            SqlQueryError: If transaction cannot be started
        """
        ...

    async def commit(self) -> None:
        """Commit the current transaction.

        Raises:
            SqlQueryError: If commit fails
        """
        ...

    async def rollback(self) -> None:
        """Rollback the current transaction.

        Safe to call even if no transaction is active.
        """
        ...

    async def execute_script(self, sql: str) -> None:
        """Execute multiple SQL statements as a script.

        Used for schema creation and multi-statement DDL.

        Args:
            sql: Multi-statement SQL script

        Raises:
            SqlSchemaError: If script execution fails
        """
        ...

    @property
    def in_transaction(self) -> bool:
        """Check if currently in a transaction."""
        ...


class DatabaseBackendBase(ABC):
    """Abstract base class for database backends.

    Provides common functionality and enforces the interface.
    Subclasses must implement all abstract methods.
    """

    dialect: DatabaseEngine
    _in_transaction: bool = False

    @abstractmethod
    async def connect(self, config: ConnectionConfig) -> None:
        """Establish database connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection."""
        pass

    @abstractmethod
    async def query(self, sql: str, params: Params = None) -> QueryResult:
        """Execute SELECT query."""
        pass

    @abstractmethod
    async def execute(self, sql: str, params: Params = None) -> QueryResult:
        """Execute INSERT/UPDATE/DELETE."""
        pass

    async def execute_many(
        self, sql: str, params_list: list[tuple[Any, ...] | dict[str, Any]]
    ) -> QueryResult:
        """Execute statement with multiple parameter sets.

        Default implementation executes in a transaction.
        Subclasses may override for batch optimization.
        """
        total_affected = 0
        last_id = None

        await self.begin_transaction()
        try:
            for params in params_list:
                result = await self.execute(sql, params)
                total_affected += result.affected_rows
                if result.last_insert_id is not None:
                    last_id = result.last_insert_id
            await self.commit()
        except Exception:
            await self.rollback()
            raise

        return QueryResult(
            affected_rows=total_affected,
            row_count=total_affected,
            last_insert_id=last_id,
        )

    @abstractmethod
    async def begin_transaction(self, isolation_level: str | None = None) -> None:
        """Begin transaction."""
        pass

    @abstractmethod
    async def commit(self) -> None:
        """Commit transaction."""
        pass

    @abstractmethod
    async def rollback(self) -> None:
        """Rollback transaction."""
        pass

    @abstractmethod
    async def execute_script(self, sql: str) -> None:
        """Execute multi-statement SQL script."""
        pass

    @property
    def in_transaction(self) -> bool:
        """Check if in transaction."""
        return self._in_transaction
