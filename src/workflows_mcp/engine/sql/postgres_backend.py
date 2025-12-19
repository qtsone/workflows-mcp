"""PostgreSQL database backend implementation.

This module provides the PostgreSQL backend for the SQL executor, using
asyncpg for native async operation with connection pooling.

Features:
    - Native async driver (asyncpg)
    - Connection pooling with configurable size
    - SSL/TLS support with certificate validation
    - Prepared statement caching
    - Binary protocol for performance

Note:
    Requires the 'asyncpg' package: pip install workflows-mcp[postgresql]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .backend import ConnectionConfig, DatabaseBackendBase, DatabaseDialect, Params, QueryResult

if TYPE_CHECKING:
    import asyncpg  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


def _import_asyncpg() -> Any:
    """Import asyncpg with helpful error message if not installed."""
    try:
        import asyncpg

        return asyncpg
    except ImportError as e:
        raise ImportError(
            "PostgreSQL backend requires 'asyncpg' package. "
            "Install with: pip install workflows-mcp[postgresql]"
        ) from e


class PostgresBackend(DatabaseBackendBase):
    """PostgreSQL backend using asyncpg with connection pooling.

    This backend uses the asyncpg library for native async PostgreSQL
    access with connection pooling. It provides high-performance
    database access using PostgreSQL's binary protocol.

    Attributes:
        dialect: DatabaseDialect.POSTGRESQL

    Example:
        backend = PostgresBackend()
        await backend.connect(ConnectionConfig(
            dialect=DatabaseDialect.POSTGRESQL,
            host="localhost",
            port=5432,
            database="mydb",
            username="user",
            password="pass"
        ))
        result = await backend.query("SELECT * FROM users WHERE id = $1", (42,))
        await backend.disconnect()
    """

    dialect = DatabaseDialect.POSTGRESQL

    def __init__(self) -> None:
        """Initialize PostgreSQL backend."""
        self._pool: asyncpg.Pool | None = None
        self._conn: asyncpg.Connection | None = None
        self._in_transaction: bool = False
        self._config: ConnectionConfig | None = None

    async def connect(self, config: ConnectionConfig) -> None:
        """Create connection pool.

        Pool settings:
            - min_size: 1
            - max_size: config.pool_size
            - max_inactive_connection_lifetime: 300s
            - command_timeout: config.timeout

        Args:
            config: Connection configuration

        Raises:
            SqlConnectionError: If connection fails
            ImportError: If asyncpg is not installed
        """
        asyncpg = _import_asyncpg()
        self._config = config

        # Build SSL context if needed
        ssl_context = None
        if config.ssl:
            if isinstance(config.ssl, str):
                # SSL mode string (require, verify-ca, verify-full)
                if config.ssl in ("require", "verify-ca", "verify-full"):
                    ssl_context = True  # asyncpg will create appropriate context
            elif config.ssl is True:
                ssl_context = True

        # Create connection pool
        self._pool = await asyncpg.create_pool(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.username,
            password=config.password,
            ssl=ssl_context,
            min_size=1,
            max_size=config.pool_size,
            max_inactive_connection_lifetime=300,
            command_timeout=config.timeout,
            timeout=config.connect_timeout,
        )

        logger.debug(f"Connected to PostgreSQL: {config.host}:{config.port}/{config.database}")

    async def disconnect(self) -> None:
        """Close connection pool gracefully.

        Waits for all connections to be released before closing.
        """
        if self._conn is not None:
            # Release any acquired connection
            if self._pool is not None:
                await self._pool.release(self._conn)
            self._conn = None

        if self._pool is not None:
            await self._pool.close()
            self._pool = None

        self._in_transaction = False
        logger.debug("Disconnected from PostgreSQL")

    async def query(self, sql: str, params: Params = None) -> QueryResult:
        """Execute SELECT query and return results.

        Uses pool.fetch() for query execution.

        Args:
            sql: SQL SELECT statement (use $1, $2 for params)
            params: Query parameters

        Returns:
            QueryResult with rows as list of dicts
        """
        self._ensure_connected()
        assert self._pool is not None

        normalized = self._normalize_params(params)

        if self._conn is not None:
            # Use transaction connection
            rows = await self._conn.fetch(sql, *normalized)
        else:
            # Use pool connection
            rows = await self._pool.fetch(sql, *normalized)

        # Convert Record objects to dicts
        result_rows = [dict(row) for row in rows]
        columns = list(rows[0].keys()) if rows else []

        return QueryResult(
            rows=result_rows,
            row_count=len(result_rows),
            columns=columns,
            last_insert_id=None,
            affected_rows=0,
        )

    async def execute(self, sql: str, params: Params = None) -> QueryResult:
        """Execute INSERT/UPDATE/DELETE statement.

        Args:
            sql: SQL statement (use $1, $2 for params)
            params: Query parameters

        Returns:
            QueryResult with affected_rows
        """
        self._ensure_connected()
        assert self._pool is not None

        normalized = self._normalize_params(params)

        if self._conn is not None:
            # Use transaction connection
            result = await self._conn.execute(sql, *normalized)
        else:
            # Use pool connection
            result = await self._pool.execute(sql, *normalized)

        # Parse affected rows from result string (e.g., "INSERT 0 1")
        affected = self._parse_affected_rows(result)

        return QueryResult(
            rows=[],
            row_count=affected,
            columns=[],
            last_insert_id=None,  # Use RETURNING clause to get this
            affected_rows=affected,
        )

    async def execute_many(
        self, sql: str, params_list: list[tuple[Any, ...] | dict[str, Any]]
    ) -> QueryResult:
        """Execute statement with multiple parameter sets.

        Uses executemany for batch efficiency.

        Args:
            sql: SQL statement
            params_list: List of parameter sets

        Returns:
            QueryResult with total affected_rows
        """
        self._ensure_connected()
        assert self._pool is not None

        # Normalize all parameter sets
        normalized = [self._normalize_params(p) for p in params_list]

        if self._conn is not None:
            # Use transaction connection
            await self._conn.executemany(sql, normalized)
        else:
            # Use pool connection
            async with self._pool.acquire() as conn:
                await conn.executemany(sql, normalized)

        return QueryResult(
            rows=[],
            row_count=len(params_list),
            columns=[],
            last_insert_id=None,
            affected_rows=len(params_list),
        )

    async def begin_transaction(self, isolation_level: str | None = None) -> None:
        """Begin a transaction with optional isolation level.

        Maps isolation levels:
            - read_uncommitted → READ UNCOMMITTED
            - read_committed → READ COMMITTED (default)
            - repeatable_read → REPEATABLE READ
            - serializable → SERIALIZABLE

        Args:
            isolation_level: Transaction isolation level
        """
        self._ensure_connected()
        assert self._pool is not None

        # Acquire a dedicated connection for the transaction
        self._conn = await self._pool.acquire()

        # Map isolation level
        pg_isolation = None
        if isolation_level:
            mapping = {
                "read_uncommitted": "read uncommitted",
                "read_committed": "read committed",
                "repeatable_read": "repeatable read",
                "serializable": "serializable",
                # SQLite modes map to closest equivalents
                "immediate": "serializable",
                "exclusive": "serializable",
            }
            pg_isolation = mapping.get(isolation_level.lower())

        # Start transaction
        if pg_isolation:
            await self._conn.execute(f"BEGIN TRANSACTION ISOLATION LEVEL {pg_isolation}")
        else:
            await self._conn.execute("BEGIN")

        self._in_transaction = True
        logger.debug(f"Started PostgreSQL transaction (isolation={isolation_level})")

    async def commit(self) -> None:
        """Commit the current transaction."""
        if self._conn is None:
            return

        await self._conn.execute("COMMIT")
        self._in_transaction = False

        # Release connection back to pool
        if self._pool is not None:
            await self._pool.release(self._conn)
        self._conn = None

        logger.debug("Committed PostgreSQL transaction")

    async def rollback(self) -> None:
        """Rollback the current transaction."""
        if self._conn is None:
            return

        try:
            await self._conn.execute("ROLLBACK")
        except Exception:
            pass  # Ignore errors on rollback

        self._in_transaction = False

        # Release connection back to pool
        if self._pool is not None:
            await self._pool.release(self._conn)
        self._conn = None

        logger.debug("Rolled back PostgreSQL transaction")

    async def execute_script(self, sql: str) -> None:
        """Execute multi-statement SQL script.

        Args:
            sql: Multi-statement SQL script
        """
        self._ensure_connected()
        assert self._pool is not None

        if self._conn is not None:
            await self._conn.execute(sql)
        else:
            await self._pool.execute(sql)

        logger.debug("Executed PostgreSQL SQL script")

    @property
    def in_transaction(self) -> bool:
        """Check if currently in a transaction."""
        return self._in_transaction

    def _ensure_connected(self) -> None:
        """Ensure database is connected."""
        if self._pool is None:
            raise RuntimeError("Not connected to database. Call connect() first.")

    def _normalize_params(self, params: Params) -> tuple[Any, ...]:
        """Normalize parameters to asyncpg format.

        asyncpg uses positional parameters passed as *args.

        Args:
            params: Query parameters

        Returns:
            Tuple of parameters
        """
        if params is None:
            return ()
        if isinstance(params, dict):
            # Convert dict to positional (order matters!)
            return tuple(params.values())
        if isinstance(params, list):
            return tuple(params)
        return params

    def _parse_affected_rows(self, result: str) -> int:
        """Parse affected row count from PostgreSQL result string.

        Result format: "COMMAND [OID] COUNT"
        Examples:
            - "INSERT 0 1" -> 1
            - "UPDATE 5" -> 5
            - "DELETE 3" -> 3

        Args:
            result: PostgreSQL command result string

        Returns:
            Number of affected rows
        """
        if not result:
            return 0

        parts = result.split()
        if len(parts) >= 2:
            try:
                return int(parts[-1])
            except ValueError:
                pass
        return 0
