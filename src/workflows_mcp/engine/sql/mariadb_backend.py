"""MariaDB/MySQL database backend implementation.

This module provides the MariaDB backend for the SQL executor, using
aiomysql for native async operation with connection pooling.

Features:
    - Native async driver (aiomysql)
    - Connection pooling with configurable size
    - SSL/TLS support
    - Compatible with MySQL 5.7+ and MariaDB 10.2+
    - Automatic reconnection on connection loss

Note:
    Requires the 'aiomysql' package: pip install workflows-mcp[mariadb]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .backend import ConnectionConfig, DatabaseBackendBase, DatabaseDialect, Params, QueryResult

if TYPE_CHECKING:
    import aiomysql  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


def _import_aiomysql() -> Any:
    """Import aiomysql with helpful error message if not installed."""
    try:
        import aiomysql

        return aiomysql
    except ImportError as e:
        raise ImportError(
            "MariaDB backend requires 'aiomysql' package. "
            "Install with: pip install workflows-mcp[mariadb]"
        ) from e


class MariaDBBackend(DatabaseBackendBase):
    """MariaDB/MySQL backend using aiomysql with connection pooling.

    This backend uses the aiomysql library for native async MariaDB/MySQL
    access with connection pooling. It provides high-performance database
    access compatible with both MariaDB and MySQL servers.

    Attributes:
        dialect: DatabaseDialect.MARIADB

    Example:
        backend = MariaDBBackend()
        await backend.connect(ConnectionConfig(
            dialect=DatabaseDialect.MARIADB,
            host="localhost",
            port=3306,
            database="mydb",
            username="user",
            password="pass"
        ))
        result = await backend.query("SELECT * FROM users WHERE id = %s", (42,))
        await backend.disconnect()
    """

    dialect = DatabaseDialect.MARIADB

    def __init__(self) -> None:
        """Initialize MariaDB backend."""
        self._pool: aiomysql.Pool | None = None
        self._conn: aiomysql.Connection | None = None
        self._in_transaction: bool = False
        self._config: ConnectionConfig | None = None

    async def connect(self, config: ConnectionConfig) -> None:
        """Create connection pool.

        Pool settings:
            - minsize: 1
            - maxsize: config.pool_size
            - pool_recycle: 300s (prevent stale connections)
            - connect_timeout: config.connect_timeout

        Args:
            config: Connection configuration

        Raises:
            SqlConnectionError: If connection fails
            ImportError: If aiomysql is not installed
        """
        aiomysql = _import_aiomysql()
        self._config = config

        # Build SSL context if needed
        ssl_context = None
        if config.ssl:
            if config.ssl is True:
                # aiomysql accepts True for SSL with default settings
                ssl_context = True
            elif isinstance(config.ssl, str):
                # SSL mode string - MariaDB uses simpler SSL options
                ssl_context = True

        # Create connection pool
        self._pool = await aiomysql.create_pool(
            host=config.host,
            port=config.port or 3306,
            db=config.database,
            user=config.username,
            password=config.password or "",
            ssl=ssl_context,
            minsize=1,
            maxsize=config.pool_size,
            pool_recycle=300,
            connect_timeout=config.connect_timeout,
            autocommit=True,  # Default to autocommit, explicit transactions override
        )

        logger.debug(f"Connected to MariaDB: {config.host}:{config.port}/{config.database}")

    async def disconnect(self) -> None:
        """Close connection pool gracefully.

        Waits for all connections to be released before closing.
        """
        if self._conn is not None:
            self._conn.close()
            self._conn = None

        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

        self._in_transaction = False
        logger.debug("Disconnected from MariaDB")

    async def query(self, sql: str, params: Params = None) -> QueryResult:
        """Execute SELECT query and return results.

        Uses DictCursor for dict-based row results.

        Args:
            sql: SQL SELECT statement (use %s for params)
            params: Query parameters

        Returns:
            QueryResult with rows as list of dicts
        """
        aiomysql = _import_aiomysql()
        self._ensure_connected()
        assert self._pool is not None

        normalized = self._normalize_params(params)

        if self._conn is not None:
            # Use transaction connection
            async with self._conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(sql, normalized)
                rows = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
        else:
            # Use pool connection
            async with self._pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(sql, normalized)
                    rows = await cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []

        return QueryResult(
            rows=list(rows),
            row_count=len(rows),
            columns=columns,
            last_insert_id=None,
            affected_rows=0,
        )

    async def execute(self, sql: str, params: Params = None) -> QueryResult:
        """Execute INSERT/UPDATE/DELETE statement.

        Args:
            sql: SQL statement (use %s for params)
            params: Query parameters

        Returns:
            QueryResult with affected_rows and last_insert_id
        """
        aiomysql = _import_aiomysql()
        self._ensure_connected()
        assert self._pool is not None

        normalized = self._normalize_params(params)

        if self._conn is not None:
            # Use transaction connection
            async with self._conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(sql, normalized)
                affected = cursor.rowcount
                last_id = cursor.lastrowid
        else:
            # Use pool connection with autocommit
            async with self._pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(sql, normalized)
                    affected = cursor.rowcount
                    last_id = cursor.lastrowid

        return QueryResult(
            rows=[],
            row_count=affected,
            columns=[],
            last_insert_id=last_id,
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
        aiomysql = _import_aiomysql()
        self._ensure_connected()
        assert self._pool is not None

        # Normalize all parameter sets
        normalized = [self._normalize_params(p) for p in params_list]

        if self._conn is not None:
            # Use transaction connection
            async with self._conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.executemany(sql, normalized)
                affected = cursor.rowcount
        else:
            # Use pool connection
            async with self._pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.executemany(sql, normalized)
                    affected = cursor.rowcount
                    await conn.commit()

        return QueryResult(
            rows=[],
            row_count=len(params_list),
            columns=[],
            last_insert_id=None,
            affected_rows=affected if affected >= 0 else len(params_list),
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

        # Set isolation level if specified
        if isolation_level:
            mapping = {
                "read_uncommitted": "READ UNCOMMITTED",
                "read_committed": "READ COMMITTED",
                "repeatable_read": "REPEATABLE READ",
                "serializable": "SERIALIZABLE",
                # SQLite modes map to closest equivalents
                "immediate": "SERIALIZABLE",
                "exclusive": "SERIALIZABLE",
            }
            mysql_isolation = mapping.get(isolation_level.lower())
            if mysql_isolation:
                await self._conn.execute_query(f"SET TRANSACTION ISOLATION LEVEL {mysql_isolation}")

        # Start transaction
        await self._conn.begin()
        self._in_transaction = True
        logger.debug(f"Started MariaDB transaction (isolation={isolation_level})")

    async def commit(self) -> None:
        """Commit the current transaction."""
        if self._conn is None:
            return

        await self._conn.commit()
        self._in_transaction = False

        # Release connection back to pool
        if self._pool is not None:
            self._pool.release(self._conn)
        self._conn = None

        logger.debug("Committed MariaDB transaction")

    async def rollback(self) -> None:
        """Rollback the current transaction."""
        if self._conn is None:
            return

        try:
            await self._conn.rollback()
        except Exception:
            pass  # Ignore errors on rollback

        self._in_transaction = False

        # Release connection back to pool
        if self._pool is not None:
            self._pool.release(self._conn)
        self._conn = None

        logger.debug("Rolled back MariaDB transaction")

    async def execute_script(self, sql: str) -> None:
        """Execute multi-statement SQL script.

        MariaDB/MySQL supports multiple statements in a single execute
        when the connection is configured with client_flag for multi-statements.

        For safety, this method executes statements individually.

        Args:
            sql: Multi-statement SQL script (semicolon-separated)
        """
        aiomysql = _import_aiomysql()
        self._ensure_connected()
        assert self._pool is not None

        # Split and execute statements individually
        statements = [s.strip() for s in sql.split(";") if s.strip()]

        if self._conn is not None:
            async with self._conn.cursor(aiomysql.DictCursor) as cursor:
                for stmt in statements:
                    await cursor.execute(stmt)
        else:
            async with self._pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    for stmt in statements:
                        await cursor.execute(stmt)
                await conn.commit()

        logger.debug("Executed MariaDB SQL script")

    @property
    def in_transaction(self) -> bool:
        """Check if currently in a transaction."""
        return self._in_transaction

    def _ensure_connected(self) -> None:
        """Ensure database is connected."""
        if self._pool is None:
            raise RuntimeError("Not connected to database. Call connect() first.")

    def _normalize_params(self, params: Params) -> tuple[Any, ...] | dict[str, Any]:
        """Normalize parameters to aiomysql format.

        aiomysql accepts both tuples and dicts for parameters.
        %s for positional, %(name)s for named.

        Args:
            params: Query parameters

        Returns:
            Tuple or dict of parameters
        """
        if params is None:
            return ()
        if isinstance(params, list):
            return tuple(params)
        return params
