"""SQLite database backend implementation.

This module provides the SQLite backend for the SQL executor, using the
stdlib sqlite3 module with asyncio run_in_executor for async operation.

Features:
    - WAL mode by default for concurrent reads
    - Automatic busy_timeout for lock contention handling
    - Foreign key enforcement enabled
    - Path validation and parent directory creation
    - PRAGMA configuration via options
    - sqlite-vec extension for vector similarity search
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Any

import sqlite_vec  # type: ignore[import-untyped]

from .backend import ConnectionConfig, DatabaseBackendBase, DatabaseEngine, Params, QueryResult

logger = logging.getLogger(__name__)


class SqliteBackend(DatabaseBackendBase):
    """SQLite backend using stdlib sqlite3 with async executor.

    This backend wraps the synchronous sqlite3 module in asyncio's
    run_in_executor to provide async operation. It configures SQLite
    for optimal performance with WAL mode and appropriate timeouts.

    Attributes:
        dialect: DatabaseEngine.SQLITE
        DEFAULT_PRAGMAS: Default PRAGMA settings applied on connection

    Example:
        backend = SqliteBackend()
        await backend.connect(ConnectionConfig(
            dialect=DatabaseEngine.SQLITE,
            path="/data/app.db"
        ))
        result = await backend.query("SELECT * FROM users WHERE id = ?", (42,))
        await backend.disconnect()
    """

    dialect = DatabaseEngine.SQLITE

    DEFAULT_PRAGMAS: dict[str, str | int] = {
        "journal_mode": "WAL",
        "busy_timeout": 30000,
        "synchronous": "NORMAL",
        "foreign_keys": "ON",
    }

    def __init__(self) -> None:
        """Initialize SQLite backend."""
        self._conn: sqlite3.Connection | None = None
        self._in_transaction: bool = False
        self._config: ConnectionConfig | None = None

    async def connect(self, config: ConnectionConfig) -> None:
        """Connect to SQLite database.

        Creates the database file and parent directories if they don't exist.
        Applies PRAGMA settings from config.options or defaults.

        Args:
            config: Connection configuration with path

        Raises:
            SqlConnectionError: If connection fails
        """
        self._config = config

        def _connect() -> sqlite3.Connection:
            path = config.path
            if path is None:
                raise ValueError("SQLite requires 'path' parameter")

            # Handle special paths
            if path != ":memory:" and not path.startswith(":"):
                # Ensure parent directory exists
                db_path = Path(path)
                db_path.parent.mkdir(parents=True, exist_ok=True)

            # Connect with check_same_thread=False for async usage
            conn = sqlite3.connect(path, check_same_thread=False)
            conn.row_factory = sqlite3.Row

            # Load sqlite-vec extension for vector similarity search
            # This must be done before any PRAGMA settings
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)  # Disable for security
                logger.debug("Loaded sqlite-vec extension")
            except Exception as e:
                logger.warning(f"Failed to load sqlite-vec extension: {e}")

            # Get PRAGMA settings from options or use defaults
            pragmas = {**self.DEFAULT_PRAGMAS}
            if config.options.get("sqlite_pragmas"):
                pragmas.update(config.options["sqlite_pragmas"])

            # Apply timeout from config if provided
            if config.timeout:
                pragmas["busy_timeout"] = config.timeout * 1000  # Convert to ms

            # Apply PRAGMA settings
            for pragma, value in pragmas.items():
                try:
                    conn.execute(f"PRAGMA {pragma}={value}")
                except sqlite3.Error as e:
                    logger.warning(f"Failed to set PRAGMA {pragma}={value}: {e}")

            logger.debug(f"Connected to SQLite database: {path}")
            return conn

        loop = asyncio.get_event_loop()
        self._conn = await loop.run_in_executor(None, _connect)

    async def disconnect(self) -> None:
        """Close SQLite connection.

        Safe to call multiple times or if not connected.
        """
        if self._conn is None:
            return

        def _close() -> None:
            if self._conn:
                self._conn.close()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _close)
        self._conn = None
        self._in_transaction = False
        logger.debug("Disconnected from SQLite database")

    async def query(self, sql: str, params: Params = None) -> QueryResult:
        """Execute SELECT query and return results.

        Args:
            sql: SQL SELECT statement
            params: Query parameters (tuple, list, or dict)

        Returns:
            QueryResult with rows as list of dicts

        Raises:
            SqlQueryError: If query execution fails
        """
        self._ensure_connected()

        def _query() -> QueryResult:
            assert self._conn is not None
            cursor = self._conn.execute(sql, self._normalize_params(params))

            # Convert Row objects to dicts
            rows = [dict(row) for row in cursor.fetchall()]
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            return QueryResult(
                rows=rows,
                row_count=len(rows),
                columns=columns,
                last_insert_id=None,
                affected_rows=0,
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _query)

    async def execute(self, sql: str, params: Params = None) -> QueryResult:
        """Execute INSERT/UPDATE/DELETE statement.

        Auto-commits unless in an explicit transaction.
        Handles RETURNING clause by fetching result rows.

        Args:
            sql: SQL statement
            params: Query parameters

        Returns:
            QueryResult with affected_rows, last_insert_id, and rows (if RETURNING)

        Raises:
            SqlQueryError: If execution fails
        """
        self._ensure_connected()

        def _execute() -> QueryResult:
            assert self._conn is not None
            cursor = self._conn.execute(sql, self._normalize_params(params))

            # Check for RETURNING clause - fetch rows if present
            has_returning = "RETURNING" in sql.upper()
            if has_returning and cursor.description:
                rows = [dict(row) for row in cursor.fetchall()]
                columns = [desc[0] for desc in cursor.description]
            else:
                rows = []
                columns = []

            # Auto-commit if not in transaction
            if not self._in_transaction:
                self._conn.commit()

            return QueryResult(
                rows=rows,
                row_count=len(rows) if rows else cursor.rowcount,
                columns=columns,
                last_insert_id=cursor.lastrowid,
                affected_rows=cursor.rowcount,
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _execute)

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

        def _execute_many() -> QueryResult:
            assert self._conn is not None

            # Normalize all parameter sets
            normalized = [self._normalize_params(p) for p in params_list]
            cursor = self._conn.executemany(sql, normalized)

            if not self._in_transaction:
                self._conn.commit()

            return QueryResult(
                rows=[],
                row_count=cursor.rowcount,
                columns=[],
                last_insert_id=cursor.lastrowid,
                affected_rows=cursor.rowcount,
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _execute_many)

    async def begin_transaction(self, isolation_level: str | None = None) -> None:
        """Begin a transaction.

        SQLite supports special isolation modes:
            - None/default: DEFERRED (acquire lock on first write)
            - immediate: BEGIN IMMEDIATE (acquire write lock immediately)
            - exclusive: BEGIN EXCLUSIVE (block all other connections)

        The 'immediate' mode is recommended for atomic read-modify-write
        operations to prevent race conditions.

        Args:
            isolation_level: Transaction isolation level
        """
        self._ensure_connected()

        def _begin() -> None:
            assert self._conn is not None

            # Map isolation level to SQLite BEGIN mode
            mode = ""
            if isolation_level:
                level = isolation_level.lower()
                if level in ("immediate", "exclusive", "deferred"):
                    mode = f" {level.upper()}"
                elif level in ("serializable",):
                    mode = " IMMEDIATE"  # Closest equivalent
                # read_committed, repeatable_read, etc. use default (DEFERRED)

            self._conn.execute(f"BEGIN{mode}")
            self._in_transaction = True

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _begin)
        logger.debug(f"Started transaction (isolation={isolation_level})")

    async def commit(self) -> None:
        """Commit the current transaction."""
        self._ensure_connected()

        def _commit() -> None:
            assert self._conn is not None
            self._conn.commit()
            self._in_transaction = False

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _commit)
        logger.debug("Committed transaction")

    async def rollback(self) -> None:
        """Rollback the current transaction.

        Safe to call even if no transaction is active.
        """
        if self._conn is None:
            return

        def _rollback() -> None:
            assert self._conn is not None
            try:
                self._conn.rollback()
            except sqlite3.Error:
                pass  # Ignore errors on rollback
            self._in_transaction = False

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _rollback)
        logger.debug("Rolled back transaction")

    async def execute_script(self, sql: str) -> None:
        """Execute multi-statement SQL script.

        Uses sqlite3.executescript() which commits any pending transaction,
        executes the script, and implicitly commits.

        Args:
            sql: Multi-statement SQL script

        Raises:
            SqlSchemaError: If script execution fails
        """
        self._ensure_connected()

        def _execute_script() -> None:
            assert self._conn is not None
            self._conn.executescript(sql)
            self._in_transaction = False  # executescript commits

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _execute_script)
        logger.debug("Executed SQL script")

    @property
    def in_transaction(self) -> bool:
        """Check if currently in a transaction."""
        return self._in_transaction

    def _ensure_connected(self) -> None:
        """Ensure database is connected.

        Raises:
            RuntimeError: If not connected
        """
        if self._conn is None:
            raise RuntimeError("Not connected to database. Call connect() first.")

    def _normalize_params(self, params: Params) -> tuple[Any, ...] | dict[str, Any]:
        """Normalize parameters to sqlite3-compatible format.

        Args:
            params: Query parameters in various formats

        Returns:
            Tuple or dict suitable for sqlite3
        """
        if params is None:
            return ()
        if isinstance(params, dict):
            return params
        if isinstance(params, list):
            return tuple(params)
        return params
