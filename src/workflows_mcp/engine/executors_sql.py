"""SQL executor for database operations across multiple backends.

Architecture (ADR-006):
- Execute returns output directly (no Result wrapper)
- Raises exceptions for failures
- Uses Execution context (not dict)
- Orchestrator creates Metadata based on success/exceptions

Features:
- Multiple database backends: SQLite, PostgreSQL, MariaDB
- Automatic parameter placeholder conversion
- Connection pooling for remote databases
- Transaction support with isolation levels
- Schema management (CREATE TABLE IF NOT EXISTS)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field, field_validator, model_validator

if TYPE_CHECKING:
    from typing import Self

from .block import BlockInput, BlockOutput
from .execution import Execution
from .executor_base import (
    BlockExecutor,
    ExecutorCapabilities,
    ExecutorSecurityLevel,
)
from .interpolation import (
    interpolatable_numeric_validator,
)
from .sql import (
    ConnectionConfig,
    DatabaseBackendBase,
    DatabaseDialect,
    MariaDBBackend,
    ParamConverter,
    PostgresBackend,
    QueryResult,
    SqliteBackend,
)

logger = logging.getLogger(__name__)


# ============================================================================
# SQL Executor Exceptions
# ============================================================================


class SqlError(Exception):
    """Base exception for SQL executor errors."""

    pass


class SqlConnectionError(SqlError):
    """Failed to establish database connection."""

    pass


class SqlQueryError(SqlError):
    """SQL execution failed."""

    pass


class SqlSchemaError(SqlError):
    """Schema creation/validation failed."""

    pass


class SqlTimeoutError(SqlError):
    """Query exceeded timeout limit."""

    pass


# ============================================================================
# SQL Executor Input/Output Models
# ============================================================================


class SqlInput(BlockInput):
    """Input schema for SQL executor."""

    # ═══════════════════════════════════════════════════════════════════
    # Connection (structured parameters)
    # ═══════════════════════════════════════════════════════════════════

    dialect: Literal["sqlite", "postgresql", "mariadb"] = Field(
        description="Database dialect. Required."
    )

    # SQLite-specific
    path: str | None = Field(
        default=None, description="SQLite: Database file path. Use ':memory:' for in-memory DB."
    )

    # Remote database connection (PostgreSQL/MariaDB)
    host: str | None = Field(default=None, description="Database host")
    port: int | str | None = Field(
        default=None, description="Database port (default: 5432 for PostgreSQL, 3306 for MariaDB)"
    )
    database: str | None = Field(default=None, description="Database name")
    username: str | None = Field(default=None, description="Database username")
    password: str | None = Field(
        default=None, description="Database password. Use {{secrets.DB_PASSWORD}} for security."
    )

    # ═══════════════════════════════════════════════════════════════════
    # Connection Options
    # ═══════════════════════════════════════════════════════════════════

    ssl: bool | str = Field(
        default=False,
        description="Enable SSL/TLS. Boolean or sslmode string (require, verify-ca, verify-full)",
    )
    timeout: int | str = Field(
        default=30,
        description="Query execution timeout in seconds",
    )
    connect_timeout: int | str = Field(
        default=10,
        description="Connection establishment timeout in seconds",
    )
    pool_size: int | str = Field(
        default=5,
        description="Connection pool size (PostgreSQL/MariaDB only)",
    )

    # ═══════════════════════════════════════════════════════════════════
    # Operation
    # ═══════════════════════════════════════════════════════════════════

    operation: Literal[
        "query",  # SELECT - returns rows
        "execute",  # INSERT/UPDATE/DELETE - returns affected count
        "execute_many",  # Batch execute with multiple param sets
        "transaction",  # Multiple statements in atomic transaction
        "script",  # Execute multi-statement SQL script
    ] = Field(description="SQL operation type")

    sql: str = Field(
        description="""
        SQL statement(s) to execute.
        - Use ? for positional params (SQLite) or $1, $2 for PostgreSQL
        - MariaDB uses %s for positional params
        - For 'transaction' and 'script': separate statements with semicolons
        """
    )

    params: list[Any] | dict[str, Any] | None = Field(
        default=None,
        description="""
        Query parameters (prevents SQL injection).
        - List for positional: [value1, value2]
        - Dict for named: {"name": value} (PostgreSQL/MariaDB)
        - For execute_many: list of param lists [[v1, v2], [v3, v4]]
        - For transaction: list of param lists, one per statement
        """,
    )

    # ═══════════════════════════════════════════════════════════════════
    # Schema Management
    # ═══════════════════════════════════════════════════════════════════

    init_sql: str | None = Field(
        default=None,
        description="""
        DDL to execute before the main operation (idempotent).
        Use CREATE TABLE IF NOT EXISTS, CREATE INDEX IF NOT EXISTS, etc.
        Executed once per connection, before the main SQL.
        """,
    )

    # ═══════════════════════════════════════════════════════════════════
    # Transaction Options
    # ═══════════════════════════════════════════════════════════════════

    isolation_level: (
        Literal[
            "default",
            "read_uncommitted",
            "read_committed",
            "repeatable_read",
            "serializable",
            "immediate",  # SQLite-specific: acquire write lock early
            "exclusive",  # SQLite-specific: exclusive lock
        ]
        | None
    ) = Field(
        default=None,
        description="""
        Transaction isolation level.
        - PostgreSQL/MariaDB: read_uncommitted, read_committed, repeatable_read, serializable
        - SQLite: immediate (recommended for writes), exclusive, or default (deferred)
        """,
    )

    # ═══════════════════════════════════════════════════════════════════
    # SQLite-Specific Options
    # ═══════════════════════════════════════════════════════════════════

    sqlite_pragmas: dict[str, str | int | bool] | None = Field(
        default=None,
        description="""
        SQLite PRAGMA settings applied on connection.
        Defaults (if not specified):
          journal_mode: WAL
          busy_timeout: 30000
          synchronous: NORMAL
          foreign_keys: ON
        Example: {"cache_size": -64000, "temp_store": "MEMORY"}
        """,
    )

    # Validators for numeric fields with interpolation support
    _validate_timeout = field_validator("timeout", mode="before")(
        interpolatable_numeric_validator(int, ge=1, le=3600)
    )
    _validate_connect_timeout = field_validator("connect_timeout", mode="before")(
        interpolatable_numeric_validator(int, ge=1, le=300)
    )
    _validate_pool_size = field_validator("pool_size", mode="before")(
        interpolatable_numeric_validator(int, ge=1, le=100)
    )

    @field_validator("port", mode="before")
    @classmethod
    def _validate_port(cls, v: Any) -> int | str | None:
        """Validate port, allowing None."""
        if v is None:
            return None
        if isinstance(v, str) and "{{" in v:
            return v  # Allow interpolation strings
        try:
            port = int(v)
            if not 1 <= port <= 65535:
                raise ValueError("port must be between 1 and 65535")
            return port
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid port value: {v}") from e

    @model_validator(mode="after")
    def validate_connection_params(self) -> Self:
        """Validate connection parameters based on dialect."""
        if self.dialect == "sqlite":
            if not self.path:
                raise ValueError("SQLite requires 'path' parameter")
        else:
            if not self.host:
                raise ValueError(f"{self.dialect} requires 'host' parameter")
            if not self.database:
                raise ValueError(f"{self.dialect} requires 'database' parameter")
        return self


class SqlOutput(BlockOutput):
    """Output from SQL executor."""

    # Query results
    rows: list[dict[str, Any]] = Field(
        default_factory=list, description="Result rows as list of dicts (query operation)"
    )
    columns: list[str] = Field(default_factory=list, description="Column names from result set")

    # Counts
    row_count: int = Field(
        default=0, description="Number of rows returned (query) or total affected (execute)"
    )
    affected_rows: int = Field(default=0, description="Rows affected by INSERT/UPDATE/DELETE")

    # Insert tracking
    last_insert_id: int | None = Field(
        default=None, description="Last inserted row ID (auto-increment)"
    )

    # Metadata
    success: bool = Field(default=True, description="Operation completed successfully")
    dialect: str = Field(default="", description="Database dialect used")
    execution_time_ms: float = Field(
        default=0.0, description="Query execution time in milliseconds"
    )


# ============================================================================
# SQL Executor
# ============================================================================


class SqlExecutor(BlockExecutor):
    """SQL executor for database operations.

    Architecture (ADR-006):
    - Returns SqlOutput directly
    - Raises exceptions for database failures
    - Uses Execution context

    Features:
    - Multiple database backends: SQLite, PostgreSQL, MariaDB
    - Automatic parameter placeholder conversion
    - Connection pooling for remote databases
    - Transaction support with isolation levels
    - Schema management (CREATE TABLE IF NOT EXISTS)

    Security:
    - All queries use parameterized statements (no SQL injection)
    - Passwords via {{secrets.VAR}} are never logged
    - Connection strings in error messages redact credentials
    - SSL/TLS support for remote databases
    """

    type_name: ClassVar[str] = "Sql"
    input_type: ClassVar[type[BlockInput]] = SqlInput
    output_type: ClassVar[type[BlockOutput]] = SqlOutput
    examples: ClassVar[str] = """```yaml
# SQLite query
- id: get_users
  type: Sql
  inputs:
    dialect: sqlite
    path: "/data/app.db"
    operation: query
    sql: "SELECT * FROM users WHERE status = ?"
    params: ["active"]

# PostgreSQL with init_sql for schema setup
- id: setup_and_query
  type: Sql
  inputs:
    dialect: postgresql
    host: "{{secrets.DB_HOST}}"
    database: mydb
    username: "{{secrets.DB_USER}}"
    password: "{{secrets.DB_PASS}}"
    operation: query
    init_sql: |
      CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL
      )
    sql: "SELECT * FROM users LIMIT $1"
    params: [10]
```"""

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(
        can_network=True, can_write_files=True
    )

    async def execute(  # type: ignore[override]
        self, inputs: SqlInput, context: Execution
    ) -> SqlOutput:
        """Execute SQL operation.

        Returns:
            SqlOutput with rows, affected counts, and metadata

        Raises:
            SqlConnectionError: Connection failures
            SqlQueryError: SQL execution errors
            SqlSchemaError: Schema creation failures
            SqlTimeoutError: Query timeout
        """
        start_time = time.perf_counter()

        # Resolve interpolatable fields
        timeout = int(inputs.timeout) if isinstance(inputs.timeout, int) else 30
        connect_timeout = (
            int(inputs.connect_timeout) if isinstance(inputs.connect_timeout, int) else 10
        )
        pool_size = int(inputs.pool_size) if isinstance(inputs.pool_size, int) else 5
        port = int(inputs.port) if inputs.port is not None else None

        # Create backend and config
        backend = self._create_backend(inputs.dialect)
        config = self._create_config(
            dialect=inputs.dialect,
            path=inputs.path,
            host=inputs.host,
            port=port,
            database=inputs.database,
            username=inputs.username,
            password=inputs.password,
            ssl=inputs.ssl,
            timeout=timeout,
            connect_timeout=connect_timeout,
            pool_size=pool_size,
            sqlite_pragmas=inputs.sqlite_pragmas,
        )

        # Convert SQL placeholders for target dialect
        converter = ParamConverter(backend.dialect)
        converted_sql = converter.convert(inputs.sql)
        converted_params = converter.convert_params(inputs.params)

        try:
            await backend.connect(config)

            # Execute init_sql (schema) if provided
            if inputs.init_sql:
                try:
                    await backend.execute_script(inputs.init_sql)
                except Exception as e:
                    raise SqlSchemaError(f"Schema execution failed: {e}") from e

            # Execute the main operation
            result = await self._execute_operation(
                backend=backend,
                operation=inputs.operation,
                sql=converted_sql,
                params=converted_params,
                isolation_level=inputs.isolation_level,
            )

            execution_time = (time.perf_counter() - start_time) * 1000

            return SqlOutput(
                rows=result.rows,
                columns=result.columns,
                row_count=result.row_count,
                affected_rows=result.affected_rows,
                last_insert_id=result.last_insert_id,
                success=True,
                dialect=inputs.dialect,
                execution_time_ms=execution_time,
            )

        except SqlError:
            raise
        except Exception as e:
            # Wrap unknown errors as SqlQueryError
            raise SqlQueryError(f"SQL operation failed: {e}") from e
        finally:
            await backend.disconnect()

    def _create_backend(self, dialect: str) -> DatabaseBackendBase:
        """Create appropriate backend for dialect."""
        if dialect == "sqlite":
            return SqliteBackend()
        elif dialect == "postgresql":
            if PostgresBackend is None:
                raise ImportError(
                    "PostgreSQL backend requires 'asyncpg' package. "
                    "Install with: pip install workflows-mcp[postgresql]"
                )
            return PostgresBackend()
        elif dialect == "mariadb":
            if MariaDBBackend is None:
                raise ImportError(
                    "MariaDB backend requires 'aiomysql' package. "
                    "Install with: pip install workflows-mcp[mariadb]"
                )
            return MariaDBBackend()
        else:
            raise ValueError(f"Unsupported dialect: {dialect}")

    def _create_config(
        self,
        dialect: str,
        path: str | None,
        host: str | None,
        port: int | None,
        database: str | None,
        username: str | None,
        password: str | None,
        ssl: bool | str,
        timeout: int,
        connect_timeout: int,
        pool_size: int,
        sqlite_pragmas: dict[str, str | int | bool] | None,
    ) -> ConnectionConfig:
        """Create connection configuration."""
        dialect_enum = DatabaseDialect(dialect)

        options: dict[str, Any] = {}
        if sqlite_pragmas:
            options["sqlite_pragmas"] = sqlite_pragmas

        return ConnectionConfig(
            dialect=dialect_enum,
            path=path,
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            ssl=ssl,
            timeout=timeout,
            connect_timeout=connect_timeout,
            pool_size=pool_size,
            options=options,
        )

    async def _execute_operation(
        self,
        backend: DatabaseBackendBase,
        operation: str,
        sql: str,
        params: Any,
        isolation_level: str | None,
    ) -> QueryResult:
        """Execute the SQL operation based on operation type."""
        if operation == "query":
            return await backend.query(sql, params)

        elif operation == "execute":
            return await backend.execute(sql, params)

        elif operation == "execute_many":
            if not isinstance(params, list):
                raise ValueError("execute_many requires params to be a list of parameter sets")
            return await backend.execute_many(sql, params)

        elif operation == "transaction":
            return await self._execute_transaction(backend, sql, params, isolation_level)

        elif operation == "script":
            await backend.execute_script(sql)
            return QueryResult()

        else:
            raise ValueError(f"Unknown operation: {operation}")

    async def _execute_transaction(
        self,
        backend: DatabaseBackendBase,
        sql: str,
        params: Any,
        isolation_level: str | None,
    ) -> QueryResult:
        """Execute multiple statements in a transaction."""
        # Split SQL into statements
        statements = [s.strip() for s in sql.split(";") if s.strip()]

        # Prepare params for each statement
        if params is None:
            params_list = [None] * len(statements)
        elif isinstance(params, list) and len(params) == len(statements):
            params_list = params
        else:
            # Use same params for all statements
            params_list = [params] * len(statements)

        # Determine effective isolation level
        effective_isolation = None
        if isolation_level and isolation_level != "default":
            effective_isolation = isolation_level

        total_affected = 0
        last_result = QueryResult()

        await backend.begin_transaction(effective_isolation)
        try:
            for stmt, stmt_params in zip(statements, params_list):
                # Detect if SELECT
                if stmt.strip().upper().startswith("SELECT"):
                    last_result = await backend.query(stmt, stmt_params)
                else:
                    result = await backend.execute(stmt, stmt_params)
                    total_affected += result.affected_rows
                    if result.last_insert_id is not None:
                        last_result = result

            await backend.commit()
        except Exception:
            await backend.rollback()
            raise

        return QueryResult(
            rows=last_result.rows,
            columns=last_result.columns,
            row_count=last_result.row_count or total_affected,
            affected_rows=total_affected,
            last_insert_id=last_result.last_insert_id,
        )
