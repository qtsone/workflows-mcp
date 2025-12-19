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
    DatabaseEngine,
    MariaDBBackend,
    ModelSchema,
    ParamConverter,
    PostgresBackend,
    QueryBuilder,
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
    """Input schema for SQL executor.

    Two operating modes (mutually exclusive):

    **Mode A: Raw SQL** - Provide `sql` field to execute SQL directly
    **Mode B: Model-based CRUD** - Provide `model` + `op` for Active Record-style operations
    """

    # ═══════════════════════════════════════════════════════════════════
    # CONNECTION (required)
    # ═══════════════════════════════════════════════════════════════════

    engine: Literal["sqlite", "postgresql", "mariadb"] = Field(
        description="Database engine. Required."
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
    # MODE A: Raw SQL (mutually exclusive with model)
    # ═══════════════════════════════════════════════════════════════════

    sql: str | None = Field(
        default=None,
        description="""
        SQL statement(s) to execute (Raw SQL mode).
        - Use ? for positional params (SQLite) or $1, $2 for PostgreSQL
        - MariaDB uses %s for positional params
        - Multi-statement scripts: separate with semicolons
        Mutually exclusive with 'model' field.
        """,
    )

    params: list[Any] | dict[str, Any] | None = Field(
        default=None,
        description="""
        Query parameters for raw SQL (prevents SQL injection).
        - List for positional: [value1, value2]
        - Dict for named: {"name": value} (PostgreSQL/MariaDB)
        """,
    )

    # ═══════════════════════════════════════════════════════════════════
    # MODE B: Model-based CRUD (mutually exclusive with sql)
    # ═══════════════════════════════════════════════════════════════════

    model: dict[str, Any] | None = Field(
        default=None,
        description="""
        Model schema for CRUD operations (Model mode).
        Defines table structure with columns, types, indexes.
        Mutually exclusive with 'sql' field.
        Example:
          model:
            table: tasks
            columns:
              id: {type: text, primary: true, auto: uuid}
              name: {type: text, required: true}
            indexes:
              - columns: [name]
        """,
    )

    op: Literal["schema", "insert", "select", "update", "delete", "upsert"] | None = Field(
        default=None,
        description="""
        CRUD operation (required when using model mode).
        - schema: Create table + indexes
        - insert: Insert row (requires data)
        - select: Query rows (optional where, order, limit, offset)
        - update: Update rows (requires where and data)
        - delete: Delete rows (requires where)
        - upsert: Insert or update on conflict (requires data and conflict)
        """,
    )

    data: dict[str, Any] | None = Field(
        default=None,
        description="Row data for insert/update/upsert operations.",
    )

    where: dict[str, Any] | None = Field(
        default=None,
        description="""
        Filter conditions for select/update/delete.
        - Simple equality: {status: running}
        - Operators: {priority: {">": 5}}
        - IN: {type: {in: [a, b, c]}}
        - IS NULL: {deleted_at: {is: null}}
        """,
    )

    order: list[str] | None = Field(
        default=None,
        description='Sort order for select. Format: ["column:asc", "column:desc"]',
    )

    limit: int | str | None = Field(
        default=None,
        description="Maximum rows to return (select).",
    )

    offset: int | str | None = Field(
        default=None,
        description="Rows to skip (select).",
    )

    conflict: list[str] | None = Field(
        default=None,
        description="Conflict columns for upsert (usually primary key).",
    )

    # ═══════════════════════════════════════════════════════════════════
    # OPTIONS (both modes)
    # ═══════════════════════════════════════════════════════════════════

    init_sql: str | None = Field(
        default=None,
        description="""
        DDL to execute before the main operation (idempotent).
        Use CREATE TABLE IF NOT EXISTS, CREATE INDEX IF NOT EXISTS, etc.
        """,
    )

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
    # CONNECTION OPTIONS
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
    # SQLite-Specific Options
    # ═══════════════════════════════════════════════════════════════════

    sqlite_pragmas: dict[str, str | int | bool] | None = Field(
        default=None,
        description="""
        SQLite PRAGMA settings applied on connection.
        Defaults: journal_mode=WAL, busy_timeout=30000, synchronous=NORMAL, foreign_keys=ON
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

    @field_validator("limit", "offset", mode="before")
    @classmethod
    def _validate_limit_offset(cls, v: Any) -> int | str | None:
        """Validate limit/offset, allowing interpolation."""
        if v is None:
            return None
        if isinstance(v, str) and "{{" in v:
            return v
        try:
            return int(v)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid value: {v}") from e

    @model_validator(mode="after")
    def validate_inputs(self) -> Self:
        """Validate connection and mode parameters."""
        # Connection validation
        if self.engine == "sqlite":
            if not self.path:
                raise ValueError("SQLite requires 'path' parameter")
        else:
            if not self.host:
                raise ValueError(f"{self.engine} requires 'host' parameter")
            if not self.database:
                raise ValueError(f"{self.engine} requires 'database' parameter")

        # Mode validation: sql XOR model
        has_sql = self.sql is not None
        has_model = self.model is not None

        if has_sql and has_model:
            raise ValueError("Cannot specify both 'sql' and 'model' - choose one mode")
        if not has_sql and not has_model:
            raise ValueError("Must specify either 'sql' (raw SQL mode) or 'model' (model mode)")

        # Model mode validation
        if has_model:
            if not self.op:
                raise ValueError(
                    "Model mode requires 'op' field (schema/insert/select/update/delete/upsert)"
                )

            if self.op == "insert" and not self.data:
                raise ValueError("insert operation requires 'data'")
            if self.op == "update" and (not self.where or not self.data):
                raise ValueError("update operation requires 'where' and 'data'")
            if self.op == "delete" and not self.where:
                raise ValueError("delete operation requires 'where'")
            if self.op == "upsert" and (not self.data or not self.conflict):
                raise ValueError("upsert operation requires 'data' and 'conflict'")

        return self


class SqlOutput(BlockOutput):
    """Output from SQL executor."""

    # Query results
    rows: list[dict[str, Any]] = Field(
        default_factory=list, description="Result rows as list of dicts"
    )
    columns: list[str] = Field(default_factory=list, description="Column names from result set")

    # Counts
    row_count: int = Field(
        default=0, description="Number of rows returned (select) or affected (insert/update/delete)"
    )
    affected_rows: int = Field(default=0, description="Rows affected by INSERT/UPDATE/DELETE")

    # Insert tracking
    last_insert_id: int | None = Field(
        default=None, description="Last inserted row ID (auto-increment)"
    )

    # Metadata
    success: bool = Field(default=True, description="Operation completed successfully")
    engine: str = Field(default="", description="Database engine used")
    execution_time_ms: float = Field(
        default=0.0, description="Query execution time in milliseconds"
    )


# ============================================================================
# SQL Executor
# ============================================================================


class SqlExecutor(BlockExecutor):
    """SQL executor for database operations.

    Two operating modes:
    - **Raw SQL mode**: Provide `sql` field to execute SQL directly
    - **Model mode**: Provide `model` + `op` for Active Record-style CRUD

    Architecture (ADR-006):
    - Returns SqlOutput directly
    - Raises exceptions for database failures
    - Uses Execution context

    Features:
    - Multiple database backends: SQLite, PostgreSQL, MariaDB
    - Automatic parameter placeholder conversion
    - Connection pooling for remote databases
    - Transaction support with isolation levels
    - Model-based CRUD with auto-generated SQL

    Security:
    - All queries use parameterized statements (no SQL injection)
    - Passwords via {{secrets.VAR}} are never logged
    - SSL/TLS support for remote databases
    """

    type_name: ClassVar[str] = "Sql"
    input_type: ClassVar[type[BlockInput]] = SqlInput
    output_type: ClassVar[type[BlockOutput]] = SqlOutput
    examples: ClassVar[str] = """```yaml
# Raw SQL mode - SQLite query
- id: get_users
  type: Sql
  inputs:
    engine: sqlite
    path: "/data/app.db"
    sql: "SELECT * FROM users WHERE status = ?"
    params: ["active"]

# Model mode - Create table and insert
- id: create_task
  type: Sql
  inputs:
    engine: sqlite
    path: "{{state.db_path}}"
    model:
      table: tasks
      columns:
        task_id: {type: text, primary: true, auto: uuid}
        name: {type: text, required: true}
        status: {type: text, default: pending}
        created_at: {type: timestamp, auto: created}
      indexes:
        - columns: [status]
    op: insert
    data:
      name: "My Task"

# Model mode - Select with filters
- id: find_tasks
  type: Sql
  inputs:
    engine: sqlite
    path: "{{state.db_path}}"
    model: "{{inputs.models.task}}"
    op: select
    where:
      status: running
    order: [created_at:desc]
    limit: 10
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
        backend = self._create_backend(inputs.engine)
        config = self._create_config(
            engine=inputs.engine,
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

        try:
            await backend.connect(config)

            # Execute init_sql (schema) if provided
            if inputs.init_sql:
                try:
                    await backend.execute_script(inputs.init_sql)
                except Exception as e:
                    raise SqlSchemaError(f"Schema execution failed: {e}") from e

            # Route to appropriate mode
            if inputs.sql is not None:
                # Raw SQL mode
                result = await self._execute_raw_sql(backend, inputs)
            else:
                # Model mode
                result = await self._execute_model_op(backend, inputs)

            execution_time = (time.perf_counter() - start_time) * 1000

            return SqlOutput(
                rows=result.rows,
                columns=result.columns,
                row_count=result.row_count,
                affected_rows=result.affected_rows,
                last_insert_id=result.last_insert_id,
                success=True,
                engine=inputs.engine,
                execution_time_ms=execution_time,
            )

        except SqlError:
            raise
        except Exception as e:
            # Wrap unknown errors as SqlQueryError
            raise SqlQueryError(f"SQL operation failed: {e}") from e
        finally:
            await backend.disconnect()

    async def _execute_raw_sql(self, backend: DatabaseBackendBase, inputs: SqlInput) -> QueryResult:
        """Execute raw SQL mode."""
        assert inputs.sql is not None

        # Convert SQL placeholders for target engine
        converter = ParamConverter(backend.dialect)
        converted_sql = converter.convert(inputs.sql)
        converted_params = converter.convert_params(inputs.params)

        # Detect operation type from SQL
        sql_upper = inputs.sql.strip().upper()
        if sql_upper.startswith("SELECT"):
            return await backend.query(converted_sql, converted_params)
        elif ";" in inputs.sql:
            # Multi-statement script
            await backend.execute_script(inputs.sql)
            return QueryResult()
        else:
            return await backend.execute(converted_sql, converted_params)

    async def _execute_model_op(
        self, backend: DatabaseBackendBase, inputs: SqlInput
    ) -> QueryResult:
        """Execute model-based CRUD operation."""
        assert inputs.model is not None
        assert inputs.op is not None

        # Parse model schema
        schema = ModelSchema.from_dict(inputs.model)
        engine_enum = DatabaseEngine(inputs.engine)
        builder = QueryBuilder(schema, engine_enum)

        if inputs.op == "schema":
            # Create table and indexes
            ddl = schema.to_create_sql(engine_enum)
            await backend.execute_script(ddl)
            for index_sql in schema.to_index_sql(engine_enum):
                await backend.execute_script(index_sql)
            return QueryResult(row_count=1, affected_rows=0)

        elif inputs.op == "insert":
            assert inputs.data is not None
            sql, params = builder.insert(inputs.data)
            return await backend.execute(sql, params)

        elif inputs.op == "select":
            # Resolve limit/offset if they're strings
            limit = int(inputs.limit) if inputs.limit is not None else None
            offset = int(inputs.offset) if inputs.offset is not None else None
            sql, params = builder.select(
                where=inputs.where,
                order=inputs.order,
                limit=limit,
                offset=offset,
            )
            return await backend.query(sql, params)

        elif inputs.op == "update":
            assert inputs.where is not None
            assert inputs.data is not None
            sql, params = builder.update(where=inputs.where, data=inputs.data)
            return await backend.execute(sql, params)

        elif inputs.op == "delete":
            assert inputs.where is not None
            sql, params = builder.delete(where=inputs.where)
            return await backend.execute(sql, params)

        elif inputs.op == "upsert":
            assert inputs.data is not None
            assert inputs.conflict is not None
            sql, params = builder.upsert(data=inputs.data, conflict=inputs.conflict)
            return await backend.execute(sql, params)

        else:
            raise ValueError(f"Unknown operation: {inputs.op}")

    def _create_backend(self, engine: str) -> DatabaseBackendBase:
        """Create appropriate backend for engine."""
        if engine == "sqlite":
            return SqliteBackend()
        elif engine == "postgresql":
            if PostgresBackend is None:
                raise ImportError(
                    "PostgreSQL backend requires 'asyncpg' package. "
                    "Install with: pip install workflows-mcp[postgresql]"
                )
            return PostgresBackend()
        elif engine == "mariadb":
            if MariaDBBackend is None:
                raise ImportError(
                    "MariaDB backend requires 'aiomysql' package. "
                    "Install with: pip install workflows-mcp[mariadb]"
                )
            return MariaDBBackend()
        else:
            raise ValueError(f"Unsupported engine: {engine}")

    def _create_config(
        self,
        engine: str,
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
        engine_enum = DatabaseEngine(engine)

        options: dict[str, Any] = {}
        if sqlite_pragmas:
            options["sqlite_pragmas"] = sqlite_pragmas

        return ConnectionConfig(
            dialect=engine_enum,
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
