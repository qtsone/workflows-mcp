"""Tests for the SQL executor and backends.

Tests focus on SQLite backend since it doesn't require external dependencies.
PostgreSQL and MariaDB backends are tested when those optional dependencies are installed.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from workflows_mcp.engine.execution import Execution
from workflows_mcp.engine.executors_sql import SqlExecutor, SqlInput
from workflows_mcp.engine.sql import (
    ConnectionConfig,
    DatabaseDialect,
    ParamConverter,
    QueryResult,
    SqliteBackend,
    convert_sql_for_dialect,
)

# ============================================================================
# ParamConverter Tests
# ============================================================================


class TestParamConverter:
    """Tests for SQL parameter placeholder conversion."""

    def test_qmark_to_numeric(self) -> None:
        """Convert ? placeholders to $1, $2 for PostgreSQL."""
        converter = ParamConverter(DatabaseDialect.POSTGRESQL)
        sql = "SELECT * FROM users WHERE id = ? AND status = ?"
        result = converter.convert(sql)
        assert result == "SELECT * FROM users WHERE id = $1 AND status = $2"

    def test_qmark_to_format(self) -> None:
        """Convert ? placeholders to %s for MariaDB."""
        converter = ParamConverter(DatabaseDialect.MARIADB)
        sql = "SELECT * FROM users WHERE id = ? AND status = ?"
        result = converter.convert(sql)
        assert result == "SELECT * FROM users WHERE id = %s AND status = %s"

    def test_numeric_to_qmark(self) -> None:
        """Convert $1, $2 placeholders to ? for SQLite."""
        converter = ParamConverter(DatabaseDialect.SQLITE)
        sql = "SELECT * FROM users WHERE id = $1 AND status = $2"
        result = converter.convert(sql)
        assert result == "SELECT * FROM users WHERE id = ? AND status = ?"

    def test_format_to_numeric(self) -> None:
        """Convert %s placeholders to $1, $2 for PostgreSQL."""
        converter = ParamConverter(DatabaseDialect.POSTGRESQL)
        sql = "SELECT * FROM users WHERE id = %s AND status = %s"
        result = converter.convert(sql)
        assert result == "SELECT * FROM users WHERE id = $1 AND status = $2"

    def test_no_placeholders(self) -> None:
        """SQL without placeholders should pass through unchanged."""
        converter = ParamConverter(DatabaseDialect.POSTGRESQL)
        sql = "SELECT * FROM users"
        result = converter.convert(sql)
        assert result == sql

    def test_same_dialect_passthrough(self) -> None:
        """SQL already in target dialect should pass through unchanged."""
        converter = ParamConverter(DatabaseDialect.POSTGRESQL)
        sql = "SELECT * FROM users WHERE id = $1"
        result = converter.convert(sql)
        assert result == sql

    def test_convert_params_dict_to_tuple(self) -> None:
        """Dict params should be converted to tuple for PostgreSQL."""
        converter = ParamConverter(DatabaseDialect.POSTGRESQL)
        params = {"name": "Alice", "status": "active"}
        result = converter.convert_params(params)
        assert result == ("Alice", "active")

    def test_convert_params_list_to_tuple(self) -> None:
        """List params should be converted to tuple."""
        converter = ParamConverter(DatabaseDialect.POSTGRESQL)
        params = ["Alice", "active"]
        result = converter.convert_params(params)
        assert result == ("Alice", "active")

    def test_convert_params_none(self) -> None:
        """None params should return None."""
        converter = ParamConverter(DatabaseDialect.POSTGRESQL)
        result = converter.convert_params(None)
        assert result is None

    def test_convenience_function(self) -> None:
        """Test the convenience function for SQL conversion."""
        sql = "SELECT * FROM users WHERE id = ?"
        result = convert_sql_for_dialect(sql, DatabaseDialect.POSTGRESQL)
        assert result == "SELECT * FROM users WHERE id = $1"


# ============================================================================
# SQLite Backend Tests
# ============================================================================


class TestSqliteBackend:
    """Tests for SQLite backend."""

    @pytest.fixture
    def db_path(self) -> str:
        """Create a temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            return f.name

    @pytest.fixture
    async def backend(self, db_path: str) -> AsyncGenerator[SqliteBackend, None]:
        """Create a connected SQLite backend."""
        backend = SqliteBackend()
        config = ConnectionConfig(dialect=DatabaseDialect.SQLITE, path=db_path)
        await backend.connect(config)
        yield backend
        await backend.disconnect()

    async def test_connect_disconnect(self, db_path: str) -> None:
        """Test basic connection and disconnection."""
        backend = SqliteBackend()
        config = ConnectionConfig(dialect=DatabaseDialect.SQLITE, path=db_path)

        await backend.connect(config)
        assert backend._conn is not None

        await backend.disconnect()
        assert backend._conn is None

    async def test_connect_creates_parent_dirs(self) -> None:
        """Test that connect creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "subdir" / "nested" / "test.db"

            backend = SqliteBackend()
            config = ConnectionConfig(dialect=DatabaseDialect.SQLITE, path=str(nested_path))
            await backend.connect(config)

            assert nested_path.parent.exists()
            await backend.disconnect()

    async def test_memory_database(self) -> None:
        """Test in-memory database connection."""
        backend = SqliteBackend()
        config = ConnectionConfig(dialect=DatabaseDialect.SQLITE, path=":memory:")
        await backend.connect(config)

        # Create table and insert data
        await backend.execute_script("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        result = await backend.execute("INSERT INTO test (name) VALUES (?)", ("Alice",))
        assert result.last_insert_id == 1

        # Query data
        result = await backend.query("SELECT * FROM test")
        assert len(result.rows) == 1
        assert result.rows[0]["name"] == "Alice"

        await backend.disconnect()

    async def test_execute_script(self, backend: SqliteBackend) -> None:
        """Test executing multi-statement SQL script."""
        script = """
        CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE posts (id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT);
        CREATE INDEX idx_posts_user ON posts(user_id);
        """
        await backend.execute_script(script)

        # Verify tables exist
        result = await backend.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = [row["name"] for row in result.rows]
        assert "users" in table_names
        assert "posts" in table_names

    async def test_query(self, backend: SqliteBackend) -> None:
        """Test SELECT query."""
        await backend.execute_script(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)"
        )
        await backend.execute("INSERT INTO users (name, age) VALUES (?, ?)", ("Alice", 30))
        await backend.execute("INSERT INTO users (name, age) VALUES (?, ?)", ("Bob", 25))

        result = await backend.query("SELECT * FROM users ORDER BY name")

        assert result.row_count == 2
        assert result.columns == ["id", "name", "age"]
        assert result.rows[0]["name"] == "Alice"
        assert result.rows[1]["name"] == "Bob"

    async def test_query_with_params(self, backend: SqliteBackend) -> None:
        """Test SELECT query with parameters."""
        await backend.execute_script(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, status TEXT)"
        )
        await backend.execute("INSERT INTO users (name, status) VALUES (?, ?)", ("Alice", "active"))
        await backend.execute("INSERT INTO users (name, status) VALUES (?, ?)", ("Bob", "inactive"))

        result = await backend.query("SELECT * FROM users WHERE status = ?", ("active",))

        assert result.row_count == 1
        assert result.rows[0]["name"] == "Alice"

    async def test_execute_insert(self, backend: SqliteBackend) -> None:
        """Test INSERT statement."""
        await backend.execute_script("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")

        result = await backend.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))

        assert result.last_insert_id == 1
        assert result.affected_rows == 1

    async def test_execute_update(self, backend: SqliteBackend) -> None:
        """Test UPDATE statement."""
        await backend.execute_script("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        await backend.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))
        await backend.execute("INSERT INTO users (name) VALUES (?)", ("Bob",))

        result = await backend.execute(
            "UPDATE users SET name = ? WHERE name = ?", ("Alicia", "Alice")
        )

        assert result.affected_rows == 1

    async def test_execute_delete(self, backend: SqliteBackend) -> None:
        """Test DELETE statement."""
        await backend.execute_script("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        await backend.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))
        await backend.execute("INSERT INTO users (name) VALUES (?)", ("Bob",))

        result = await backend.execute("DELETE FROM users WHERE name = ?", ("Alice",))

        assert result.affected_rows == 1

        # Verify deletion
        query_result = await backend.query("SELECT * FROM users")
        assert query_result.row_count == 1

    async def test_execute_many(self, backend: SqliteBackend) -> None:
        """Test batch insert with execute_many."""
        await backend.execute_script("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")

        params_list: list[tuple[Any, ...] | dict[str, Any]] = [
            ("Alice",),
            ("Bob",),
            ("Charlie",),
        ]
        result = await backend.execute_many("INSERT INTO users (name) VALUES (?)", params_list)

        assert result.affected_rows == 3

        # Verify all inserted
        query_result = await backend.query("SELECT * FROM users ORDER BY name")
        assert query_result.row_count == 3

    async def test_transaction_commit(self, backend: SqliteBackend) -> None:
        """Test transaction commit."""
        await backend.execute_script("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")

        await backend.begin_transaction()
        assert backend.in_transaction

        await backend.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))
        await backend.execute("INSERT INTO users (name) VALUES (?)", ("Bob",))

        await backend.commit()
        assert not backend.in_transaction

        # Verify both inserted
        result = await backend.query("SELECT * FROM users")
        assert result.row_count == 2

    async def test_transaction_rollback(self, backend: SqliteBackend) -> None:
        """Test transaction rollback."""
        await backend.execute_script("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        await backend.execute("INSERT INTO users (name) VALUES (?)", ("Existing",))

        await backend.begin_transaction()
        await backend.execute("INSERT INTO users (name) VALUES (?)", ("New",))
        await backend.rollback()

        # Verify rollback (only original row should exist)
        result = await backend.query("SELECT * FROM users")
        assert result.row_count == 1
        assert result.rows[0]["name"] == "Existing"

    async def test_transaction_immediate(self, backend: SqliteBackend) -> None:
        """Test IMMEDIATE isolation level."""
        await backend.execute_script("CREATE TABLE counter (id INTEGER PRIMARY KEY, value INTEGER)")
        await backend.execute("INSERT INTO counter (id, value) VALUES (1, 0)")

        await backend.begin_transaction("immediate")
        assert backend.in_transaction

        # Read and update atomically
        result = await backend.query("SELECT value FROM counter WHERE id = 1")
        current = result.rows[0]["value"]

        await backend.execute("UPDATE counter SET value = ? WHERE id = 1", (current + 1,))
        await backend.commit()

        # Verify update
        result = await backend.query("SELECT value FROM counter WHERE id = 1")
        assert result.rows[0]["value"] == 1

    async def test_query_result_dataclass(self) -> None:
        """Test QueryResult dataclass defaults."""
        result = QueryResult()
        assert result.rows == []
        assert result.row_count == 0
        assert result.columns == []
        assert result.last_insert_id is None
        assert result.affected_rows == 0


# ============================================================================
# ConnectionConfig Tests
# ============================================================================


class TestConnectionConfig:
    """Tests for ConnectionConfig validation."""

    def test_sqlite_requires_path(self) -> None:
        """SQLite config requires path."""
        with pytest.raises(ValueError, match="SQLite requires 'path'"):
            ConnectionConfig(dialect=DatabaseDialect.SQLITE)

    def test_postgresql_requires_host(self) -> None:
        """PostgreSQL config requires host."""
        with pytest.raises(ValueError, match="postgresql requires 'host'"):
            ConnectionConfig(
                dialect=DatabaseDialect.POSTGRESQL,
                database="mydb",
            )

    def test_postgresql_requires_database(self) -> None:
        """PostgreSQL config requires database."""
        with pytest.raises(ValueError, match="postgresql requires 'database'"):
            ConnectionConfig(
                dialect=DatabaseDialect.POSTGRESQL,
                host="localhost",
            )

    def test_postgresql_default_port(self) -> None:
        """PostgreSQL gets default port 5432."""
        config = ConnectionConfig(
            dialect=DatabaseDialect.POSTGRESQL,
            host="localhost",
            database="mydb",
        )
        assert config.port == 5432

    def test_mariadb_default_port(self) -> None:
        """MariaDB gets default port 3306."""
        config = ConnectionConfig(
            dialect=DatabaseDialect.MARIADB,
            host="localhost",
            database="mydb",
        )
        assert config.port == 3306

    def test_sqlite_valid_config(self) -> None:
        """Valid SQLite config should work."""
        config = ConnectionConfig(
            dialect=DatabaseDialect.SQLITE,
            path="/data/test.db",
        )
        assert config.path == "/data/test.db"

    def test_custom_options(self) -> None:
        """Custom options should be preserved."""
        config = ConnectionConfig(
            dialect=DatabaseDialect.SQLITE,
            path=":memory:",
            options={"sqlite_pragmas": {"cache_size": -64000}},
        )
        assert config.options["sqlite_pragmas"]["cache_size"] == -64000


# ============================================================================
# SqlExecutor Integration Tests
# ============================================================================


class TestSqlExecutorIntegration:
    """Integration tests for SqlExecutor through workflow blocks."""

    async def test_sqlite_query_operation(self) -> None:
        """Test SQLite query through executor."""
        executor = SqlExecutor()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        # First, create schema and insert data
        inputs = SqlInput(
            dialect="sqlite",
            path=db_path,
            operation="script",
            sql="""
                CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);
                INSERT INTO users (name) VALUES ('Alice');
                INSERT INTO users (name) VALUES ('Bob');
            """,
        )
        context = Execution()
        await executor.execute(inputs, context)

        # Then query
        inputs = SqlInput(
            dialect="sqlite",
            path=db_path,
            operation="query",
            sql="SELECT * FROM users ORDER BY name",
        )
        result = await executor.execute(inputs, context)

        assert result.success
        assert result.row_count == 2
        assert result.rows[0]["name"] == "Alice"
        assert result.dialect == "sqlite"

    async def test_sqlite_execute_operation(self) -> None:
        """Test SQLite execute through executor."""
        executor = SqlExecutor()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        # Create schema
        inputs = SqlInput(
            dialect="sqlite",
            path=db_path,
            operation="script",
            sql="CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)",
        )
        context = Execution()
        await executor.execute(inputs, context)

        # Insert data
        inputs = SqlInput(
            dialect="sqlite",
            path=db_path,
            operation="execute",
            sql="INSERT INTO users (name) VALUES (?)",
            params=["Alice"],
        )
        result = await executor.execute(inputs, context)

        assert result.success
        assert result.affected_rows == 1
        assert result.last_insert_id == 1

    async def test_sqlite_with_init_sql(self) -> None:
        """Test SQLite with init_sql for schema setup."""
        executor = SqlExecutor()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        # Query with init_sql that creates the table
        inputs = SqlInput(
            dialect="sqlite",
            path=db_path,
            operation="query",
            init_sql="""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                );
                INSERT OR IGNORE INTO users (id, name) VALUES (1, 'Alice');
            """,
            sql="SELECT * FROM users",
        )
        context = Execution()
        result = await executor.execute(inputs, context)

        assert result.success
        assert result.row_count >= 1

    async def test_sqlite_transaction_operation(self) -> None:
        """Test SQLite transaction operation."""
        executor = SqlExecutor()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        # Create schema first
        setup_inputs = SqlInput(
            dialect="sqlite",
            path=db_path,
            operation="script",
            sql="CREATE TABLE counter (id INTEGER PRIMARY KEY, value INTEGER)",
        )
        context = Execution()
        await executor.execute(setup_inputs, context)

        # Transaction with multiple statements
        inputs = SqlInput(
            dialect="sqlite",
            path=db_path,
            operation="transaction",
            isolation_level="immediate",
            sql="""
                INSERT INTO counter (id, value) VALUES (1, 100);
                UPDATE counter SET value = value + 50 WHERE id = 1;
                SELECT value FROM counter WHERE id = 1
            """,
        )
        result = await executor.execute(inputs, context)

        assert result.success
        # Last statement is SELECT, so we should have rows
        assert len(result.rows) == 1
        assert result.rows[0]["value"] == 150

    async def test_invalid_dialect_error(self) -> None:
        """Test that invalid dialect raises error."""
        with pytest.raises(ValidationError):
            SqlInput(
                dialect="invalid",  # type: ignore[arg-type]
                path="/data/test.db",
                operation="query",
                sql="SELECT 1",
            )

    async def test_sqlite_missing_path_error(self) -> None:
        """Test that SQLite without path raises error."""
        with pytest.raises(ValidationError, match="SQLite requires 'path'"):
            SqlInput(
                dialect="sqlite",
                operation="query",
                sql="SELECT 1",
            )

    async def test_postgresql_missing_host_error(self) -> None:
        """Test that PostgreSQL without host raises error."""
        with pytest.raises(ValidationError, match="postgresql requires 'host'"):
            SqlInput(
                dialect="postgresql",
                database="mydb",
                operation="query",
                sql="SELECT 1",
            )
