"""Unit tests for SQL model schema and query builder."""

from __future__ import annotations

import pytest

from workflows_mcp.engine.sql.backend import DatabaseEngine
from workflows_mcp.engine.sql.model import (
    ColumnDef,
    IndexDef,
    ModelSchema,
)
from workflows_mcp.engine.sql.query_builder import QueryBuilder


class TestColumnDef:
    """Tests for ColumnDef class."""

    def test_basic_column(self) -> None:
        """Test basic column definition."""
        col = ColumnDef.from_dict("name", {"type": "text"})
        assert col.name == "name"
        assert col.type == "text"
        assert col.primary is False
        assert col.required is False

    def test_primary_key_column(self) -> None:
        """Test primary key column."""
        col = ColumnDef.from_dict("id", {"type": "text", "primary": True})
        sql = col.to_sql(DatabaseEngine.SQLITE)
        assert '"id" TEXT PRIMARY KEY' == sql

    def test_required_column(self) -> None:
        """Test NOT NULL column."""
        col = ColumnDef.from_dict("name", {"type": "text", "required": True})
        sql = col.to_sql(DatabaseEngine.SQLITE)
        assert '"name" TEXT NOT NULL' == sql

    def test_default_value(self) -> None:
        """Test column with default value."""
        col = ColumnDef.from_dict("status", {"type": "text", "default": "pending"})
        sql = col.to_sql(DatabaseEngine.SQLITE)
        assert "\"status\" TEXT DEFAULT 'pending'" == sql

    def test_check_constraint(self) -> None:
        """Test column with CHECK constraint."""
        col = ColumnDef.from_dict("kind", {"type": "text", "check": "kind IN ('a', 'b')"})
        sql = col.to_sql(DatabaseEngine.SQLITE)
        assert "CHECK(kind IN ('a', 'b'))" in sql

    def test_foreign_key(self) -> None:
        """Test foreign key generation."""
        col = ColumnDef.from_dict(
            "parent_id",
            {"type": "text", "references": "tasks.task_id", "on_delete": "cascade"},
        )
        fk_sql = col.foreign_key_sql(DatabaseEngine.SQLITE)
        assert fk_sql is not None
        assert 'FOREIGN KEY ("parent_id") REFERENCES "tasks"("task_id")' in fk_sql
        assert "ON DELETE CASCADE" in fk_sql

    def test_json_column_sqlite(self) -> None:
        """Test JSON column uses 'JSON TEXT' type for SQLite (official pattern)."""
        col = ColumnDef.from_dict("metadata", {"type": "json", "default": "{}"})
        sql = col.to_sql(DatabaseEngine.SQLITE)
        # SQLite pattern: JSON prefix documents intent, TEXT suffix gives TEXT affinity
        assert '"metadata" JSON TEXT' in sql
        assert "DEFAULT '{}'" in sql

    def test_json_column_postgresql(self) -> None:
        """Test JSON column uses JSONB for PostgreSQL."""
        col = ColumnDef.from_dict("metadata", {"type": "json"})
        sql = col.to_sql(DatabaseEngine.POSTGRESQL)
        assert '"metadata" JSONB' in sql

    def test_json_column_mariadb(self) -> None:
        """Test JSON column uses JSON for MariaDB."""
        col = ColumnDef.from_dict("metadata", {"type": "json"})
        sql = col.to_sql(DatabaseEngine.MARIADB)
        assert '"metadata" JSON' in sql


class TestIndexDef:
    """Tests for IndexDef class."""

    def test_simple_index(self) -> None:
        """Test simple single-column index."""
        idx = IndexDef.from_dict({"columns": ["status"]})
        sql = idx.to_sql("tasks", DatabaseEngine.SQLITE)
        assert 'CREATE INDEX IF NOT EXISTS "idx_tasks_status"' in sql
        assert 'ON "tasks" ("status")' in sql

    def test_composite_index(self) -> None:
        """Test multi-column index."""
        idx = IndexDef.from_dict({"columns": ["kind", "status"]})
        sql = idx.to_sql("tasks", DatabaseEngine.SQLITE)
        assert '"kind", "status"' in sql

    def test_named_index(self) -> None:
        """Test index with explicit name."""
        idx = IndexDef.from_dict({"name": "my_index", "columns": ["created_at"]})
        sql = idx.to_sql("tasks", DatabaseEngine.SQLITE)
        assert '"my_index"' in sql

    def test_unique_index(self) -> None:
        """Test unique index."""
        idx = IndexDef.from_dict({"columns": ["email"], "unique": True})
        sql = idx.to_sql("users", DatabaseEngine.SQLITE)
        assert "CREATE UNIQUE INDEX" in sql

    def test_ordered_index(self) -> None:
        """Test index with sort order."""
        idx = IndexDef.from_dict({"columns": ["created_at"], "order": "desc"})
        sql = idx.to_sql("tasks", DatabaseEngine.SQLITE)
        assert '"created_at" DESC' in sql


class TestModelSchema:
    """Tests for ModelSchema class."""

    @pytest.fixture
    def task_model(self) -> dict[str, object]:
        """Sample task model for testing."""
        return {
            "table": "tasks",
            "columns": {
                "task_id": {"type": "text", "primary": True},
                "parent_id": {
                    "type": "text",
                    "references": "tasks.task_id",
                    "on_delete": "cascade",
                },
                "name": {"type": "text", "required": True},
                "status": {"type": "text", "default": "pending"},
                "created_at": {"type": "timestamp", "auto": "created"},
            },
            "indexes": [
                {"columns": ["parent_id"]},
                {"columns": ["status"]},
            ],
        }

    def test_from_dict(self, task_model: dict[str, object]) -> None:
        """Test creating ModelSchema from dict."""
        schema = ModelSchema.from_dict(task_model)
        assert schema.table == "tasks"
        assert len(schema.columns) == 5
        assert len(schema.indexes) == 2

    def test_missing_table_raises(self) -> None:
        """Test that missing table name raises error."""
        with pytest.raises(ValueError, match="table"):
            ModelSchema.from_dict({"columns": {"id": {"type": "text"}}})

    def test_missing_columns_raises(self) -> None:
        """Test that missing columns raises error."""
        with pytest.raises(ValueError, match="column"):
            ModelSchema.from_dict({"table": "test"})

    def test_get_primary_key(self, task_model: dict[str, object]) -> None:
        """Test getting primary key column."""
        schema = ModelSchema.from_dict(task_model)
        pk = schema.get_primary_key()
        assert pk is not None
        assert pk.name == "task_id"

    def test_to_create_sql_sqlite(self, task_model: dict[str, object]) -> None:
        """Test CREATE TABLE generation for SQLite."""
        schema = ModelSchema.from_dict(task_model)
        sql = schema.to_create_sql(DatabaseEngine.SQLITE)

        assert 'CREATE TABLE IF NOT EXISTS "tasks"' in sql
        assert '"task_id" TEXT PRIMARY KEY' in sql
        assert '"name" TEXT NOT NULL' in sql
        assert "DEFAULT 'pending'" in sql
        assert 'FOREIGN KEY ("parent_id") REFERENCES "tasks"("task_id")' in sql

    def test_to_create_sql_postgresql(self, task_model: dict[str, object]) -> None:
        """Test CREATE TABLE generation for PostgreSQL."""
        schema = ModelSchema.from_dict(task_model)
        sql = schema.to_create_sql(DatabaseEngine.POSTGRESQL)

        assert 'CREATE TABLE IF NOT EXISTS "tasks"' in sql
        assert "TIMESTAMPTZ" in sql  # PostgreSQL timestamp type

    def test_to_index_sql(self, task_model: dict[str, object]) -> None:
        """Test CREATE INDEX generation."""
        schema = ModelSchema.from_dict(task_model)
        indexes = schema.to_index_sql(DatabaseEngine.SQLITE)

        assert len(indexes) == 2
        assert any("parent_id" in idx for idx in indexes)
        assert any("status" in idx for idx in indexes)

    def test_column_names(self, task_model: dict[str, object]) -> None:
        """Test getting column names."""
        schema = ModelSchema.from_dict(task_model)
        names = schema.column_names()

        assert "task_id" in names
        assert "name" in names
        assert len(names) == 5


class TestQueryBuilder:
    """Tests for QueryBuilder class."""

    @pytest.fixture
    def schema(self) -> ModelSchema:
        """Sample schema for testing."""
        return ModelSchema.from_dict(
            {
                "table": "tasks",
                "columns": {
                    "task_id": {"type": "text", "primary": True, "auto": "uuid"},
                    "name": {"type": "text", "required": True},
                    "status": {"type": "text", "default": "pending"},
                    "created_at": {"type": "timestamp", "auto": "created"},
                    "updated_at": {"type": "timestamp", "auto": "updated"},
                },
            }
        )

    @pytest.fixture
    def builder(self, schema: ModelSchema) -> QueryBuilder:
        """Query builder for SQLite."""
        return QueryBuilder(schema, DatabaseEngine.SQLITE)

    def test_insert_basic(self, builder: QueryBuilder) -> None:
        """Test basic INSERT generation."""
        sql, params = builder.insert({"name": "Test Task"})

        assert 'INSERT INTO "tasks"' in sql
        assert "VALUES" in sql
        assert "?" in sql  # SQLite placeholder
        assert "Test Task" in params
        assert "RETURNING" in sql

    def test_insert_auto_values(self, builder: QueryBuilder) -> None:
        """Test INSERT with auto-generated values."""
        sql, params = builder.insert({"name": "Test Task"})

        # Should have auto-generated task_id, created_at, updated_at
        assert '"task_id"' in sql
        assert '"created_at"' in sql
        assert '"updated_at"' in sql
        # Default status should be applied
        assert "pending" in params

    def test_select_all(self, builder: QueryBuilder) -> None:
        """Test SELECT * generation."""
        sql, params = builder.select()

        assert 'SELECT * FROM "tasks"' in sql
        assert len(params) == 0

    def test_select_with_where(self, builder: QueryBuilder) -> None:
        """Test SELECT with WHERE clause."""
        sql, params = builder.select(where={"status": "running"})

        assert "WHERE" in sql
        assert '"status" = ?' in sql
        assert params == ["running"]

    def test_select_with_operators(self, builder: QueryBuilder) -> None:
        """Test SELECT with operator conditions."""
        sql, params = builder.select(where={"priority": {">": 5}})

        assert '"priority" > ?' in sql
        assert params == [5]

    def test_select_with_in(self, builder: QueryBuilder) -> None:
        """Test SELECT with IN operator."""
        sql, params = builder.select(where={"status": {"in": ["a", "b", "c"]}})

        assert '"status" IN' in sql
        assert params == ["a", "b", "c"]

    def test_select_with_order(self, builder: QueryBuilder) -> None:
        """Test SELECT with ORDER BY."""
        sql, params = builder.select(order=["created_at:desc"])

        assert 'ORDER BY "created_at" DESC' in sql

    def test_select_with_limit_offset(self, builder: QueryBuilder) -> None:
        """Test SELECT with LIMIT and OFFSET."""
        sql, params = builder.select(limit=10, offset=20)

        assert "LIMIT ?" in sql
        assert "OFFSET ?" in sql
        assert params == [10, 20]

    def test_update_basic(self, builder: QueryBuilder) -> None:
        """Test basic UPDATE generation."""
        sql, params = builder.update(where={"task_id": "123"}, data={"status": "done"})

        assert 'UPDATE "tasks" SET' in sql
        assert '"status" = ?' in sql
        assert "WHERE" in sql
        assert '"task_id" = ?' in sql

    def test_update_requires_where(self, builder: QueryBuilder) -> None:
        """Test UPDATE requires WHERE clause."""
        with pytest.raises(ValueError, match="WHERE"):
            builder.update(where={}, data={"status": "done"})

    def test_update_auto_updated_at(self, builder: QueryBuilder) -> None:
        """Test UPDATE auto-sets updated_at."""
        sql, params = builder.update(where={"task_id": "123"}, data={"status": "done"})

        assert '"updated_at"' in sql

    def test_delete_basic(self, builder: QueryBuilder) -> None:
        """Test basic DELETE generation."""
        sql, params = builder.delete(where={"task_id": "123"})

        assert 'DELETE FROM "tasks"' in sql
        assert "WHERE" in sql
        assert params == ["123"]

    def test_delete_requires_where(self, builder: QueryBuilder) -> None:
        """Test DELETE requires WHERE clause."""
        with pytest.raises(ValueError, match="WHERE"):
            builder.delete(where={})

    def test_upsert_sqlite(self, builder: QueryBuilder) -> None:
        """Test UPSERT generation for SQLite."""
        sql, params = builder.upsert(data={"task_id": "123", "name": "Task"}, conflict=["task_id"])

        assert 'INSERT INTO "tasks"' in sql
        assert "ON CONFLICT" in sql
        assert '"task_id"' in sql
        assert "DO UPDATE SET" in sql

    def test_upsert_postgresql(self, schema: ModelSchema) -> None:
        """Test UPSERT generation for PostgreSQL."""
        builder = QueryBuilder(schema, DatabaseEngine.POSTGRESQL)
        sql, params = builder.upsert(data={"task_id": "123", "name": "Task"}, conflict=["task_id"])

        assert "ON CONFLICT" in sql
        assert "EXCLUDED" in sql  # PostgreSQL uses EXCLUDED

    def test_upsert_mariadb(self, schema: ModelSchema) -> None:
        """Test UPSERT generation for MariaDB."""
        builder = QueryBuilder(schema, DatabaseEngine.MARIADB)
        sql, params = builder.upsert(data={"task_id": "123", "name": "Task"}, conflict=["task_id"])

        assert "ON DUPLICATE KEY UPDATE" in sql

    def test_placeholder_sqlite(self, builder: QueryBuilder) -> None:
        """Test SQLite placeholder format."""
        assert builder._placeholder(0) == "?"
        assert builder._placeholder(5) == "?"

    def test_placeholder_postgresql(self, schema: ModelSchema) -> None:
        """Test PostgreSQL placeholder format."""
        builder = QueryBuilder(schema, DatabaseEngine.POSTGRESQL)
        assert builder._placeholder(0) == "$1"
        assert builder._placeholder(5) == "$6"

    def test_placeholder_mariadb(self, schema: ModelSchema) -> None:
        """Test MariaDB placeholder format."""
        builder = QueryBuilder(schema, DatabaseEngine.MARIADB)
        assert builder._placeholder(0) == "%s"
        assert builder._placeholder(5) == "%s"
