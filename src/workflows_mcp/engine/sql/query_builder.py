"""Query builder for model-based CRUD operations.

This module generates parameterized SQL statements for insert, select, update,
delete, and upsert operations based on model schemas.

Example:
    schema = ModelSchema.from_dict(model_spec)
    builder = QueryBuilder(schema, DatabaseEngine.SQLITE)

    # Insert
    sql, params = builder.insert({"name": "Task 1", "status": "pending"})
    # -> INSERT INTO "tasks" ("name", "status") VALUES (?, ?)
    # -> ["Task 1", "pending"]

    # Select with filters
    sql, params = builder.select(where={"status": "running"}, limit=10)
    # -> SELECT * FROM "tasks" WHERE "status" = ? LIMIT 10
    # -> ["running"]
"""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime
from typing import Any

from .backend import DatabaseEngine
from .model import ModelSchema


class QueryBuilder:
    """Builds parameterized SQL queries from model schemas.

    Generates CRUD SQL statements with proper parameter placeholders
    for the target database engine.

    Attributes:
        schema: The model schema defining the table structure
        engine: Target database engine for SQL dialect differences
    """

    def __init__(self, schema: ModelSchema, engine: DatabaseEngine):
        """Initialize query builder.

        Args:
            schema: Model schema for the table
            engine: Target database engine
        """
        self.schema = schema
        self.engine = engine

    def _placeholder(self, index: int) -> str:
        """Get parameter placeholder for the engine.

        Args:
            index: 0-based parameter index

        Returns:
            Placeholder string (?, $1, %s depending on engine)
        """
        if self.engine == DatabaseEngine.SQLITE:
            return "?"
        elif self.engine == DatabaseEngine.POSTGRESQL:
            return f"${index + 1}"
        else:  # MARIADB
            return "%s"

    def _generate_auto_values(self, data: dict[str, Any], is_insert: bool) -> dict[str, Any]:
        """Generate auto values (uuid, created_at, updated_at).

        Args:
            data: User-provided data
            is_insert: Whether this is an insert (affects which auto values apply)

        Returns:
            Data with auto-generated values added
        """
        result = dict(data)

        for col in self.schema.columns:
            if col.auto == "uuid" and col.name not in result and is_insert:
                result[col.name] = secrets.token_hex(16)
            elif col.auto == "created" and col.name not in result and is_insert:
                result[col.name] = datetime.now(UTC).isoformat()
            elif col.auto == "updated":
                result[col.name] = datetime.now(UTC).isoformat()

        return result

    def _apply_defaults(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply column defaults for missing fields.

        Args:
            data: User-provided data

        Returns:
            Data with defaults applied
        """
        result = dict(data)

        for col in self.schema.columns:
            if col.name not in result and col.default is not None:
                result[col.name] = col.default

        return result

    def _serialize_json_columns(self, data: dict[str, Any]) -> dict[str, Any]:
        """Serialize dict/list values for JSON columns.

        SQLite and other databases can't bind Python dicts/lists directly.
        This method JSON-serializes values for columns with type='json'.

        Args:
            data: Row data with potentially non-serialized values

        Returns:
            Data with JSON columns serialized to strings
        """
        result = dict(data)

        # Build a lookup of column types
        col_types = {col.name: col.type for col in self.schema.columns}

        for key, value in result.items():
            # Serialize dicts and lists for JSON columns
            if col_types.get(key) == "json" and isinstance(value, (dict, list)):
                result[key] = json.dumps(value)

        return result

    def insert(self, data: dict[str, Any]) -> tuple[str, list[Any]]:
        """Generate INSERT statement.

        Args:
            data: Row data to insert

        Returns:
            Tuple of (SQL statement, parameter list)
        """
        # Apply defaults and auto-generated values
        row = self._apply_defaults(data)
        row = self._generate_auto_values(row, is_insert=True)
        row = self._serialize_json_columns(row)

        columns = list(row.keys())
        values = list(row.values())
        placeholders = [self._placeholder(i) for i in range(len(values))]

        col_list = ", ".join(f'"{c}"' for c in columns)
        val_list = ", ".join(placeholders)

        sql = f'INSERT INTO "{self.schema.table}" ({col_list}) VALUES ({val_list})'

        # Add RETURNING for the primary key if exists
        pk = self.schema.get_primary_key()
        if pk:
            sql += f' RETURNING "{pk.name}"'

        return sql, values

    def select(
        self,
        where: dict[str, Any] | None = None,
        order: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        columns: list[str] | None = None,
    ) -> tuple[str, list[Any]]:
        """Generate SELECT statement.

        Args:
            where: Filter conditions (optional)
            order: Sort order as list of "column:dir" strings (optional)
            limit: Maximum rows to return (optional)
            offset: Rows to skip (optional)
            columns: Specific columns to select (optional, default all)

        Returns:
            Tuple of (SQL statement, parameter list)
        """
        # Column list
        if columns:
            col_list = ", ".join(f'"{c}"' for c in columns)
        else:
            col_list = "*"

        sql = f'SELECT {col_list} FROM "{self.schema.table}"'
        params: list[Any] = []

        # WHERE clause
        if where:
            where_sql, where_params = self._build_where(where, len(params))
            sql += f" WHERE {where_sql}"
            params.extend(where_params)

        # ORDER BY clause
        if order:
            order_sql = self._build_order(order)
            sql += f" ORDER BY {order_sql}"

        # LIMIT clause
        if limit is not None:
            sql += f" LIMIT {self._placeholder(len(params))}"
            params.append(limit)

        # OFFSET clause
        if offset is not None:
            sql += f" OFFSET {self._placeholder(len(params))}"
            params.append(offset)

        return sql, params

    def update(self, where: dict[str, Any], data: dict[str, Any]) -> tuple[str, list[Any]]:
        """Generate UPDATE statement.

        Args:
            where: Filter conditions (required)
            data: Column values to update

        Returns:
            Tuple of (SQL statement, parameter list)
        """
        if not where:
            raise ValueError("UPDATE requires a WHERE clause for safety")
        if not data:
            raise ValueError("UPDATE requires data to update")

        # Apply auto-updated values and serialize JSON columns
        row = self._generate_auto_values(data, is_insert=False)
        row = self._serialize_json_columns(row)

        # Build SET clause
        set_parts = []
        params: list[Any] = []
        for col, value in row.items():
            set_parts.append(f'"{col}" = {self._placeholder(len(params))}')
            params.append(value)

        set_sql = ", ".join(set_parts)

        # Build WHERE clause
        where_sql, where_params = self._build_where(where, len(params))
        params.extend(where_params)

        sql = f'UPDATE "{self.schema.table}" SET {set_sql} WHERE {where_sql}'

        return sql, params

    def delete(self, where: dict[str, Any]) -> tuple[str, list[Any]]:
        """Generate DELETE statement.

        Args:
            where: Filter conditions (required)

        Returns:
            Tuple of (SQL statement, parameter list)
        """
        if not where:
            raise ValueError("DELETE requires a WHERE clause for safety")

        where_sql, params = self._build_where(where, 0)
        sql = f'DELETE FROM "{self.schema.table}" WHERE {where_sql}'

        return sql, params

    def upsert(self, data: dict[str, Any], conflict: list[str]) -> tuple[str, list[Any]]:
        """Generate UPSERT (INSERT ... ON CONFLICT DO UPDATE) statement.

        Args:
            data: Row data to insert/update
            conflict: Columns to check for conflict (usually primary key)

        Returns:
            Tuple of (SQL statement, parameter list)
        """
        if not conflict:
            raise ValueError("UPSERT requires conflict columns")

        # Apply defaults, auto-generated values, and serialize JSON columns
        row = self._apply_defaults(data)
        row = self._generate_auto_values(row, is_insert=True)
        row = self._serialize_json_columns(row)

        columns = list(row.keys())
        values = list(row.values())
        placeholders = [self._placeholder(i) for i in range(len(values))]

        col_list = ", ".join(f'"{c}"' for c in columns)
        val_list = ", ".join(placeholders)
        conflict_cols = ", ".join(f'"{c}"' for c in conflict)

        # Build update set for non-conflict columns
        update_cols = [c for c in columns if c not in conflict]
        if update_cols:
            if self.engine == DatabaseEngine.SQLITE:
                update_set = ", ".join(f'"{c}" = excluded."{c}"' for c in update_cols)
            elif self.engine == DatabaseEngine.POSTGRESQL:
                update_set = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)
            else:  # MARIADB uses different syntax
                update_set = ", ".join(f'"{c}" = VALUES("{c}")' for c in update_cols)
        else:
            # If all columns are conflict columns, do nothing on conflict
            update_set = None

        if self.engine == DatabaseEngine.MARIADB:
            # MariaDB uses ON DUPLICATE KEY UPDATE
            sql = f'INSERT INTO "{self.schema.table}" ({col_list}) VALUES ({val_list})'
            if update_set:
                sql += f" ON DUPLICATE KEY UPDATE {update_set}"
        else:
            # SQLite and PostgreSQL use ON CONFLICT
            sql = f'INSERT INTO "{self.schema.table}" ({col_list}) VALUES ({val_list})'
            sql += f" ON CONFLICT ({conflict_cols})"
            if update_set:
                sql += f" DO UPDATE SET {update_set}"
            else:
                sql += " DO NOTHING"

        # Add RETURNING for primary key
        pk = self.schema.get_primary_key()
        if pk and self.engine != DatabaseEngine.MARIADB:
            sql += f' RETURNING "{pk.name}"'

        return sql, values

    def _build_where(self, where: dict[str, Any], param_offset: int = 0) -> tuple[str, list[Any]]:
        """Build WHERE clause from conditions.

        Supports:
        - Simple equality: {"status": "running"}
        - Operators: {"priority": {">": 5}}
        - IN: {"type": {"in": ["a", "b"]}}
        - IS NULL: {"deleted": {"is": null}}
        - LIKE: {"name": {"like": "%test%"}}

        Args:
            where: Condition dictionary
            param_offset: Starting index for parameters

        Returns:
            Tuple of (WHERE clause SQL, parameter list)
        """
        conditions = []
        params: list[Any] = []

        for col, value in where.items():
            if isinstance(value, dict):
                # Operator syntax
                for op, operand in value.items():
                    sql_op = self._map_operator(op)
                    if op.lower() == "in":
                        # IN operator with list
                        if not isinstance(operand, list):
                            operand = [operand]
                        placeholders = [
                            self._placeholder(param_offset + len(params) + i)
                            for i in range(len(operand))
                        ]
                        conditions.append(f'"{col}" IN ({", ".join(placeholders)})')
                        params.extend(operand)
                    elif op.lower() == "is":
                        # IS NULL / IS NOT NULL
                        if operand is None or str(operand).lower() == "null":
                            conditions.append(f'"{col}" IS NULL')
                        else:
                            conditions.append(f'"{col}" IS NOT NULL')
                    elif op.lower() == "not":
                        # IS NOT NULL
                        if operand is None or str(operand).lower() == "null":
                            conditions.append(f'"{col}" IS NOT NULL')
                        else:
                            conditions.append(
                                f'"{col}" != {self._placeholder(param_offset + len(params))}'
                            )
                            params.append(operand)
                    else:
                        # Comparison operators
                        conditions.append(
                            f'"{col}" {sql_op} {self._placeholder(param_offset + len(params))}'
                        )
                        params.append(operand)
            elif value is None:
                # NULL check
                conditions.append(f'"{col}" IS NULL')
            else:
                # Simple equality
                conditions.append(f'"{col}" = {self._placeholder(param_offset + len(params))}')
                params.append(value)

        where_sql = " AND ".join(conditions)
        return where_sql, params

    def _map_operator(self, op: str) -> str:
        """Map operator string to SQL operator."""
        mapping = {
            "=": "=",
            "==": "=",
            "!=": "!=",
            "<>": "!=",
            "<": "<",
            ">": ">",
            "<=": "<=",
            ">=": ">=",
            "like": "LIKE",
            "ilike": "ILIKE",
        }
        return mapping.get(op.lower(), op)

    def _build_order(self, order: list[str]) -> str:
        """Build ORDER BY clause.

        Supports "column" or "column:desc" format.

        Args:
            order: List of order specifications

        Returns:
            ORDER BY clause (without the ORDER BY keywords)
        """
        parts = []
        for spec in order:
            if ":" in spec:
                col, direction = spec.split(":", 1)
                parts.append(f'"{col}" {direction.upper()}')
            else:
                parts.append(f'"{spec}"')
        return ", ".join(parts)
