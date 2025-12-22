"""Model schema and DDL generation for Active Record-style database operations.

This module provides declarative model definitions that generate database DDL
(CREATE TABLE, CREATE INDEX) for multiple database engines.

Example model definition:
    model = {
        "table": "tasks",
        "columns": {
            "task_id": {"type": "text", "primary": True, "auto": "uuid"},
            "name": {"type": "text", "required": True},
            "status": {"type": "text", "default": "pending"},
            "created_at": {"type": "timestamp", "auto": "created"},
        },
        "indexes": [
            {"columns": ["status"]},
            {"name": "idx_tasks_created", "columns": ["created_at"], "order": "desc"},
        ],
    }

Usage:
    schema = ModelSchema.from_dict(model)
    ddl = schema.to_create_sql(DatabaseEngine.SQLITE)
    indexes = schema.to_index_sql(DatabaseEngine.SQLITE)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .backend import DatabaseEngine

# Column type mapping per engine
TYPE_MAPPING: dict[str, dict[DatabaseEngine, str]] = {
    "text": {
        DatabaseEngine.SQLITE: "TEXT",
        DatabaseEngine.POSTGRESQL: "TEXT",
        DatabaseEngine.MARIADB: "TEXT",
    },
    "integer": {
        DatabaseEngine.SQLITE: "INTEGER",
        DatabaseEngine.POSTGRESQL: "INTEGER",
        DatabaseEngine.MARIADB: "INT",
    },
    "real": {
        DatabaseEngine.SQLITE: "REAL",
        DatabaseEngine.POSTGRESQL: "DOUBLE PRECISION",
        DatabaseEngine.MARIADB: "DOUBLE",
    },
    "boolean": {
        DatabaseEngine.SQLITE: "INTEGER",
        DatabaseEngine.POSTGRESQL: "BOOLEAN",
        DatabaseEngine.MARIADB: "TINYINT(1)",
    },
    "json": {
        DatabaseEngine.SQLITE: "JSON TEXT",
        DatabaseEngine.POSTGRESQL: "JSONB",
        DatabaseEngine.MARIADB: "JSON",
    },
    "timestamp": {
        DatabaseEngine.SQLITE: "TEXT",
        DatabaseEngine.POSTGRESQL: "TIMESTAMPTZ",
        DatabaseEngine.MARIADB: "DATETIME",
    },
    "blob": {
        DatabaseEngine.SQLITE: "BLOB",
        DatabaseEngine.POSTGRESQL: "BYTEA",
        DatabaseEngine.MARIADB: "BLOB",
    },
}


@dataclass
class ColumnDef:
    """Column definition for a database table.

    Attributes:
        name: Column name
        type: Column type (text, integer, real, boolean, json, timestamp, blob)
        primary: Whether this is the primary key
        required: Whether the column is NOT NULL
        default: Default value (as string)
        auto: Auto-generation type (uuid, created, updated)
        references: Foreign key reference (table.column)
        on_delete: Foreign key on delete action (cascade, set_null, restrict)
        check: CHECK constraint expression
    """

    name: str
    type: str
    primary: bool = False
    required: bool = False
    default: str | None = None
    auto: Literal["uuid", "created", "updated"] | None = None
    references: str | None = None
    on_delete: Literal["cascade", "set_null", "restrict"] | None = None
    check: str | None = None

    @classmethod
    def from_dict(cls, name: str, spec: dict[str, Any]) -> ColumnDef:
        """Create ColumnDef from dictionary specification."""
        return cls(
            name=name,
            type=spec.get("type", "text"),
            primary=spec.get("primary", False),
            required=spec.get("required", False),
            default=spec.get("default"),
            auto=spec.get("auto"),
            references=spec.get("references"),
            on_delete=spec.get("on_delete"),
            check=spec.get("check"),
        )

    def to_sql(self, engine: DatabaseEngine) -> str:
        """Generate column definition SQL for the specified engine."""
        type_map = TYPE_MAPPING.get(self.type)
        if not type_map:
            raise ValueError(f"Unknown column type: {self.type}")

        sql_type = type_map[engine]
        parts = [f'"{self.name}"', sql_type]

        # PRIMARY KEY
        if self.primary:
            parts.append("PRIMARY KEY")

        # NOT NULL (implied by primary key, explicit for required)
        if self.required and not self.primary:
            parts.append("NOT NULL")

        # DEFAULT value
        if self.default is not None:
            default_value = self._format_default(self.default, engine)
            parts.append(f"DEFAULT {default_value}")
        elif self.auto in ("created", "updated"):
            # Auto-timestamped columns get DEFAULT CURRENT_TIMESTAMP
            # This ensures triggers and raw SQL inserts also get timestamps
            parts.append("DEFAULT CURRENT_TIMESTAMP")

        # CHECK constraint
        if self.check:
            parts.append(f"CHECK({self.check})")

        return " ".join(parts)

    def _format_default(self, value: Any, engine: DatabaseEngine) -> str:
        """Format default value for SQL."""
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            if engine == DatabaseEngine.POSTGRESQL:
                return str(value).upper()
            return "1" if value else "0"
        if isinstance(value, (int, float)):
            return str(value)
        # String value - quote it
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"

    def foreign_key_sql(self, engine: DatabaseEngine) -> str | None:
        """Generate FOREIGN KEY constraint SQL if this column has a reference."""
        if not self.references:
            return None

        # Parse table.column format
        if "." not in self.references:
            raise ValueError(
                f"Invalid reference format '{self.references}'. Expected 'table.column'"
            )

        ref_table, ref_column = self.references.split(".", 1)
        fk_sql = f'FOREIGN KEY ("{self.name}") REFERENCES "{ref_table}"("{ref_column}")'

        if self.on_delete:
            action = self.on_delete.upper().replace("_", " ")
            fk_sql += f" ON DELETE {action}"

        return fk_sql


@dataclass
class IndexDef:
    """Index definition for a database table.

    Attributes:
        columns: List of column names in the index
        name: Optional explicit index name (auto-generated if not provided)
        unique: Whether this is a unique index
        order: Sort order for the index (asc or desc)
    """

    columns: list[str]
    name: str | None = None
    unique: bool = False
    order: Literal["asc", "desc"] | None = None

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> IndexDef:
        """Create IndexDef from dictionary specification."""
        columns = spec.get("columns", [])
        if isinstance(columns, str):
            columns = [columns]
        return cls(
            columns=columns,
            name=spec.get("name"),
            unique=spec.get("unique", False),
            order=spec.get("order"),
        )

    def to_sql(self, table_name: str, engine: DatabaseEngine) -> str:
        """Generate CREATE INDEX IF NOT EXISTS SQL."""
        # Generate index name if not provided
        idx_name = self.name or f"idx_{table_name}_{'_'.join(self.columns)}"

        unique_clause = "UNIQUE " if self.unique else ""

        # Format column list with optional order
        col_list = []
        for col in self.columns:
            col_sql = f'"{col}"'
            if self.order:
                col_sql += f" {self.order.upper()}"
            col_list.append(col_sql)

        return (
            f'CREATE {unique_clause}INDEX IF NOT EXISTS "{idx_name}" '
            f'ON "{table_name}" ({", ".join(col_list)})'
        )


@dataclass
class ModelSchema:
    """Complete table schema definition.

    Generates DDL for creating tables and indexes across different database engines.

    Attributes:
        table: Table name
        columns: List of column definitions
        indexes: List of index definitions
    """

    table: str
    columns: list[ColumnDef] = field(default_factory=list)
    indexes: list[IndexDef] = field(default_factory=list)

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> ModelSchema:
        """Create ModelSchema from dictionary specification.

        Args:
            spec: Model specification dictionary with:
                - table: Table name (required)
                - columns: Dict of column_name -> column_spec
                - indexes: List of index specifications

        Returns:
            ModelSchema instance

        Raises:
            ValueError: If table name is missing or columns is empty
        """
        table = spec.get("table")
        if not table:
            raise ValueError("Model must have a 'table' name")

        columns_spec = spec.get("columns", {})
        if not columns_spec:
            raise ValueError("Model must have at least one column")

        columns = [ColumnDef.from_dict(name, col_spec) for name, col_spec in columns_spec.items()]

        indexes_spec = spec.get("indexes", [])
        indexes = [IndexDef.from_dict(idx_spec) for idx_spec in indexes_spec]

        return cls(table=table, columns=columns, indexes=indexes)

    def get_primary_key(self) -> ColumnDef | None:
        """Get the primary key column, if any."""
        for col in self.columns:
            if col.primary:
                return col
        return None

    def get_auto_columns(self, auto_type: str) -> list[ColumnDef]:
        """Get columns with the specified auto-generation type."""
        return [col for col in self.columns if col.auto == auto_type]

    def to_create_sql(self, engine: DatabaseEngine) -> str:
        """Generate CREATE TABLE IF NOT EXISTS SQL.

        Args:
            engine: Target database engine

        Returns:
            Complete CREATE TABLE statement
        """
        column_defs = [col.to_sql(engine) for col in self.columns]

        # Collect foreign key constraints
        fk_constraints = []
        for col in self.columns:
            fk_sql = col.foreign_key_sql(engine)
            if fk_sql:
                fk_constraints.append(fk_sql)

        all_defs = column_defs + fk_constraints
        columns_sql = ",\n    ".join(all_defs)

        return f'CREATE TABLE IF NOT EXISTS "{self.table}" (\n    {columns_sql}\n)'

    def to_index_sql(self, engine: DatabaseEngine) -> list[str]:
        """Generate CREATE INDEX IF NOT EXISTS statements.

        Args:
            engine: Target database engine

        Returns:
            List of CREATE INDEX statements
        """
        return [idx.to_sql(self.table, engine) for idx in self.indexes]

    def to_full_schema_sql(self, engine: DatabaseEngine) -> str:
        """Generate complete schema SQL (table + indexes).

        Args:
            engine: Target database engine

        Returns:
            Complete DDL script
        """
        statements = [self.to_create_sql(engine)]
        statements.extend(self.to_index_sql(engine))
        return ";\n".join(statements) + ";"

    def column_names(self) -> list[str]:
        """Get list of all column names."""
        return [col.name for col in self.columns]
