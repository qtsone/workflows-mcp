"""Parameter placeholder normalization for cross-dialect SQL compatibility.

This module converts SQL parameter placeholders between different formats
to enable writing portable SQL that works across SQLite, PostgreSQL, and MariaDB.

Supported placeholder formats:
    - ? (qmark) - SQLite native format
    - $1, $2, ... (numeric) - PostgreSQL native format
    - %s (format) - MariaDB/MySQL native format
    - :name (named) - SQLite native, converted for others

The converter normalizes all formats to the target dialect's native format.
"""

from __future__ import annotations

import re
from typing import Any

from .backend import DatabaseEngine

# Regex patterns for detecting placeholder formats
QMARK_PATTERN = re.compile(r"(?<![:%$])\?(?!\?)")  # ? but not ?? or :? or %? or $?
NUMERIC_PATTERN = re.compile(r"\$(\d+)")  # $1, $2, etc.
FORMAT_PATTERN = re.compile(r"(?<!%)%s(?!s)")  # %s but not %%s or %ss
NAMED_PATTERN = re.compile(r":([a-zA-Z_][a-zA-Z0-9_]*)")  # :name


class ParamConverter:
    """Converts SQL parameter placeholders between dialect formats.

    This class provides methods to detect the current placeholder format
    and convert SQL statements to use the appropriate format for the
    target database dialect.

    Example:
        converter = ParamConverter(DatabaseEngine.POSTGRESQL)
        sql = "SELECT * FROM users WHERE id = ? AND status = ?"
        converted_sql = converter.convert(sql)
        # Result: "SELECT * FROM users WHERE id = $1 AND status = $2"
    """

    def __init__(self, target_dialect: DatabaseEngine):
        """Initialize converter for target dialect.

        Args:
            target_dialect: The database dialect to convert placeholders to
        """
        self.target_dialect = target_dialect

    def convert(self, sql: str) -> str:
        """Convert SQL placeholders to target dialect format.

        Args:
            sql: SQL statement with any placeholder format

        Returns:
            SQL statement with placeholders in target dialect format
        """
        source_format = self._detect_format(sql)

        if source_format == "none":
            return sql

        if source_format == self._target_format():
            return sql

        # Convert to intermediate (indexed list) then to target
        placeholders = self._extract_placeholders(sql, source_format)

        return self._replace_with_target(sql, source_format, len(placeholders))

    def convert_params(
        self, params: tuple[Any, ...] | list[Any] | dict[str, Any] | None
    ) -> tuple[Any, ...] | list[Any] | dict[str, Any] | None:
        """Convert parameters to format expected by target dialect.

        Most dialects accept tuples/lists directly. Named parameters may
        need conversion for some backends.

        Args:
            params: Query parameters in any format

        Returns:
            Parameters in format expected by target dialect
        """
        if params is None:
            return None

        if isinstance(params, dict):
            # Named parameters - convert to positional for PostgreSQL
            if self.target_dialect == DatabaseEngine.POSTGRESQL:
                # PostgreSQL uses $1, $2 which are positional
                # Named params need to be extracted in order
                return tuple(params.values())
            elif self.target_dialect == DatabaseEngine.MARIADB:
                # MariaDB supports %(name)s format
                return params
            else:
                # SQLite supports :name format natively
                return params

        if isinstance(params, list):
            return tuple(params)

        return params

    def _detect_format(self, sql: str) -> str:
        """Detect the placeholder format used in SQL.

        Returns:
            One of: "qmark", "numeric", "format", "named", "none"
        """
        if QMARK_PATTERN.search(sql):
            return "qmark"
        if NUMERIC_PATTERN.search(sql):
            return "numeric"
        if FORMAT_PATTERN.search(sql):
            return "format"
        if NAMED_PATTERN.search(sql):
            return "named"
        return "none"

    def _target_format(self) -> str:
        """Get the native placeholder format for target dialect."""
        if self.target_dialect == DatabaseEngine.SQLITE:
            return "qmark"
        elif self.target_dialect == DatabaseEngine.POSTGRESQL:
            return "numeric"
        elif self.target_dialect == DatabaseEngine.MARIADB:
            return "format"
        return "qmark"

    def _extract_placeholders(self, sql: str, format_type: str) -> list[str | int | None]:
        """Extract placeholder positions/names from SQL.

        Args:
            sql: SQL statement
            format_type: Detected placeholder format

        Returns:
            List of placeholder identifiers (indices or names)
        """
        if format_type == "qmark":
            return [None] * len(QMARK_PATTERN.findall(sql))
        elif format_type == "numeric":
            matches = NUMERIC_PATTERN.findall(sql)
            return [int(m) for m in matches]
        elif format_type == "format":
            return [None] * len(FORMAT_PATTERN.findall(sql))
        elif format_type == "named":
            return NAMED_PATTERN.findall(sql)
        return []

    def _replace_with_target(self, sql: str, source_format: str, count: int) -> str:
        """Replace source format placeholders with target format.

        Args:
            sql: Original SQL
            source_format: Current placeholder format
            count: Number of placeholders

        Returns:
            SQL with converted placeholders
        """
        target = self._target_format()

        if target == "qmark":
            return self._convert_to_qmark(sql, source_format)
        elif target == "numeric":
            return self._convert_to_numeric(sql, source_format)
        elif target == "format":
            return self._convert_to_format(sql, source_format)

        return sql

    def _convert_to_qmark(self, sql: str, source_format: str) -> str:
        """Convert any format to SQLite ? placeholders."""
        if source_format == "numeric":
            return NUMERIC_PATTERN.sub("?", sql)
        elif source_format == "format":
            return FORMAT_PATTERN.sub("?", sql)
        elif source_format == "named":
            # Keep named format for SQLite - it supports :name natively
            return sql
        return sql

    def _convert_to_numeric(self, sql: str, source_format: str) -> str:
        """Convert any format to PostgreSQL $1, $2 placeholders."""
        if source_format == "qmark":
            counter = [0]

            def replace(match: re.Match[str]) -> str:
                counter[0] += 1
                return f"${counter[0]}"

            return QMARK_PATTERN.sub(replace, sql)

        elif source_format == "format":
            counter = [0]

            def replace(match: re.Match[str]) -> str:
                counter[0] += 1
                return f"${counter[0]}"

            return FORMAT_PATTERN.sub(replace, sql)

        elif source_format == "named":
            # Named to numeric requires knowing the order
            # Extract names in order of appearance
            names = NAMED_PATTERN.findall(sql)
            seen: dict[str, int] = {}
            result = sql

            for name in names:
                if name not in seen:
                    seen[name] = len(seen) + 1
                # Replace only first occurrence each time
                result = result.replace(f":{name}", f"${seen[name]}", 1)

            return result

        return sql

    def _convert_to_format(self, sql: str, source_format: str) -> str:
        """Convert any format to MariaDB %s placeholders."""
        if source_format == "qmark":
            return QMARK_PATTERN.sub("%s", sql)
        elif source_format == "numeric":
            return NUMERIC_PATTERN.sub("%s", sql)
        elif source_format == "named":
            # Convert :name to %(name)s for MariaDB
            return NAMED_PATTERN.sub(r"%(\1)s", sql)
        return sql


def convert_sql_for_dialect(sql: str, dialect: DatabaseEngine) -> str:
    """Convenience function to convert SQL placeholders for a dialect.

    Args:
        sql: SQL statement with any placeholder format
        dialect: Target database dialect

    Returns:
        SQL with placeholders converted to dialect's native format

    Example:
        >>> convert_sql_for_dialect("SELECT * FROM users WHERE id = ?", DatabaseEngine.POSTGRESQL)
        "SELECT * FROM users WHERE id = $1"
    """
    return ParamConverter(dialect).convert(sql)
