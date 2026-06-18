"""Single SQL entry point binding raw-SQL execution and transaction policy.

``SqlSession`` wraps a typed :class:`DatabaseBackend` so every raw-SQL site and
transaction in a service routes through one object instead of touching the
backend directly. ``query``/``execute`` forward to the backend; ``transaction``
owns the begin/commit/rollback policy in one place (commit on clean exit,
rollback on any exception). Holding the typed backend here also keeps the
backend's concrete type out of the consuming service's surface.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from .backend import DatabaseBackend, Params, QueryResult


class SqlSession:
    """Typed SQL surface: forwards queries and scopes transactions."""

    def __init__(self, backend: DatabaseBackend) -> None:
        self._backend = backend

    async def query(self, sql: str, params: Params = None) -> QueryResult:
        """Execute a SELECT (or RETURNING) statement and return its rows."""
        return await self._backend.query(sql, params)

    async def execute(self, sql: str, params: Params = None) -> QueryResult:
        """Execute an INSERT/UPDATE/DELETE statement."""
        return await self._backend.execute(sql, params)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Run the block in a transaction: commit on success, roll back on error.

        Any exception leaving the block (including a service-local control-flow
        sentinel) triggers a rollback and propagates, so the caller decides
        whether to surface or absorb it. A failure of ``begin_transaction``
        itself also rolls back, keeping cleanup unconditional.
        """
        try:
            await self._backend.begin_transaction()
            yield
        except Exception:
            await self._backend.rollback()
            raise
        else:
            await self._backend.commit()
