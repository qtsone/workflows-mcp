"""PostgreSQL probe for readiness checks.

This module provides a ``PostgresProbe`` that implements the ``Probe``
protocol defined in ``readiness.py``. It performs a bounded suitability
check against the configured PostgreSQL DSN and reports blockers on failure.

Suitability checks (spec §7A.1 / addendum §7A):
1. Version gate: server must report version >= 14.0 (140000).
2. Extension gate: when ``require_pgvector`` is True, the ``vector``
   extension must be installed.
3. Read/write gate: a bounded temp-table insert + select must succeed.

Timeout and retry policy (spec §7A.2):
- Per-attempt timeout: 3 s.
- Retry count: 2 additional attempts after the first failure.
- Jittered backoff between retries (max 0.5 s).

Blocker identifiers returned in ``check()`` output:
- ``postgresql_dsn_missing``: no DSN configured.
- ``postgresql_connectivity``: could not reach server (network / auth).
- ``postgresql_version_unsupported``: server version < 140000.
- ``pgvector_missing``: ``vector`` extension absent (when required).
- ``postgresql_readwrite_failed``: bounded read/write probe failed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_SECONDS = 3.0
_PROBE_RETRIES = 2
_JITTER_MAX = 0.5
_MIN_VERSION_NUM = 140000

# Type alias for an async callable that returns a connection-like object.
ConnectionFactory = Callable[[], Awaitable[Any]]


async def _asyncpg_connect(dsn: str) -> Any:
    """Default connection factory: connects via asyncpg."""
    try:
        import asyncpg  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "asyncpg is required for PostgreSQL probing. "
            "Install it with: uv add asyncpg"
        ) from None
    return await asyncpg.connect(dsn)


@dataclass
class PostgresProbe:
    """Suitability probe for an external PostgreSQL server.

    Parameters
    ----------
    dsn:
        libpq-compatible connection string, e.g.
        ``postgresql://user:pass@host/dbname``.
    timeout:
        Per-attempt connection timeout in seconds.
    retries:
        Number of additional attempts after the first failure.
    require_pgvector:
        When ``True``, the ``vector`` extension must be present.
    _connection_factory:
        Async callable that returns a connection-like object. Defaults to
        the real asyncpg connect. Can be replaced in tests to inject fakes
        without requiring a live PostgreSQL server.
    """

    dsn: str
    timeout: float = _PROBE_TIMEOUT_SECONDS
    retries: int = _PROBE_RETRIES
    require_pgvector: bool = False
    _connection_factory: ConnectionFactory | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> PostgresProbe:
        """Construct a probe from environment variables.

        ``WORKFLOWS_POSTGRES_DSN``: libpq DSN (required for connectivity).
        ``WORKFLOWS_POSTGRES_REQUIRE_PGVECTOR=true``: enforce pgvector check.

        If ``WORKFLOWS_POSTGRES_DSN`` is absent the DSN is set to an empty
        string, which causes every probe check to fail with a clear blocker
        message rather than raising at construction time.
        """
        dsn = os.getenv("WORKFLOWS_POSTGRES_DSN", "")
        require_pgvector = os.getenv("WORKFLOWS_POSTGRES_REQUIRE_PGVECTOR", "").lower() == "true"
        return cls(dsn=dsn, require_pgvector=require_pgvector)

    async def check(self) -> tuple[bool, list[str]]:
        """Attempt a suitability check against PostgreSQL and return ``(ok, blockers)``.

        Returns
        -------
        ok:
            ``True`` if all suitability checks pass within the configured timeout.
        blockers:
            List of blocker identifiers when ``ok`` is ``False``.
        """
        if not self.dsn:
            return False, ["postgresql_dsn_missing"]

        last_error: Exception | None = None
        for attempt in range(1 + self.retries):
            try:
                result = await asyncio.wait_for(
                    self._run_suitability_checks(),
                    timeout=self.timeout,
                )
                return result
            except TimeoutError:
                last_error = TimeoutError(
                    f"PostgreSQL probe timed out after {self.timeout}s"
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc

            if attempt < self.retries:
                jitter = random.uniform(0, _JITTER_MAX)  # noqa: S311
                await asyncio.sleep(jitter)

        logger.warning("PostgreSQL probe failed: %s", last_error)
        return False, ["postgresql_connectivity"]

    async def _run_suitability_checks(self) -> tuple[bool, list[str]]:
        """Run all suitability checks against an open connection.

        Returns ``(ok, blockers)`` where blockers is empty on success.
        The connection is always closed on exit.
        """
        factory = self._connection_factory or (lambda: _asyncpg_connect(self.dsn))
        conn = await factory()
        blockers: list[str] = []
        try:
            # --- version check ---
            version_str = await conn.fetchval("SHOW server_version_num")
            try:
                version_num = int(version_str)
            except (TypeError, ValueError):
                version_num = 0
            if version_num < _MIN_VERSION_NUM:
                blockers.append("postgresql_version_unsupported")

            # --- extension check ---
            if self.require_pgvector:
                has_vector = await conn.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                )
                if not has_vector:
                    blockers.append("pgvector_missing")

            # --- bounded read/write check ---
            if not blockers:
                # Only run read/write probe when prior checks passed to avoid
                # noise — callers need actionable, non-redundant blocker lists.
                try:
                    await conn.execute(
                        "CREATE TEMP TABLE IF NOT EXISTS _wf_readiness_probe (id int)"
                    )
                    await conn.execute(
                        "INSERT INTO _wf_readiness_probe (id) VALUES (1)"
                    )
                    await conn.fetchval(
                        "SELECT COUNT(*) FROM _wf_readiness_probe"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("PostgreSQL read/write probe failed: %s", exc)
                    blockers.append("postgresql_readwrite_failed")
        finally:
            await conn.close()

        return len(blockers) == 0, blockers


def _load_saved_postgres_dsn(base_dir: Path) -> str | None:
    from .metadata.db import connect_metadata_db
    from .metadata.repos.postgres_repo import SQLitePostgresSettingsRepository

    db_path = base_dir / "server.db"
    if not db_path.exists():
        return None

    try:
        conn = connect_metadata_db(db_path)
    except Exception:  # noqa: BLE001
        return None
    try:
        try:
            return SQLitePostgresSettingsRepository(
                conn=conn,
                key_path=base_dir / "secrets.key",
            ).load_dsn()
        except Exception:  # noqa: BLE001
            return None
    finally:
        conn.close()


@dataclass
class ConfiguredPostgresProbe:
    """PostgreSQL probe that reads the saved admin database profile on each check."""

    base_dir: Path
    timeout: float = _PROBE_TIMEOUT_SECONDS
    retries: int = _PROBE_RETRIES
    require_pgvector: bool = True
    _connection_factory: ConnectionFactory | None = field(default=None, repr=False)

    async def check(self) -> tuple[bool, list[str]]:
        """Check PostgreSQL using SQLite admin settings, falling back to env DSN."""
        saved_dsn = _load_saved_postgres_dsn(self.base_dir)
        env_dsn = os.getenv("WORKFLOWS_POSTGRES_DSN") or ""
        dsn = saved_dsn or env_dsn
        env_requires_pgvector = (
            os.getenv("WORKFLOWS_POSTGRES_REQUIRE_PGVECTOR", "").lower() == "true"
        )
        probe = PostgresProbe(
            dsn=dsn,
            timeout=self.timeout,
            retries=self.retries,
            require_pgvector=self.require_pgvector or env_requires_pgvector,
            _connection_factory=self._connection_factory,
        )
        return await probe.check()
