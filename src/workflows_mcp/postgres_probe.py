"""PostgreSQL probe for readiness checks.

This module provides a ``PostgresProbe`` that implements the ``Probe``
protocol defined in ``readiness.py``. It performs a bounded connectivity
check against the configured PostgreSQL DSN and reports blockers on failure.

Design notes (spec section 11):
- Timeout: 3 s per attempt, 2 retries, jittered backoff.
- The probe validates only that the server is reachable and accepts the
  connection; schema or extension checks can be layered on top later.
- Never embeds or provisions a PostgreSQL instance.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_SECONDS = 3.0
_PROBE_RETRIES = 2
_JITTER_MAX = 0.5


@dataclass(slots=True)
class PostgresProbe:
    """Connectivity probe for an external PostgreSQL server.

    Parameters
    ----------
    dsn:
        libpq-compatible connection string, e.g.
        ``postgresql://user:pass@host/dbname``.
    timeout:
        Per-attempt connection timeout in seconds.
    retries:
        Number of additional attempts after the first failure.
    """

    dsn: str
    timeout: float = _PROBE_TIMEOUT_SECONDS
    retries: int = _PROBE_RETRIES

    @classmethod
    def from_env(cls) -> PostgresProbe:
        """Construct a probe from ``WORKFLOWS_POSTGRES_DSN`` environment variable.

        If the variable is absent the DSN is set to an empty string, which
        will cause every probe check to fail with a clear blocker message
        rather than raising at construction time.
        """
        dsn = os.getenv("WORKFLOWS_POSTGRES_DSN", "")
        return cls(dsn=dsn)

    async def check(self) -> tuple[bool, list[str]]:
        """Attempt a connection to PostgreSQL and return ``(ok, blockers)``.

        Returns
        -------
        ok:
            ``True`` if the server is reachable within the configured timeout.
        blockers:
            List of blocker identifiers when ``ok`` is ``False``.
        """
        if not self.dsn:
            return False, ["postgresql_dsn_missing"]

        last_error: Exception | None = None
        for attempt in range(1 + self.retries):
            try:
                ok = await asyncio.wait_for(
                    self._connect_once(),
                    timeout=self.timeout,
                )
                if ok:
                    return True, []
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

    async def _connect_once(self) -> bool:
        """Open and immediately close a single asyncpg connection.

        Raises on any connection failure so the retry loop can handle it.
        """
        try:
            import asyncpg  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "asyncpg is required for PostgreSQL probing. "
                "Install it with: uv add asyncpg"
            ) from None

        conn = await asyncpg.connect(self.dsn)
        await conn.close()
        return True
