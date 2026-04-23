"""Tests for the readiness service and PostgreSQL probe interface.

Covers the three readiness states defined in the spec (section 9.1):
- unconfigured: ~/.workflows/ or llm-config.yml missing
- partially_configured: config artifacts exist but DB check fails
- ready: config present and DB probe succeeds

Also covers suitability checks (§7A.1) and resilience behavior (§7A.3):
- version check, extension check, read-write check in PostgresProbe
- outage-after-ready transitions to partially_configured
- recovery-to-ready without restart
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workflows_mcp.http_models import ReadinessState
from workflows_mcp.postgres_probe import PostgresProbe
from workflows_mcp.readiness import ReadinessService


class FakeProbe:
    """Test double for the PostgreSQL probe protocol.

    ``ok`` is mutable so tests can flip it to simulate outage/recovery.
    ``blockers_on_fail`` controls which blockers are returned.
    """

    def __init__(self, ok: bool, blockers_on_fail: list[str] | None = None) -> None:
        self.ok = ok
        self.blockers_on_fail = blockers_on_fail or ["postgresql_connectivity"]

    async def check(self) -> tuple[bool, list[str]]:
        if self.ok:
            return True, []
        return False, list(self.blockers_on_fail)


# ---------------------------------------------------------------------------
# Existing state-transition tests (preserved)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_workflows_dir_is_unconfigured(tmp_path: Path) -> None:
    """When ~/.workflows/ does not exist, state must be UNCONFIGURED."""
    service = ReadinessService(base_dir=tmp_path / ".workflows", probe=FakeProbe(ok=True))
    report = await service.evaluate()
    assert report.state == ReadinessState.UNCONFIGURED


@pytest.mark.asyncio
async def test_missing_llm_config_is_unconfigured(tmp_path: Path) -> None:
    """When ~/.workflows/ exists but llm-config.yml is absent, state must be UNCONFIGURED."""
    base_dir = tmp_path / ".workflows"
    base_dir.mkdir()
    # Do NOT create llm-config.yml
    service = ReadinessService(base_dir=base_dir, probe=FakeProbe(ok=True))
    report = await service.evaluate()
    assert report.state == ReadinessState.UNCONFIGURED


@pytest.mark.asyncio
async def test_invalid_db_is_partially_configured(tmp_path: Path) -> None:
    """Config artifacts exist but DB probe fails -> PARTIALLY_CONFIGURED."""
    base_dir = tmp_path / ".workflows"
    base_dir.mkdir()
    (base_dir / "llm-config.yml").write_text("profiles: []\n")
    service = ReadinessService(base_dir=base_dir, probe=FakeProbe(ok=False))
    report = await service.evaluate()
    assert report.state == ReadinessState.PARTIALLY_CONFIGURED
    assert "postgresql_connectivity" in report.blockers


@pytest.mark.asyncio
async def test_valid_config_and_db_is_ready(tmp_path: Path) -> None:
    """Config artifacts exist and DB probe succeeds -> READY."""
    base_dir = tmp_path / ".workflows"
    base_dir.mkdir()
    (base_dir / "llm-config.yml").write_text("profiles: []\n")
    service = ReadinessService(base_dir=base_dir, probe=FakeProbe(ok=True))
    report = await service.evaluate()
    assert report.state == ReadinessState.READY
    assert report.blockers == []


@pytest.mark.asyncio
async def test_unconfigured_report_includes_blocker(tmp_path: Path) -> None:
    """UNCONFIGURED report must include at least one blocker identifier."""
    service = ReadinessService(base_dir=tmp_path / ".workflows", probe=FakeProbe(ok=True))
    report = await service.evaluate()
    assert report.state == ReadinessState.UNCONFIGURED
    assert len(report.blockers) >= 1


# ---------------------------------------------------------------------------
# §7A.1 Suitability check tests (new — must fail before implementation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_postgres_probe_missing_dsn_reports_blocker() -> None:
    """PostgresProbe with no DSN must report postgresql_dsn_missing immediately."""
    probe = PostgresProbe(dsn="")
    ok, blockers = await probe.check()
    assert ok is False
    assert "postgresql_dsn_missing" in blockers


@pytest.mark.asyncio
async def test_postgres_probe_from_env_without_dsn_reports_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PostgresProbe.from_env() without WORKFLOWS_POSTGRES_DSN must report dsn missing."""
    monkeypatch.delenv("WORKFLOWS_POSTGRES_DSN", raising=False)
    probe = PostgresProbe.from_env()
    ok, blockers = await probe.check()
    assert ok is False
    assert "postgresql_dsn_missing" in blockers


@pytest.mark.asyncio
async def test_postgres_probe_unsupported_version_reports_blocker() -> None:
    """When the server reports version < 140000, probe must report version_unsupported."""

    class FakeConn:
        async def fetchval(self, query: str, *args: object) -> object:
            if "server_version_num" in query:
                return "130005"  # PG 13 — below minimum
            if "pg_extension" in query:
                return True
            if "readiness_probe" in query:
                return 1
            return None

        async def execute(self, query: str, *args: object) -> None:
            pass

        async def close(self) -> None:
            pass

    probe = PostgresProbe(
        dsn="postgresql://fake/db",
        _connection_factory=lambda: _async_return(FakeConn()),
    )
    ok, blockers = await probe.check()
    assert ok is False
    assert "postgresql_version_unsupported" in blockers


@pytest.mark.asyncio
async def test_postgres_probe_missing_pgvector_reports_blocker() -> None:
    """When pgvector is required but absent, probe must report pgvector_missing."""

    class FakeConn:
        async def fetchval(self, query: str, *args: object) -> object:
            if "server_version_num" in query:
                return "140005"  # PG 14 — supported
            if "pg_extension" in query:
                return False  # extension absent
            if "readiness_probe" in query:
                return 1
            return None

        async def execute(self, query: str, *args: object) -> None:
            pass

        async def close(self) -> None:
            pass

    probe = PostgresProbe(
        dsn="postgresql://fake/db",
        require_pgvector=True,
        _connection_factory=lambda: _async_return(FakeConn()),
    )
    ok, blockers = await probe.check()
    assert ok is False
    assert "pgvector_missing" in blockers


@pytest.mark.asyncio
async def test_postgres_probe_read_write_failure_reports_blocker() -> None:
    """When the bounded read/write check fails, probe must report postgresql_readwrite_failed."""

    class FakeConn:
        async def fetchval(self, query: str, *args: object) -> object:
            if "server_version_num" in query:
                return "140005"
            if "pg_extension" in query:
                return True
            if "readiness_probe" in query:
                raise RuntimeError("permission denied")
            return None

        async def execute(self, query: str, *args: object) -> None:
            if "readiness_probe" in query:
                raise RuntimeError("permission denied")

        async def close(self) -> None:
            pass

    probe = PostgresProbe(
        dsn="postgresql://fake/db",
        _connection_factory=lambda: _async_return(FakeConn()),
    )
    ok, blockers = await probe.check()
    assert ok is False
    assert "postgresql_readwrite_failed" in blockers


@pytest.mark.asyncio
async def test_postgres_probe_all_checks_pass() -> None:
    """When version, extension, and read/write checks all pass, probe returns ok=True."""

    class FakeConn:
        async def fetchval(self, query: str, *args: object) -> object:
            if "server_version_num" in query:
                return "150000"
            if "pg_extension" in query:
                return True
            if "readiness_probe" in query:
                return 1
            return None

        async def execute(self, query: str, *args: object) -> None:
            pass

        async def close(self) -> None:
            pass

    probe = PostgresProbe(
        dsn="postgresql://fake/db",
        require_pgvector=True,
        _connection_factory=lambda: _async_return(FakeConn()),
    )
    ok, blockers = await probe.check()
    assert ok is True
    assert blockers == []


# ---------------------------------------------------------------------------
# §7A.3 Resilience behavior tests (new — must fail before implementation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_readiness_transitions_to_partially_configured_on_outage(tmp_path: Path) -> None:
    """When PostgreSQL goes down after reaching ready, service must return PARTIALLY_CONFIGURED.

    The ReadinessService is stateless (evaluates on demand), so this tests
    that a subsequent evaluate() with a failing probe returns the right state.
    """
    base_dir = tmp_path / ".workflows"
    base_dir.mkdir()
    (base_dir / "llm-config.yml").write_text("profiles: []\n")

    probe = FakeProbe(ok=True)
    service = ReadinessService(base_dir=base_dir, probe=probe)

    # First evaluation: ready
    report = await service.evaluate()
    assert report.state == ReadinessState.READY

    # Simulate PostgreSQL outage
    probe.ok = False
    probe.blockers_on_fail = ["postgresql_connectivity"]

    # Second evaluation: must degrade without restart
    report = await service.evaluate()
    assert report.state == ReadinessState.PARTIALLY_CONFIGURED
    assert "postgresql_connectivity" in report.blockers


@pytest.mark.asyncio
async def test_readiness_recovers_to_ready_without_restart(tmp_path: Path) -> None:
    """When PostgreSQL recovers, service must return READY without requiring restart.

    Verifies the full outage -> recovery cycle using the same service instance.
    """
    base_dir = tmp_path / ".workflows"
    base_dir.mkdir()
    (base_dir / "llm-config.yml").write_text("profiles: []\n")

    probe = FakeProbe(ok=True)
    service = ReadinessService(base_dir=base_dir, probe=probe)

    # Reach READY
    report = await service.evaluate()
    assert report.state == ReadinessState.READY

    # Simulate outage
    probe.ok = False
    report = await service.evaluate()
    assert report.state == ReadinessState.PARTIALLY_CONFIGURED

    # Simulate recovery
    probe.ok = True
    report = await service.evaluate()
    assert report.state == ReadinessState.READY
    assert report.blockers == []


@pytest.mark.asyncio
async def test_suitability_failure_blocker_is_surfaced_in_readiness_report(
    tmp_path: Path,
) -> None:
    """Suitability blockers (e.g. version, extension) must appear verbatim in report.blockers."""
    base_dir = tmp_path / ".workflows"
    base_dir.mkdir()
    (base_dir / "llm-config.yml").write_text("profiles: []\n")

    probe = FakeProbe(ok=False, blockers_on_fail=["postgresql_version_unsupported"])
    service = ReadinessService(base_dir=base_dir, probe=probe)

    report = await service.evaluate()
    assert report.state == ReadinessState.PARTIALLY_CONFIGURED
    assert "postgresql_version_unsupported" in report.blockers


# ---------------------------------------------------------------------------
# §7A.2 Timeout / retry policy tests (new)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_postgres_probe_timeout_and_retry_uses_configured_values() -> None:
    """PostgresProbe must use configured timeout and retries, not hardcoded values."""
    probe = PostgresProbe(dsn="postgresql://fake/db", timeout=1.0, retries=1)
    assert probe.timeout == 1.0
    assert probe.retries == 1


@pytest.mark.asyncio
async def test_postgres_probe_default_timeout_and_retries() -> None:
    """Default probe must have 3 s timeout and 2 retries as per spec §7A.2."""
    probe = PostgresProbe(dsn="postgresql://fake/db")
    assert probe.timeout == 3.0
    assert probe.retries == 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_return(value: object) -> object:
    """Coroutine wrapper that simply returns a value, for monkey-patching async methods."""
    return value
