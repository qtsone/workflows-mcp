"""Tests for the readiness service and PostgreSQL probe interface.

Covers the three readiness states defined in the spec (section 9.1):
- unconfigured: ~/.workflows/ or llm-config.yml missing
- partially_configured: config artifacts exist but DB check fails
- ready: config present and DB probe succeeds
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workflows_mcp.http_models import ReadinessState
from workflows_mcp.readiness import ReadinessService


class FakeProbe:
    """Test double for the PostgreSQL probe protocol."""

    def __init__(self, ok: bool) -> None:
        self.ok = ok

    async def check(self) -> tuple[bool, list[str]]:
        if self.ok:
            return True, []
        return False, ["postgresql_connectivity"]


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
